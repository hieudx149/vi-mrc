from torch.nn import CrossEntropyLoss
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaModel
from .module import HighwayEncoder, LSTM_Modeling, AnswerableClassifier, SpanPredictor


class Joint_XLMRobertaForQuestionAnswering(RobertaPreTrainedModel):
    def __init__(self, config, args):
        super(Joint_XLMRobertaForQuestionAnswering, self).__init__(config)
        self.args = args
        self.xlm_roberta = XLMRobertaModel(config)
        self.highway_net = HighwayEncoder(num_layers=2, hidden_size=config.hidden_size)
        self.lstm = LSTM_Modeling(input_size=config.hidden_size, hidden_size=self.args.lstm_hidden_size, num_layers=2)
        self.answerable_classifier = AnswerableClassifier(input_size=config.hidden_size, output_size=2)
        # input_size = lstm_hidden_size*2 for bidirectional lstm
        self.span_predictor = SpanPredictor(input_size=self.args.lstm_hidden_size * 2, output_size=1)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, start_positions=None,
                end_positions=None, is_impossibles=None):
        outputs = self.xlm_roberta(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids)

        # compute span logits
        sequence_outputs = outputs[0]
        if self.args.use_highway:
            sequence_outputs = self.highway_net(sequence_outputs)
        lstm_seq_outputs = self.lstm(sequence_outputs)
        start_logits, end_logits = self.span_predictor(lstm_seq_outputs)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        has_logits = None
        # compute answerable logits
        if self.args.has_loss_coef > 0:
            pooler_output = outputs[1]
            has_logits = self.answerable_classifier(pooler_output)
            outputs = (start_logits, end_logits, has_logits,) + outputs[2:]
        else:
            outputs = (start_logits, end_logits,) + outputs[2:]

        if start_positions is not None and end_positions is not None:
            # sometimes start_positions or end_positions are longer than sequence input lengths
            # should limit index == max_seq_len
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            is_impossibles.clamp_(0, ignored_index)
            # compute loss
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = 0
            if has_logits is not None:
                has_loss = loss_fct(has_logits, is_impossibles)
                total_loss = self.args.has_loss_coef * has_loss
            total_loss += (1 - self.args.has_loss_coef) * (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, choice_logits, (hidden_states), (attentions)
