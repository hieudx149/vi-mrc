from torch.nn import CrossEntropyLoss
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaModel
from .module import AnswerableClassifier


class XLM_RobertaForSequenceClassification(RobertaPreTrainedModel):
    def __init__(self, config, num_labels):
        super(XLM_RobertaForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.xlm_roberta = XLMRobertaModel(config)
        self.answerable_classifier = AnswerableClassifier(input_size=config.hidden_size, output_size=self.num_labels)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.xlm_roberta(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids)

        # compute answerable logits
        pooler_output = outputs[1]  # CLS output
        has_logits = self.answerable_classifier(pooler_output)
        loss_fct = CrossEntropyLoss()
        has_loss = loss_fct(has_logits.view(-1, self.num_labels), labels.view(-1))
        outputs = (has_loss, has_logits,) + outputs[2:]

        return outputs  # (loss), logits, (hidden_states), (attentions)
