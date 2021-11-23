import torch
import torch.nn as nn
import torch.nn.functional as F


class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.
    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """

    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


class LSTM_Modeling(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_prob=0.):
        super(LSTM_Modeling, self).__init__()
        self.drop_prob = dropout_prob
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=self.drop_prob if num_layers > 1 else 0.)

    def forward(self, pretrained_outputs):
        outputs, _ = self.lstm(pretrained_outputs)
        return outputs


class AnswerableClassifier(nn.Module):
    def __init__(self, input_size, output_size=2, dropout_rate=0.0):
        super(AnswerableClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(in_features=input_size, out_features=output_size)

    def forward(self, cls_output):
        cls_output = self.dropout(cls_output)
        has_logits = self.linear(cls_output)

        return has_logits


class SpanPredictor(nn.Module):
    def __init__(self, input_size, output_size=1, dropout_rate=0.0):
        super(SpanPredictor, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.ffnn_start = nn.Linear(in_features=input_size, out_features=output_size)
        self.ffnn_end = nn.Linear(in_features=input_size + 1, out_features=output_size)

    def forward(self, lstm_outputs):
        lstm_outputs = self.dropout(lstm_outputs)

        start_logits = self.ffnn_start(lstm_outputs)
        combined = torch.cat((lstm_outputs, start_logits), dim=2)
        end_logits = self.ffnn_end(combined)

        return start_logits, end_logits

