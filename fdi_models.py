# fdi_models.py

import torch
from torch import nn


class BiLSTMFDIDetector(nn.Module):
    """
    Binary sequence classifier:
      input:  (batch, seq_len, input_dim)
      output: logits of shape (batch,) -> use BCEWithLogitsLoss
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),  # binary logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_dim)
        returns: logits of shape (batch,)
        """
        lstm_out, (h_n, c_n) = self.lstm(x)
        # h_n: (num_layers * num_directions, batch, hidden_dim)

        if self.bidirectional:
            # last layer's forward and backward hidden states
            h_forward = h_n[-2, :, :]  # (batch, hidden_dim)
            h_backward = h_n[-1, :, :] # (batch, hidden_dim)
            h_cat = torch.cat([h_forward, h_backward], dim=1)
        else:
            h_cat = h_n[-1, :, :]  # (batch, hidden_dim)

        logits = self.classifier(h_cat).squeeze(1)  # (batch,)
        return logits


def count_parameters(model: nn.Module) -> int:
    """
    Utility: count trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)