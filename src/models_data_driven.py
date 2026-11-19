"""
Data-Driven Deep Learning Model for Battery State-of-Charge (SOC) Estimation
Journal of Energy Storage (Elsevier)

Exact architecture synchronized with Section 3.5 & Appendix C.2:
- Causal sliding window X_k = [I_{k-W:k}, V_{k-W:k}, T_{k-W:k}] in R^{W x 3}, W = 30
- 1-layer Bidirectional LSTM (D = 32 per direction -> 64-dim hidden state)
- Temporal self-attention mechanism Eqs. (36)-(38) + MLP regressor head
- Exact parameter count: 13,698 parameters (54.79 KB in float32)
"""

import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    """Temporal self-attention layer computing scalar alignment weights Eqs. (36)-(37)."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.attn_weights = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, rnn_outputs):
        scores = self.attn_weights(rnn_outputs)
        weights = torch.softmax(scores, dim=1)
        context_vector = torch.sum(weights * rnn_outputs, dim=1)
        return context_vector, weights


class BiLSTMAttentionSOC(nn.Module):
    """
    Causal Bidirectional LSTM with Temporal Self-Attention.
    Default (W=30, input_dim=3, hidden_dim=32, num_layers=1): 13,698 float32 parameters.
    """

    def __init__(self, input_dim=3, hidden_dim=32, num_layers=1, dropout=0.0):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)

        self.bilstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if self.num_layers > 1 else 0.0,
        )
        self.attention = TemporalAttention(self.hidden_dim * 2)
        self.regressor = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        lstm_out, _ = self.bilstm(x)
        context, _ = self.attention(lstm_out)
        return self.regressor(context).squeeze(-1)


if __name__ == "__main__":
    m = BiLSTMAttentionSOC()
    n_p = sum(p.numel() for p in m.parameters())
    print(f"BiLSTM-Attn params: {n_p} ({n_p * 4 / 1024:.2f} KB float32)")
