"""
Data-Driven Deep Learning Models for Battery State-of-Charge (SOC) Estimation
Journal of Energy Storage (Elsevier)

Architectures:
1. BiLSTM-Attention: Bidirectional Long Short-Term Memory with Temporal Self-Attention
2. Lightweight GRU: Efficient Recurrent Baseline for Embedded BMS
"""

import torch
import torch.nn as nn
import numpy as np


class TemporalAttention(nn.Module):
    """
    Self-attention layer for weighting critical dynamic transitions
    (e.g., rapid acceleration spikes, regenerative braking pulses).
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.attn_weights = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, rnn_outputs):
        # rnn_outputs: (batch_size, seq_len, hidden_dim)
        scores = self.attn_weights(rnn_outputs)  # (batch_size, seq_len, 1)
        weights = torch.softmax(scores, dim=1)  # Normalize across time steps
        context_vector = torch.sum(weights * rnn_outputs, dim=1)  # (batch_size, hidden_dim)
        return context_vector, weights


class BiLSTMAttentionSOC(nn.Module):
    """
    Bidirectional LSTM with Attention Mechanism for SOC Estimation.
    Input features per time step: [Current (A), Voltage (V), Temperature (C), dV/dt]
    """

    def __init__(self, input_dim=4, hidden_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # BiLSTM outputs hidden_dim * 2
        self.attention = TemporalAttention(hidden_dim * 2)

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Constrain SOC output to physical [0.0, 1.0] range
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.bilstm(x)
        context, attn_weights = self.attention(lstm_out)
        soc_pred = self.regressor(context)
        return soc_pred.squeeze(-1)


class GRUNetworkSOC(nn.Module):
    """
    Lightweight Gated Recurrent Unit (GRU) for SOC Estimation.
    Tailored for low-latency embedded MCU computation.
    """

    def __init__(self, input_dim=4, hidden_dim=32, num_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)
        # Use last time step
        last_step = gru_out[:, -1, :]
        soc_pred = self.fc(last_step)
        return soc_pred.squeeze(-1)
