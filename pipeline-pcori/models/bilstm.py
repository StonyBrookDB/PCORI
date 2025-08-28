#!/data4/yinan/envs/myenv/bin/python
# -*- coding: utf-8 -*-
from .registry import register
from .lstm import LSTMSeqClassifier

@register("bilstm")
class BiLSTM(LSTMSeqClassifier):
    """
        Bidirectional LSTM
        Equal to
        --model lstm --bidirectional
    """
    def __init__(self, input_dim: int, hidden_size: int = 128, num_layers: int = 1,
                 dropout: float = 0.1, lr: float = 1e-3, batch_size: int = 128,
                 epochs: int = 10, device: str = None, use_amp: bool = False):
        super().__init__(
            input_dim=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,          # Key Defaults
            lr=lr, batch_size=batch_size, epochs=epochs,
            device=device, use_amp=use_amp
        )
