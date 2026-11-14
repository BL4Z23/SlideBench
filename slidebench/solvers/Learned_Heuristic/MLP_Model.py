from torch import nn
import torch
from .Dataset import tensor_maker

MAX_MOVES = 45


class ResBlock(nn.Module):
    def __init__(self, width, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
        )

    def forward(self, x):
        return x + self.net(x)


class MLP_Heuristic_Trainer(nn.Module):
    def __init__(self, width=256, blocks=3, dropout=0.0):
        super().__init__()
        self.flatten = nn.Flatten()
        self.inp = nn.Linear(216, width)
        self.blocks = nn.Sequential(*[ResBlock(width, dropout) for _ in range(blocks)])
        self.out = nn.Sequential(
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, 1),
        )

    def forward(self, x):
        x = self.flatten(x).float()
        return self.out(self.blocks(self.inp(x)))


_model = None
_margin = 0.0


def _load(model_path):
    ckpt = torch.load(model_path, map_location="cpu")
    model = MLP_Heuristic_Trainer(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, float(ckpt.get("margin", 0.0))


@torch.inference_mode()
def calculate_MLP_heuristic(board, model_path="model.pth"):
    global _model, _margin
    if _model is None:
        _model, _margin = _load(model_path)

    x = tensor_maker(board)
    raw = _model(x).squeeze(-1) * MAX_MOVES
    # floor (never round up) and subtract the calibrated safety margin
    preds = torch.floor(raw - _margin).clamp(min=0)
    return preds.int().tolist()[0]