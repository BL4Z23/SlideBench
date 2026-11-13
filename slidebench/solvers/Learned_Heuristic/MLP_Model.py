from torch import nn
import torch
from Dataset import tensor_maker

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


def _load(model_path):
    ckpt = torch.load(model_path, map_location="cpu")
    model = MLP_Heuristic_Trainer(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


@torch.inference_mode()
def calculate_heuristics(boards, model_path="model.pth"):
    global _model
    if _model is None:
        _model = _load(model_path)

    x = torch.cat([tensor_maker(b) for b in boards], dim=0)
    preds = _model(x).squeeze(-1) * MAX_MOVES
    return preds.round().clamp(min=0).int().tolist()


def calculate_MLP_heuristic(board, model_path="model.pth"):
    return calculate_heuristics([board], model_path)[0]