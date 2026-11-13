from datetime import datetime
import copy
import torch
from torch import nn
from torch.utils.data import random_split
from slidebench.solvers.Learned_Heuristic.Dataset import Board_Data
from slidebench.solvers.Learned_Heuristic.MLP_Model import MLP_Heuristic_Trainer, MAX_MOVES

DB = "../../../data/generated/run_1/slidebench.db"
Save_Path = "../../../data/Heuristic_Data/"

SEED = 41
CONFIG = {"width": 256, "blocks": 3, "dropout": 0.0}
batch_size = 1024
lr = 2e-3
weight_decay = 1e-4
over_weight = 1.0
max_epochs = 2000
sched_patience = 8
limit = 30
min_lr = 1e-6
rel_improve = 1e-3

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

Training_Data = Board_Data(DB)
generator = torch.Generator().manual_seed(41)
val_size = int(0.1 * len(Training_Data))
train_data, val_data = random_split(
    Training_Data, [len(Training_Data) - val_size, val_size], generator=generator
)


def to_tensors(subset):
    boards, moves = zip(*(subset[i] for i in range(len(subset))))
    return (
        torch.stack([torch.as_tensor(b) for b in boards]).to(device),
        torch.as_tensor(moves).float().to(device),
    )


train_x, train_y = to_tensors(train_data)
val_x, val_y = to_tensors(val_data)
n_train = train_x.size(0)

torch.manual_seed(SEED)
model = MLP_Heuristic_Trainer(**CONFIG).to(device)
mse = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=sched_patience
)


def train_loss_fn(pred, target):
    err = pred - target
    w = torch.where(err > 0, over_weight, 1.0)
    return (w * err ** 2).mean()


best_val_loss = float("inf")
best_state = None
epochs_without_improvement = 0

for epoch in range(max_epochs):
    model.train()
    perm = torch.randperm(n_train, device=device)
    train_total = torch.zeros((), device=device)

    for i in range(0, n_train, batch_size):
        idx = perm[i:i + batch_size]
        boards, moves = train_x[idx], train_y[idx]

        predictions = model(boards).squeeze(-1)
        loss = train_loss_fn(predictions, moves)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_total += loss.detach() * boards.size(0)
    train_loss = train_total.item() / n_train

    model.eval()
    with torch.no_grad():
        val_pred = model(val_x).squeeze(-1)
        val_loss = mse(val_pred, val_y).item()
        err_moves = (val_pred - val_y) * MAX_MOVES
        mae = err_moves.abs().mean().item()
        rounded = err_moves.round()
        exact = (rounded == 0).float().mean().item()
        over = (rounded > 0).float().mean().item()

    scheduler.step(val_loss)
    cur_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch [{epoch + 1}/{max_epochs}] "
        f"Train: {train_loss:.5f} | Val: {val_loss:.5f} | Best: {best_val_loss:.5f} | "
        f"MAE: {mae:.3f} moves | exact: {exact:.1%} | over: {over:.1%} | lr: {cur_lr:.1e}"
    )

    if val_loss < best_val_loss * (1 - rel_improve):
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= limit or cur_lr <= min_lr:
        print(f"Early stopping at epoch {epoch}. Best val loss: {best_val_loss:.6f}")
        break

if best_state is None:
    best_state = copy.deepcopy(model.state_dict())

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
torch.save(
    {
        "state_dict": best_state,
        "config": CONFIG,
        "max_moves": MAX_MOVES,
        "val_loss": best_val_loss,
    },
    f"{Save_Path}model_{timestamp}.pth",
)