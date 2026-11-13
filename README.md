# SlideBench

A sliding block puzzle benchmark. It compares classical solvers (BFS, A\*, IDA\*) against trained neural models on Unblock Me style puzzles.

> **Status: work in progress.** Classical baselines are done. The learned heuristic is working. The other models are still planned.
>
> **Note:** this is a basic version of the README. A full version with detailed methodology, results and graphs is planned.

## Research question

*Can an AI be trained to be better at Unblock Me than a BFS?*

BFS always finds the optimal move count, but it explores a lot of states. The hypothesis is that a trained model can reach a short (ideally optimal) solution while exploring far fewer states and taking less time. The open question is how each solver scales as levels get harder.

## The puzzle

- 6x6 grid.
- The red block must slide out of the exit, which is always on the right side of row 2 (0-indexed).
- Other pieces are 2 or 3 cells long and slide along one axis only.
- Sliding a piece any distance counts as **1 move**.
- Goal: fewest moves.

### Board encoding

A board is a 6x6 matrix of cell values:

| Value | Meaning |
|-------|---------|
| 0 | empty |
| 1 | red block |
| 2 | 2-long horizontal |
| 3 | 3-long horizontal |
| 4 | 2-long vertical |
| 5 | 3-long vertical |

Pieces have no IDs. Fixed piece lengths are enough to split runs of identical cells back into pieces. A board can also be stored as a 36-character flattened string.

A solution is a list of `((row, col), distance)` tuples. `(row, col)` is the piece's top-left cell, and `distance` is signed (positive = right/down, negative = left/up).

## Solvers

| Solver | Type | Status |
|--------|------|--------|
| BFS | classical | done |
| A\* (blocking pieces + 1 heuristic) | classical | done |
| IDA\* | classical | done |
| A\* with learned heuristic (MLP) | learned | working, being evaluated |
| Imitation of BFS (predicts next optimal board) | learned | planned |
| RL policy (masked move scores) | learned | planned |

## Data

**Test set:** 16,799 real levels from a popular mobile Unblock Me game, across 5 tiers.

| Tier | Levels |
|------|--------|
| Starter | 2,000 |
| Beginner | 8,299 |
| Intermediate | 4,200 |
| Advanced | 1,500 |
| Expert | 800 |

**Training set:** the models train only on self-generated boards, never on the real levels. Boards are generated, kept only if solvable, deduplicated, and labelled with their optimal move count from BFS. The dataset is about 820,000 boards covering optimal lengths of 5 to 45 moves, stored in a SQLite database. Very deep boards are rare, so the hardest bucket is much smaller than the others.

## Learned heuristic

A residual MLP takes the flattened board (216 inputs) and predicts the remaining distance, scaled by `MAX_MOVES = 45`. At inference the output is multiplied back by `MAX_MOVES`, rounded, clamped at 0 and returned as an int, so A\* only ever sees integers.

Training details:
- 90/10 train/validation split with a fixed seed
- AdamW with a `ReduceLROnPlateau` scheduler
- early stopping on validation loss, restoring the best weights
- configurable `over_weight` to penalize overestimation (overestimates can make A\* return non-optimal solutions)
- the checkpoint stores weights, config, `max_moves` and best validation loss

Preliminary result on the generated validation set: 98.4% exact, 1.0% overestimates. Evaluation on the real levels is still to do.

## Planned: CNN heuristic

A CNN version of the learned heuristic is also planned. The MLP flattens the board, so it has to learn from scratch which cells sit next to each other. A CNN works on the 6x6 grid directly, so it should see spatial arrangements (like a piece blocking the red block's row) more easily. It will follow the same interface as the MLP, so A\* can swap between the two, and it will be trained and tested on the same data split for a fair comparison.

## Evaluation plan

For each solver on the real levels:
- **Solution length** compared with the BFS optimum. A level where the solution is longer than optimal counts as a failure.
- **Nodes explored/discovered.** Wall-clock time is recorded only in the timed benchmark runs, since it depends on the machine.
- **Scaling** with level difficulty.

Classical baselines were run 5 times. Raw runs are kept, and a master dataset aggregates one row per level per solver.

## Project layout

```
SlideBench/
├── data/
│   ├── generated/          # generated training boards (SQLite database)
│   ├── Heuristic_Data/     # saved model checkpoints
│   ├── levels/             # real levels in the converted JSON format
│   └── raw_scans/          # raw scans of the real levels
├── results/
│   ├── baseline_runs/      # raw benchmark runs (BFS, A*, IDA*)
│   ├── baseline_masters/   # aggregated master datasets
│   └── baseline_graphs/    # baseline figures
├── scanner/                # tools for scanning levels from the app
│   ├── templates/
│   ├── calibrate_board.py
│   ├── crop_template.py
│   ├── detect.py
│   ├── scan_app.py
│   └── view_raw.py
├── scripts/
│   ├── convert_scans.py        # raw scans -> level JSON
│   ├── database_generator.py   # builds the training database
│   ├── run_baseline.py         # runs the classical solvers
│   ├── baseline_master_maker.py
│   └── baseline_graph_maker.py
└── slidebench/             # the package
    ├── engine.py           # board, move generation, solver interface
    └── solvers/
        ├── bfs.py
        ├── astar.py
        ├── idastar.py
        └── Learned_Heuristic/
            ├── Dataset.py      # Board_Data dataset + tensor_maker
            ├── MLP_Model.py    # model + calculate_heuristics()
            └── MLP_Trainer.py  # training, validation, early stopping
```

## Roadmap

- [x] Engine, BFS, A\*, IDA\*
- [x] Classical benchmark runs and graphs
- [x] Training data generator
- [x] Learned heuristic (MLP)
- [ ] Evaluate learned heuristic in A\* on all real levels
- [ ] Compare architectures (e.g. CNN) and `over_weight` settings
- [ ] Imitation model
- [ ] RL policy model
- [ ] Final head-to-head race and replays
- [ ] CLI: solve a pasted 36-character board

## Contributing

SlideBench is meant to be open. Convert your own solver to the engine's output format and benchmark it against the others.
