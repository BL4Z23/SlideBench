from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/"results"/"baseline_runs"
OUT = ROOT/"results"/"baseline_masters"
SOLVERS = ["bfs", "astar", "idastar"]
KEY = ["file", "index"]
SAME = ["solved", "moves", "explored", "discovered"]


def load(solver):
    frames = []
    run_dirs = sorted(RUNS.glob("run_*"))
    for run_dir in run_dirs:
        df = pd.read_csv(run_dir/f"results_{solver}.csv",
                         usecols=KEY + SAME + ["time", "error"])
        df["run"] = run_dir.name
        frames.append(df)
    print(f"{solver}: {len(frames)} runs, {[len(f) for f in frames]} rows each")
    return pd.concat(frames, ignore_index=True), len(frames)


def build(solver):
    df, n_runs = load(solver)

    per_level = df.groupby(KEY).size()
    assert (per_level == n_runs).all(), f"{solver}: some levels missing from some runs"

    changed = df.groupby(KEY)[SAME].nunique(dropna=False) > 1
    bad = changed.any(axis=1).sum()
    assert bad == 0, f"{solver}: {bad} levels differ between runs in {SAME}"

    master = df.groupby(KEY, sort=False).agg(
        solved=("solved", "first"),
        moves=("moves", "first"),
        explored=("explored", "first"),
        discovered=("discovered", "first"),
        time_median=("time", "median"),
        time_mean=("time", "mean"),
        time_std=("time", "std"),
        time_min=("time", "min"),
        time_max=("time", "max"),
        n_runs=("time", "count"),
        errors=("error", lambda s: s.notna().sum()),
    ).reset_index()

    # one column per run: time_run_1 ... time_run_5
    per_run = (df.pivot(index=KEY, columns="run", values="time")
                 .rename(columns=lambda name: f"time_{name}")
                 .reset_index())
    master = master.merge(per_run, on=KEY, how="left", validate="one_to_one")
    master.insert(2, "solver", solver)
    return master


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for solver in SOLVERS:
        master = build(solver)
        master.to_csv(OUT / f"master_{solver}.csv", index=False)
        print(f"  -> master_{solver}.csv: {len(master)} levels, "
              f"{int(master.solved.sum())} solved, {int(master.errors.sum())} error rows")


if __name__ == "__main__":
    main()