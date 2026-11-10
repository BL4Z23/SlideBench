import argparse
import csv
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import psutil
from tqdm import tqdm

from Game_Engine import decode
from solvers.bfs import bfs
from solvers.astar import astar
from solvers.idastar import idastar

ORDER = ["Starter", "Beginner", "Intermediate", "Advanced", "Expert"]

fields = [
    "file", "index", "board", "solved",
    "moves", "explored", "discovered", "time", "path", "error",
]

names = ["bfs", "astar", "idastar"]

folder = Path("Converters/Converted")

WORKERS = max(1, (psutil.cpu_count(logical=False) or 2) - 1)
IDLE_LIMIT = 5.0          # percent CPU allowed before the run starts
CHECKPOINT_EVERY = 500    # boards between fsync + checkpoint file


def is_grid(item):
    return (
        isinstance(item, list)
        and len(item) == 6
        and all(
            isinstance(row, list)
            and len(row) == 6
            and all(isinstance(value, int) for value in row)
            for row in item
        )
    )


def iter_grids(data):
    if is_grid(data):
        yield data
    elif isinstance(data, list):
        for item in data:
            yield from iter_grids(item)
    elif isinstance(data, dict):
        for value in data.values():
            yield from iter_grids(value)


def skipped():
    return {"solved": False, "moves": None, "explored": None,
            "discovered": None, "time": None, "path": None}


def timed(solver, pieces):
    t0 = time.perf_counter()
    result = solver(pieces)
    result["time"] = time.perf_counter() - t0  # seconds, solver call only
    return result


def safe(solver, pieces):
    """A crash inside a solver becomes an error row instead of killing the run."""
    try:
        return timed(solver, pieces)
    except Exception as error:  # includes MemoryError, RecursionError
        result = skipped()
        result["error"] = f"{type(error).__name__}: {error}"
        return result


def solve_board(task):
    """Runs in a worker process. Returns (file, index, grid, results_or_None, error_or_None)."""
    file_name, index, grid = task
    try:
        pieces = decode(grid)
    except (ValueError, TypeError, AssertionError) as error:
        return file_name, index, grid, None, str(error)

    results = {"bfs": safe(bfs, pieces), "astar": safe(astar, pieces)}
    if results["bfs"]["solved"]:
        results["idastar"] = safe(idastar, pieces)
    else:
        results["idastar"] = skipped()

    for result in results.values():
        result.pop("solver", None)
        result["path"] = json.dumps(result["path"])
    return file_name, index, grid, results, None


def order_key(file):
    for i, name in enumerate(ORDER):
        if file.name.startswith(name):
            return i
    raise ValueError(f"{file.name} does not match ORDER")


def iter_tasks():
    for file in sorted(folder.glob("*.json"), key=order_key):
        with open(file) as f:
            data = json.load(f)
        for index, grid in enumerate(iter_grids(data)):
            yield file.name, index, grid


# ---------- data protection ----------

def read_state(path):
    """Returns (rows, last_key, solved_count, error_count) for an existing CSV."""
    rows = solved = errs = 0
    last = None
    if not path.exists() or path.stat().st_size == 0:
        return 0, None, 0, 0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows += 1
            last = (row["file"], int(row["index"]))
            solved += row["solved"] == "True"
            errs += bool(row["error"])
    return rows, last, solved, errs


def truncate_to(path, n):
    """Keep the header plus the first n data rows (atomic replace)."""
    tmp = path.with_suffix(".tmp")
    with open(path, newline="") as src, open(tmp, "w", newline="") as dst:
        writer = csv.writer(dst)
        for i, row in enumerate(csv.reader(src)):
            if i > n:
                break
            writer.writerow(row)
        dst.flush()
        os.fsync(dst.fileno())
    os.replace(tmp, path)


def write_json_atomic(path, payload):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def sync_all(outs):
    for f in outs.values():
        f.flush()
        os.fsync(f.fileno())


def check_machine_idle(prompt):
    for p in psutil.process_iter():
        p.cpu_percent(None)
    load = psutil.cpu_percent(interval=5)
    print(f"Background CPU load: {load:.1f}%")
    if load > IDLE_LIMIT:
        busy = []
        for p in psutil.process_iter(["name"]):
            try:
                busy.append((p.cpu_percent(None), p.info["name"]))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        print("Busiest processes right now:")
        for pct, name in sorted(busy, reverse=True)[:8]:
            print(f"  {pct:5.1f}%  {name}")
        if prompt:
            if input("Machine is not idle. Start anyway? [y/N] ").strip().lower() != "y":
                raise SystemExit("Aborted: close background programs and retry.")
        else:
            print("WARNING: machine not idle, continuing (--no-prompt).")
    battery = psutil.sensors_battery()
    if battery is not None and not battery.power_plugged:
        print("WARNING: laptop is on battery; plug in to avoid CPU throttling.")


def run_once(out_folder, tasks, run_label):
    """Solves every board once and writes one full copy of the dataset to out_folder."""
    total = len(tasks)
    out_folder.mkdir(parents=True, exist_ok=True)
    paths = {name: out_folder / f"results_{name}.csv" for name in names}

    # ---- resume: keep only rows present in ALL three files, in the same order ----
    n_done = min(read_state(p)[0] for p in paths.values())
    for p in paths.values():
        if read_state(p)[0] > n_done:
            print(f"Trimming {p.name} to {n_done} rows (crash left it ahead).")
            truncate_to(p, n_done)
    resumed = n_done > 0

    if n_done > total:
        raise SystemExit(f"{out_folder} has {n_done} rows but only {total} boards exist.")
    if n_done:
        _, last, _, _ = read_state(paths["bfs"])
        if last != (tasks[n_done - 1][0], tasks[n_done - 1][1]):
            raise SystemExit("Existing rows do not match the board order; refusing to continue. "
                             "Move that folder away or fix it.")
        print(f"Resuming {run_label}: {n_done}/{total} boards already saved.")

    solved_counts = {name: read_state(paths[name])[2] for name in names}
    decode_errors = read_state(paths["bfs"])[3]

    write_json_atomic(out_folder / "run_info.json", {
        "run": run_label,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "workers": WORKERS,
        "cpu": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "python": sys.version,
        "os": platform.platform(),
        "resumed_from_row": n_done,
    })

    pending = tasks[n_done:]
    start = time.time()
    count = n_done

    with ExitStack() as stack:
        outs, writers = {}, {}
        for name in names:
            fresh = not paths[name].exists() or paths[name].stat().st_size == 0
            outs[name] = stack.enter_context(open(paths[name], "a", newline=""))
            writers[name] = csv.DictWriter(outs[name], fieldnames=fields, extrasaction="ignore")
            if fresh:
                writers[name].writeheader()

        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            bar = tqdm(
                pool.map(solve_board, pending, chunksize=1),
                total=total, initial=n_done, unit="board",
                desc=f"{run_label} ({WORKERS} workers)", dynamic_ncols=True,
            )
            for file_name, index, grid, results, error in bar:
                base = {"file": file_name, "index": index, "board": json.dumps(grid)}

                if error is not None:
                    decode_errors += 1
                    for name in names:
                        writers[name].writerow({**base, "error": error})
                else:
                    for name in names:
                        writers[name].writerow({**base, **results[name]})
                        if results[name]["solved"]:
                            solved_counts[name] += 1

                count += 1
                if count % CHECKPOINT_EVERY == 0:
                    sync_all(outs)
                    write_json_atomic(out_folder / "checkpoint.json",
                                      {"boards_saved": count, "total": total,
                                       "time": time.strftime("%Y-%m-%d %H:%M:%S")})

                bar.set_postfix(file=file_name[:20], bfs=solved_counts["bfs"],
                                astar=solved_counts["astar"],
                                ida=solved_counts["idastar"], err=decode_errors)

        sync_all(outs)

    elapsed = time.time() - start
    write_json_atomic(out_folder / "DONE.json", {
        "boards": total, "seconds_this_session": round(elapsed, 1),
        "resumed": resumed, "solved": solved_counts, "decode_errors": decode_errors,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    print(f"\n{run_label} done: {total} boards ({elapsed:.1f}s this session, resumed={resumed})")
    for name in names:
        print(f"  {name:8s} solved {solved_counts[name]}/{total}")
    print(f"  decode errors: {decode_errors}")
    print(f"Results written to {out_folder}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", nargs="?", default="Base_Solved_Level_Data",
                        help="output folder prefix; runs go to <out>_run1, <out>_run2, ...")
    parser.add_argument("--runs", type=int, default=4,
                        help="number of full passes over the dataset (default 4)")
    parser.add_argument("--no-prompt", action="store_true",
                        help="never ask questions (for unattended runs)")
    args = parser.parse_args()

    print("Loading boards...")
    tasks = list(iter_tasks())
    print(f"Found {len(tasks)} boards across {len(set(t[0] for t in tasks))} files")

    run_folders = [Path(f"{args.out}_run{i}") for i in range(1, args.runs + 1)]
    idle_checked = False

    for i, out_folder in enumerate(run_folders, start=1):
        label = f"Run {i}/{args.runs}"
        if (out_folder / "DONE.json").exists():
            print(f"{label}: {out_folder} already complete, skipping.")
            continue
        if not idle_checked:
            check_machine_idle(prompt=not args.no_prompt)
            idle_checked = True
        print(f"\n===== {label} -> {out_folder} =====")
        run_once(out_folder, tasks, label)

    print(f"\nAll {args.runs} runs complete: " + ", ".join(str(p) for p in run_folders))


if __name__ == "__main__":
    main()