import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from pathlib import Path

from tqdm import tqdm

from Game_Engine import decode
from files.bfs import bfs
from files.astar import astar
from files.idastar import idastar

fields = [
    "file", "index", "board", "solver", "solved",
    "moves", "explored", "discovered", "path", "error",
]

names = ["bfs", "astar", "idastar"]

folder = Path("Converters/Converted")


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


def skipped(solver):
    return {
        "solver": solver,
        "solved": False,
        "moves": None,
        "explored": None,
        "discovered": None,
        "path": None,
    }


def solve_board(task):
    """Runs in a worker process. Returns (file, index, grid, results_or_None, error_or_None)."""
    file_name, index, grid = task
    try:
        pieces = decode(grid)
    except (ValueError, TypeError) as error:
        return file_name, index, grid, None, str(error)

    results = {"bfs": bfs(pieces), "astar": astar(pieces)}
    if results["bfs"]["solved"]:
        results["idastar"] = idastar(pieces)
    else:
        results["idastar"] = skipped("idastar")

    for result in results.values():
        result["path"] = json.dumps(result["path"])
    return file_name, index, grid, results, None


def iter_tasks():
    for file in sorted(folder.glob("*.json")):
        with open(file) as f:
            data = json.load(f)
        for index, grid in enumerate(iter_grids(data)):
            yield file.name, index, grid


def main():
    Path("solved_csvs").mkdir(exist_ok=True)

    with ExitStack() as stack:
        outs = {
            name: stack.enter_context(open(f"solved_csvs/results_{name}.csv", "w", newline=""))
            for name in names
        }
        writers = {
            name: csv.DictWriter(outs[name], fieldnames=fields)
            for name in names
        }
        for writer in writers.values():
            writer.writeheader()

        print("Loading boards...")
        tasks = list(iter_tasks())
        total = len(tasks)
        print(f"Found {total} boards across {len(set(t[0] for t in tasks))} files")

        workers = os.cpu_count() or 1
        solved_counts = {name: 0 for name in names}
        decode_errors = 0
        start = time.time()

        with ProcessPoolExecutor(max_workers=workers) as pool:
            # map() yields results in input order, so the CSV row order matches
            # the original sequential script.
            bar = tqdm(
                pool.map(solve_board, tasks, chunksize=1),
                total=total,
                unit="board",
                desc=f"Solving ({workers} workers)",
                dynamic_ncols=True,
            )
            for file_name, index, grid, results, error in bar:
                base = {"file": file_name, "index": index, "board": json.dumps(grid)}

                if error is not None:
                    decode_errors += 1
                    for name in names:
                        writers[name].writerow({**base, "solver": name, "error": error})
                        outs[name].flush()
                else:
                    for name in names:
                        writers[name].writerow({**base, **results[name]})
                        outs[name].flush()
                        if results[name]["solved"]:
                            solved_counts[name] += 1

                bar.set_postfix(
                    file=file_name[:20],
                    bfs=solved_counts["bfs"],
                    astar=solved_counts["astar"],
                    ida=solved_counts["idastar"],
                    err=decode_errors,
                )

        elapsed = time.time() - start
        print(f"\nDone: {total} boards in {elapsed:.1f}s ({elapsed / max(total, 1):.2f}s/board)")
        for name in names:
            print(f"  {name:8s} solved {solved_counts[name]}/{total}")
        print(f"  decode errors: {decode_errors}")
        print("Results written to solved_csvs/")


if __name__ == "__main__":
    main()