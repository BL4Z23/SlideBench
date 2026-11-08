import csv
import json
from contextlib import ExitStack
from pathlib import Path

from Game_Engine import decode
from files.bfs import bfs
from files.astar import astar
from files.idastar import idastar

fields = [
    "file", "index", "board", "solver", "solved",
    "moves", "explored", "discovered", "path", "error",
]

names = ["bfs", "astar", "idastar"]

folder = Path("../Converters/Converted")


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

    for file in sorted(folder.glob("*.json")):
        with open(file) as f:
            data = json.load(f)

        for index, grid in enumerate(iter_grids(data)):
            base = {"file": file.name, "index": index, "board": json.dumps(grid)}

            try:
                pieces = decode(grid)
            except (ValueError, TypeError) as error:
                for name in names:
                    writers[name].writerow({**base, "solver": name, "error": str(error)})
                    outs[name].flush()
                continue

            results = {"bfs": bfs(pieces), "astar": astar(pieces)}
            if results["bfs"]["solved"]:
                results["idastar"] = idastar(pieces)
            else:
                results["idastar"] = skipped("idastar")

            for name in names:
                result = results[name]
                result["path"] = json.dumps(result["path"])
                writers[name].writerow({**base, **result})
                outs[name].flush()