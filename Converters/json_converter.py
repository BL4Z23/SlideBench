import json
from pathlib import Path


def dump_boards(boards, fp):
    fp.write("[\n")
    for i, board in enumerate(boards):
        rows = ",\n    ".join(json.dumps(row) for row in board)
        fp.write("  [\n    " + rows + "\n  ]")
        fp.write(",\n" if i < len(boards) - 1 else "\n")
    fp.write("]\n")


def converter(path):
    with open(path, "r") as fie:
        data = json.load(fie)

    js = []
    for level in data:
        matrix = [[0 for _ in range(6)] for _ in range(6)]
        for piece in level["pieces"]:
            row, col = piece["row"], piece["col"]
            if piece["is_red"]:
                matrix[row][col] = 1
                matrix[row][col + 1] = 1
            elif piece["length"] == 2 and piece["orientation"] == "H":
                matrix[row][col] = 2
                matrix[row][col + 1] = 2
            elif piece["length"] == 3 and piece["orientation"] == "H":
                matrix[row][col] = 3
                matrix[row][col + 1] = 3
                matrix[row][col + 2] = 3
            elif piece["length"] == 2 and piece["orientation"] == "V":
                matrix[row][col] = 4
                matrix[row + 1][col] = 4
            elif piece["length"] == 3 and piece["orientation"] == "V":
                matrix[row][col] = 5
                matrix[row + 1][col] = 5
                matrix[row + 2][col] = 5
            else:
                raise ValueError(f"Unexpected piece: {piece}")
        js.append(matrix)

    output_dir = Path(__file__).parent / "Converted"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / Path(path).name.replace(".json", "_converted.json")
    with open(output_path, "w") as fil:
        dump_boards(js, fil)


if __name__ == "__main__":
    folder = Path("../scans")
    for file in folder.iterdir():
        if file.is_file() and file.suffix == ".json":
            converter(str(file))

