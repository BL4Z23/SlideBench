import csv
import json
from Game_Engine import decode, SHAPE, VALUE_TO_TYPE

N = 6
GOAL_COL = 4
INF = float("inf")
TID = {name: value for value, name in VALUE_TO_TYPE.items()}
RED = TID["R"]

INFO = {}
STEPS = {}
RM = {}


def build_tables():
    for ptype, tid in TID.items():
        length, orient = SHAPE[ptype]
        horiz = orient == "H"
        for r in range(N):
            for c in range(N):
                if horiz and c + length > N:
                    continue
                if not horiz and r + length > N:
                    continue

                pos = r * N + c
                code = pos * 8 + tid

                mask = 0
                for k in range(length):
                    cr, cc = (r, c + k) if horiz else (r + k, c)
                    mask |= 1 << (cr * N + cc)
                INFO[code] = (mask, tid, r, c)

                neg_bits, neg_codes, pos_bits, pos_codes = [], [], [], []
                if horiz:
                    for k in range(1, c + 1):
                        neg_bits.append(1 << (r * N + c - k))
                        neg_codes.append((pos - k) * 8 + tid)
                    for k in range(1, N - length - c + 1):
                        pos_bits.append(1 << (r * N + c + length - 1 + k))
                        pos_codes.append((pos + k) * 8 + tid)
                else:
                    for k in range(1, r + 1):
                        neg_bits.append(1 << ((r - k) * N + c))
                        neg_codes.append((pos - k * N) * 8 + tid)
                    for k in range(1, N - length - r + 1):
                        pos_bits.append(1 << ((r + length - 1 + k) * N + c))
                        pos_codes.append((pos + k * N) * 8 + tid)

                STEPS[code] = (
                    tuple(neg_bits), tuple(neg_codes),
                    tuple(pos_bits), tuple(pos_codes),
                )

    for r in range(N):
        for c in range(N):
            mask = 0
            for cc in range(c + 2, N):
                mask |= 1 << (r * N + cc)
            RM[r * N + c] = mask


build_tables()


def encode(pieces):
    return tuple(sorted(
        (piece[1][0] * N + piece[1][1]) * 8 + TID[piece[0]]
        for piece in pieces
    ))


def full_heuristic(board):
    for code in board:
        if code & 7 == RED:
            red_pos = code >> 3
            break
    if red_pos % N == GOAL_COL:
        return 0
    rm = RM[red_pos]
    count = 0
    for code in board:
        if code & 7 != RED and INFO[code][0] & rm:
            count += 1
    return count + 1


def idastar(initial_board):
    start = encode(initial_board)
    explored = 0
    generated = 1
    path = []

    def search(board, g, h, threshold, seen):
        nonlocal explored, generated

        explored += 1

        if h == 0:
            return True

        occ = 0
        for code in board:
            occ |= INFO[code][0]
            if code & 7 == RED:
                red_code = code
        rm = RM[red_code >> 3]

        minimum = INF
        new_g = g + 1

        for i, code in enumerate(board):
            mask, tid, r, c = INFO[code]
            neg_bits, neg_codes, pos_bits, pos_codes = STEPS[code]
            rest = board[:i] + board[i + 1:]
            is_red = tid == RED

            for sign, bits, codes in ((-1, neg_bits, neg_codes), (1, pos_bits, pos_codes)):
                for k in range(len(bits)):
                    if occ & bits[k]:
                        break

                    new_code = codes[k]
                    child = tuple(sorted(rest + (new_code,)))

                    if seen.get(child, INF) <= new_g:
                        continue

                    seen[child] = new_g
                    generated += 1

                    if is_red:
                        new_pos = new_code >> 3
                        if new_pos % N == GOAL_COL:
                            child_h = 0
                        else:
                            new_rm = RM[new_pos]
                            child_h = 1
                            for other in rest:
                                if INFO[other][0] & new_rm:
                                    child_h += 1
                    else:
                        child_h = h
                        if mask & rm:
                            child_h -= 1
                        if INFO[new_code][0] & rm:
                            child_h += 1

                    f = new_g + child_h

                    if f > threshold:
                        if f < minimum:
                            minimum = f
                        continue

                    path.append(((r, c), sign * (k + 1)))
                    result = search(child, new_g, child_h, threshold, seen)

                    if result is True:
                        return True

                    path.pop()

                    if result < minimum:
                        minimum = result

        return minimum

    h0 = full_heuristic(start)
    threshold = h0

    while True:
        seen = {start: 0}
        result = search(start, 0, h0, threshold, seen)

        if result is True:
            return {
                "solver": "idastar",
                "solved": True,
                "moves": len(path),
                "path": list(path),
                "explored": explored,
                "discovered": generated,
            }

        if result == INF:
            return {
                "solver": "idastar",
                "solved": False,
                "moves": None,
                "path": None,
                "explored": explored,
                "discovered": generated,
            }

        threshold = result


if __name__ == '__main__':
    boards = [
        [
            [3, 3, 3, 0, 0, 5],
            [0, 0, 5, 0, 0, 5],
            [1, 1, 5, 0, 0, 5],
            [4, 0, 5, 0, 2, 2],
            [4, 0, 0, 0, 4, 0],
            [3, 3, 3, 0, 4, 0]
        ],
    ]

    fields = ["board", "solver", "solved", "moves", "explored", "discovered", "path"]

    with open("idastar_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for grid in boards:
            result = idastar(decode(grid))
            result["board"] = json.dumps(grid)
            result["path"] = json.dumps(result["path"])
            writer.writerow(result)