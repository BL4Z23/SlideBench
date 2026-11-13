import numpy as np
VALUE_TO_TYPE = {1: "R", 2: "2H", 3: "3H", 4: "2V", 5: "3V"}
TYPE_TO_VALUE = {v: k for k, v in VALUE_TO_TYPE.items()}
SHAPE = {"R": (2, "H"), "2H": (2, "H"), "3H": (3, "H"),
         "2V": (2, "V"), "3V": (3, "V")}
N = 6

def decode(board):
    pieces = []
    for row in range(N):
        col = 0
        while col < N:
            v = int(board[row][col])
            if v in (1, 2, 3):
                t = VALUE_TO_TYPE[v]
                length = SHAPE[t][0]
                if col + length > N or any(int(board[row][col + i]) != v for i in range(length)):
                    raise ValueError(f"Malformed horizontal piece at {(row, col)}")
                pieces.append((t, (row, col)))
                col += length
            else:
                col += 1
    for col in range(N):
        row = 0
        while row < N:
            v = int(board[row][col])
            if v in (4, 5):
                t = VALUE_TO_TYPE[v]
                length = SHAPE[t][0]
                if row + length > N or any(int(board[row + i][col]) != v for i in range(length)):
                    raise ValueError(f"Malformed vertical piece at {(row, col)}")
                pieces.append((t, (row, col)))
                row += length
            else:
                row += 1
    return sorted(pieces, key=lambda p: p[1])

def encode(pieces):
    board = np.zeros((N, N), dtype=int)

    for t, (row, col) in pieces:
        value = TYPE_TO_VALUE[t]
        length, orientation = SHAPE[t]

        for i in range(length):
            if orientation == "H":
                board[row][col + i] = value
            else:
                board[row + i][col] = value

    return board

def piece_cells(piece):
    t, (r, c) = piece
    length = SHAPE[t][0]
    if t == "R" or t.endswith("H"):
        return [(r, c + i) for i in range(length)]
    return [(r + i, c) for i in range(length)]

def move(board, pos, distance):
    board = list(board)
    for piece in board:
        if piece[1][0] == pos[0] and piece[1][1] == pos[1]:
            row = piece[1][0]
            col = piece[1][1]

            if piece[0][0] == 'R':
                if distance > 0:
                    if col + distance + 1 < N:
                        occupied = []
                        for i in range(1,distance+1):
                            occupied.append((row, col + 1 + i))
                        op = {cell for p in board for cell in piece_cells(p)}
                        if all(coord not in op for coord in occupied):
                            board[board.index(piece)] = (piece[0], (row, col + distance))
                            return tuple(sorted(board, key=lambda p: p[1]))
                elif distance < 0:
                    if col + distance >= 0:
                        occupied = []
                        for i in range(1, abs(distance) + 1):
                            occupied.append((row, col - i))
                        op = {cell for p in board for cell in piece_cells(p)}
                        if all(coord not in op for coord in occupied):
                            board[board.index(piece)] = (piece[0], (row, col + distance))
                            return tuple(sorted(board, key=lambda p: p[1]))
            elif piece[0][1] == 'H':
                length = int(piece[0][0])
                idx = board.index(piece)
                op = {cell for p in board for cell in piece_cells(p)}

                if distance > 0:
                    if col + distance + length <= N:
                        path = [(row, col + length - 1 + i) for i in range(1, distance + 1)]
                        if all(cell not in op for cell in path):
                            board[idx] = (piece[0], (row, col + distance))
                            return tuple(sorted(board, key=lambda p: p[1]))
                elif distance < 0:
                    if col + distance >= 0:
                        path = [(row, col - i) for i in range(1, abs(distance) + 1)]
                        if all(cell not in op for cell in path):
                            board[idx] = (piece[0], (row, col + distance))
                            return tuple(sorted(board, key=lambda p: p[1]))
            elif piece[0][1] == 'V':
                length = int(piece[0][0])
                idx = board.index(piece)
                op = {cell for p in board for cell in piece_cells(p)}

                if distance > 0:
                    if row + distance + length <= N:
                        path = [(row + length - 1 + i, col) for i in range(1, distance + 1)]
                        if all(cell not in op for cell in path):
                            board[idx] = (piece[0], (row + distance, col))
                            return tuple(sorted(board, key=lambda p: p[1]))
                elif distance < 0:
                    if row + distance >= 0:
                        path = [(row-i, col) for i in range(1, abs(distance) + 1)]
                        if all(cell not in op for cell in path):
                            board[idx] = (piece[0], (row + distance, col))
                            return tuple(sorted(board, key=lambda p: p[1]))
            else:
                return None
    return None
def valid_moves(board):
    valid = []
    op = {cell for p in board for cell in piece_cells(p)}
    for piece in board:
        distance = -1
        row = piece[1][0]
        col = piece[1][1]
        if (piece[0][0] == 'R'):
            while col + distance > -1 and (row, col + distance) not in op:
                valid.append(((row, col), distance))
                distance -= 1
            distance = 1
            while col + distance + 1 < N and (row, col + distance + 1) not in op:
                valid.append(((row, col), distance))
                distance += 1
        elif piece[0][1] == 'H':
            while col + distance > -1 and (row, col + distance) not in op:
                valid.append(((row, col), distance))
                distance -= 1
            distance = 1
            while col + distance + int(piece[0][0]) <= N and (row, col + int(piece[0][0]) - 1 + distance) not in op:
                valid.append(((row, col), distance))
                distance += 1
        elif piece[0][1] == 'V':
            while row + distance > -1 and (row + distance, col) not in op:
                valid.append(((row, col), distance))
                distance -= 1
            distance = 1
            while row + distance + int(piece[0][0]) <= N and (row + int(piece[0][0]) - 1 + distance, col ) not in op:
                valid.append(((row, col), distance))
                distance += 1

    return valid
def won(board):
    for piece in board:
        if piece[0] == "R":
            if piece[1][1] == 4:
                return True
            else:
                return False
    return False







