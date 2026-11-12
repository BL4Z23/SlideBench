import sqlite3
import numpy as np
from jinja2.filters import do_attr

DB = "../data/generated/run_1/slidebench.db"
import torch
from torch.utils.data import Dataset

class Data(Dataset):
    def __init__(self, filename):
        self.conn = sqlite3.connect(filename)
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
                       DELETE
                       FROM boards
                       WHERE accepted = 0
                       """)

        self.conn.commit()
        self.count = self.conn.execute("SELECT COUNT(*) FROM boards").fetchone()[0]
        self.board_data = np.zeros((self.count, 36))
        self.move_data = np.zeros((self.count, 1))
        self.cursor.execute("SELECT board, moves FROM boards")
        i = 0
        for row in self.cursor:
            board = row[0]
            moves = row[1]
            x = []
            for char in reversed(board):
                x.append(int(char))
            self.board_data[i][:len(board)] = list(x)
            self.move_data[i] = moves
            i += 1

        self.conn.close()

    def __getitem__(self, i):
        item = self.board_data[i]
        moves = self.move_data[i]

        matrix = item.reshape(6, 6)
        masks = np.array([
            (matrix == tile).astype(np.float32)
            for tile in range(6)
        ])
        board_tensor = torch.tensor(masks, dtype=torch.float32)
        move_tensor = torch.tensor(moves.item()/45, dtype=torch.float32)

        return board_tensor, move_tensor

    def __len__(self):
        return self.count

