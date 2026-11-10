import pandas as pd
import matplotlib.pyplot as plt

bfs = pd.read_csv("Base_Solved_Level_Data/results_bfs.csv")
astar = pd.read_csv("Base_Solved_Level_Data/results_astar.csv")
idastar = pd.read_csv("Base_Solved_Level_Data/results_idastar.csv")

print(bfs.columns.tolist())

ORDER = ["Starter","Beginner","Intermediate","Advanced","Expert"]

