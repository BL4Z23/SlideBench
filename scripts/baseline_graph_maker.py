import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sns.set_context("talk")

sns.set_theme(
    style="darkgrid",
    context="notebook",
    palette="muted",
    rc={
        "figure.facecolor": "#0B0F19",
        "axes.facecolor": "#111827",
        "axes.edgecolor": "#263244",
        "axes.labelcolor": "#CBD5E1",
        "text.color": "#F8FAFC",
        "xtick.color": "#94A3B8",
        "ytick.color": "#94A3B8",
        "grid.color": "#334155",
        "grid.alpha": 0.3,
        "axes.titleweight": "bold",
        "axes.titlesize": 16,
        "font.family": "DejaVu Sans",
    }
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "baseline_graphs"
OUTPUT.mkdir(parents=True, exist_ok=True)

dfs = [
    pd.read_csv(ROOT / "results" / "baseline_masters" / f"master_{s}.csv")
    for s in ["astar", "bfs", "idastar"]
]

labels = ['A*', 'BFS', 'IDA*']
colors = ["#E76F51", "#3A86FF", "#2A9D8F"]

def save_plot(fig, filename):
    fig.tight_layout()
    fig.savefig(OUTPUT / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

def optimized_range(df = dfs[0]):
    n = len(df)
    sizes = df.groupby("moves").size()
    i = 0
    j = len(sizes) - 1

    k = 0

    while i < len(sizes) and k < n/1000:
        k += sizes.iloc[i]
        i += 1

    k = 0

    while j >= 0 and k < n/1000:
        k += sizes.iloc[j]
        j -= 1

    if i > j:
        raise ValueError("Not enough samples for plotting.")
    return [sizes.index[i], sizes.index[j]]

def get_grouped_stats(df, column):
    sizes = df.groupby("moves").size()
    i = 0
    j = len(sizes) - 1

    while i < len(sizes) and sizes.iloc[i] < 20:
        i += 1

    while j >= 0 and sizes.iloc[j] < 20:
        j -= 1

    if i > j:
        raise ValueError("Not enough samples for plotting.")

    grouped = df.groupby("moves")[column]

    median = grouped.median().iloc[i:j + 1]
    q25 = grouped.quantile(0.25).iloc[i:j + 1]
    q75 = grouped.quantile(0.75).iloc[i:j + 1]

    return median, q25, q75

def explored_vs_length(dfs):
    combined_fig, combined_ax = plt.subplots(figsize=(10, 6))

    for df, label, color in zip(dfs, labels, colors):
        median, q25, q75 = get_grouped_stats(df, "explored")

        # Individual plot
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(median.index, median.values,
                label=label, color=color, linewidth=2)

        ax.fill_between(median.index, q25.values, q75.values,
                        color=color, alpha=0.15)

        ax.set_title(f"{label}: Explored Nodes vs Solution Length")
        ax.set_xlabel("Optimal Solution Length (moves)")
        ax.set_ylabel("Median Explored Nodes")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

        save_plot(fig, f"explored_vs_length_{label.replace('*', 'star')}.png")

        # Combined plot
        combined_ax.plot(median.index, median.values,
                         label=label, color=color, linewidth=2)

        combined_ax.fill_between(median.index, q25.values, q75.values,
                                 color=color, alpha=0.15)

    combined_ax.set_title("Explored Nodes vs Solution Length: All Solvers")
    combined_ax.set_xlabel("Optimal Solution Length (moves)")
    combined_ax.set_ylabel("Median Explored Nodes")
    combined_ax.set_yscale("log")
    combined_ax.legend()
    combined_ax.grid(True, which="both", alpha=0.3)

    save_plot(combined_fig, "explored_vs_length_all.png")


def time_vs_length(dfs):
    combined_fig, combined_ax = plt.subplots(figsize=(10, 6))

    for df, label, color in zip(dfs, labels, colors):
        median, q25, q75 = get_grouped_stats(df, "time_median")

        # Individual plot
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(median.index, median.values,
                label=label, color=color, linewidth=2)

        ax.fill_between(median.index, q25.values, q75.values,
                        color=color, alpha=0.15)

        ax.set_title(f"{label}: Execution Time vs Solution Length")
        ax.set_xlabel("Optimal Solution Length (moves)")
        ax.set_ylabel("Median Execution Time (seconds)")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

        save_plot(fig, f"time_vs_length_{label.replace('*', 'star')}.png")

        # Combined plot
        combined_ax.plot(median.index, median.values,
                         label=label, color=color, linewidth=2)

        combined_ax.fill_between(median.index, q25.values, q75.values,
                                 color=color, alpha=0.15)

    combined_ax.set_title("Execution Time vs Solution Length: All Solvers")
    combined_ax.set_xlabel("Optimal Solution Length (moves)")
    combined_ax.set_ylabel("Median Execution Time (seconds)")
    combined_ax.set_yscale("log")
    combined_ax.legend()
    combined_ax.grid(True, which="both", alpha=0.3)

    save_plot(combined_fig, "time_vs_length_all.png")


def time_vs_explored(dfs):
    combined_fig, combined_ax = plt.subplots(figsize=(10, 6))

    for df, label, color in zip(dfs, labels, colors):
        # Individual plot
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(
            df["explored"], df["time_median"],
            label=label, color=color, alpha=0.1, s=1
        )

        ax.set_title(f"{label}: Execution Time vs Explored Nodes")
        ax.set_xlabel("Explored Nodes")
        ax.set_ylabel("Execution Time (seconds)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

        save_plot(fig, f"time_vs_explored_{label.replace('*', 'star')}.png")

        # Combined plot
        combined_ax.scatter(
            df["explored"], df["time_median"],
            label=label, color=color, alpha=0.1, s=1
        )

    combined_ax.set_title("Execution Time vs Explored Nodes: All Solvers")
    combined_ax.set_xlabel("Explored Nodes")
    combined_ax.set_ylabel("Execution Time (seconds)")
    combined_ax.set_xscale("log")
    combined_ax.set_yscale("log")
    combined_ax.legend()
    combined_ax.grid(True, which="both", alpha=0.3)

    save_plot(combined_fig, "time_vs_explored_all.png")


def nodes_per_second(dfs):
    combined_fig, combined_ax = plt.subplots(figsize=(10, 6))

    for df, label, color in zip(dfs, labels, colors):
        efficiency = df["explored"] / df["time_median"]

        temp_df = df.copy()
        temp_df["efficiency"] = efficiency

        median, q25, q75 = get_grouped_stats(temp_df, "efficiency")

        # Individual plot
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(median.index, median.values,
                label=label, color=color, linewidth=2)

        ax.fill_between(median.index, q25.values, q75.values,
                        color=color, alpha=0.15)

        ax.set_title(f"{label}: Node Processing Rate vs Solution Length")
        ax.set_xlabel("Optimal Solution Length (moves)")
        ax.set_ylabel("Median Nodes Per Second")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

        save_plot(fig, f"nodes_per_second_{label.replace('*', 'star')}.png")

        # Combined plot
        combined_ax.plot(median.index, median.values,
                         label=label, color=color, linewidth=2)

        combined_ax.fill_between(median.index, q25.values, q75.values,
                                 color=color, alpha=0.15)

    combined_ax.set_title("Node Processing Rate vs Solution Length: All Solvers")
    combined_ax.set_xlabel("Optimal Solution Length (moves)")
    combined_ax.set_ylabel("Median Nodes Per Second")
    combined_ax.set_yscale("log")
    combined_ax.legend()
    combined_ax.grid(True, which="both", alpha=0.3)

    save_plot(combined_fig, "nodes_per_second_all.png")

def move_num_frequency(df):
    sizes = df.groupby("moves").size()
    print(sizes)

def main():
    functions = [
        explored_vs_length,
        time_vs_length,
        time_vs_explored,
        nodes_per_second
    ]

    for func in functions:
        func(dfs)
    print(f"All graphs saved to: {OUTPUT}")


if __name__ == '__main__':
    optimized_range()
    main()