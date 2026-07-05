"""
fit.py -- turns a feature table into a difficulty equation. Two modes:

  python3 fit.py linear   features.csv "starter,beginner,intermediate,advanced,expert"
  python3 fit.py symbolic features.csv "starter,beginner,intermediate,advanced,expert"

`linear` searches for weights w such that score = sum(w_j * feature_j)
strictly separates the tiers (falling back to a soft-margin version if
strict separation is infeasible), with roughly uniform tier widths.

`symbolic` uses PySR to search NONLINEAR equation forms, with a custom
loss function that directly optimizes the same separation+uniformity
objective (not just correlation with tier order). PySR needs a Julia
runtime -- `pip install pysr && python3 -c "import pysr; pysr.install()"`
one-time first, which needs real internet access.

Both modes share: data quality/outlier checks, a stratified train/test
split, and identical separation reporting -- so results are directly
comparable.
"""

import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr


FEATURE_COLUMNS = [
    "node_count", "max_depth", "mean_depth", "depth_variance",
    "cycle_count", "num_sccs_with_size_ge_2", "largest_scc_size",
    "branching_factor", "edge_density", "max_in_degree", "leaf_count",
    "root_degree", "dependency_entropy",
    "red_distance", "fill_ratio", "long_piece_ratio", "orientation_ratio",
    "num_pieces", "pieces_in_red_row", "avg_piece_length", "length3_count",
    "symmetry_score",
]


# ============================================================
# Shared: data loading, quality checks, separation reporting
# ============================================================

def load_features(path, tier_order):
    df = pd.read_csv(path)
    missing_tiers = set(tier_order) - set(df["tier"].unique())
    if missing_tiers:
        raise ValueError(f"These tiers have zero puzzles in {path}: {missing_tiers}")
    return df[df["tier"].isin(tier_order)].reset_index(drop=True)


def tier_counts(df, tier_order):
    return {t: int((df["tier"] == t).sum()) for t in tier_order}


def flag_outliers(df, tier_order, iqr_multiplier=3.0):
    """Tukey's-rule outlier flagging within each tier, per feature. Flags
    likely mis-scans before they distort the fit -- doesn't remove anything."""
    flags = []
    for t in tier_order:
        sub = df[df["tier"] == t]
        for col in FEATURE_COLUMNS:
            q1, q3 = sub[col].quantile(0.25), sub[col].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lo, hi = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr
            bad = sub[(sub[col] < lo) | (sub[col] > hi)]
            for idx in bad.index:
                flags.append({"index": idx, "id": df.loc[idx, "id"], "tier": t,
                              "feature": col, "value": df.loc[idx, col],
                              "expected_range": f"[{lo:.2f}, {hi:.2f}]"})
    return pd.DataFrame(flags)


def data_quality_report(df, tier_order):
    print("\n=== Data quality summary ===")
    counts = tier_counts(df, tier_order)
    for t in tier_order:
        print(f"  {t:14s} n={counts[t]}")

    outliers = flag_outliers(df, tier_order)
    if len(outliers):
        print(f"\n  {len(outliers)} outlier feature-values flagged across "
              f"{outliers['id'].nunique()} puzzles (Tukey 3xIQR, per tier).")
        print("  Worth spot-checking -- often mis-scans, not real puzzles.")
        top = outliers['id'].value_counts().head(10)
        print("  Puzzles with the most flags:")
        for pid, n in top.items():
            print(f"    {pid}: {n} flagged feature(s)")
    else:
        print("\n  No outliers flagged.")
    return outliers


def tier_stats(df, scores, tier_order):
    stats = {}
    for tier in tier_order:
        mask = (df["tier"] == tier).to_numpy()
        s = scores[mask]
        stats[tier] = (s.min(), s.max(), s.mean(), s.std())
    return stats


def report_separation(df, scores, tier_order, label=""):
    stats = tier_stats(df, scores, tier_order)
    if label:
        print(f"\n--- {label} ---")

    print("\nTier score ranges:")
    widths = []
    for t in tier_order:
        lo, hi, mean, std = stats[t]
        widths.append(hi - lo)
        print(f"  {t:14s} [{lo:10.4f}, {hi:10.4f}]  width={hi-lo:10.4f}  mean={mean:8.4f}  std={std:8.4f}")

    print(f"\nWidth std dev across tiers: {np.std(widths):.4f}  (0 = perfectly uniform)")

    print("\nAdjacent tier gaps:")
    n_violations = 0
    for k in range(len(tier_order) - 1):
        gap = stats[tier_order[k + 1]][0] - stats[tier_order[k]][1]
        status = "" if gap > 0 else "  <-- OVERLAP"
        if gap <= 0:
            n_violations += 1
        print(f"  {tier_order[k]:14s} -> {tier_order[k+1]:14s}: gap = {gap:9.4f}{status}")

    tier_to_idx = {t: i for i, t in enumerate(tier_order)}
    y_ordinal = df["tier"].map(tier_to_idx).to_numpy()
    rho, pval = spearmanr(scores, y_ordinal)
    print(f"\nSpearman correlation (score vs tier ordinal): rho={rho:.4f}  (p={pval:.2e})")

    return {"widths": widths, "width_std": np.std(widths), "n_violations": n_violations, "spearman_rho": rho}


def stratified_split(df, tier_order, test_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for t in tier_order:
        idx = df.index[df["tier"] == t].to_numpy().copy()
        rng.shuffle(idx)
        n_test = max(1, int(len(idx) * test_frac))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    return np.array(train_idx), np.array(test_idx)


# ============================================================
# Linear fitting (strict separation, falls back to soft-margin)
# ============================================================

def compute_scores(df, weights):
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    return X @ weights


def tier_minmax(df, scores, tier_order):
    return {t: (scores[(df["tier"] == t).to_numpy()].min(),
                scores[(df["tier"] == t).to_numpy()].max()) for t in tier_order}


def gaps(df, scores, tier_order):
    stats = tier_minmax(df, scores, tier_order)
    return np.array([stats[tier_order[k + 1]][0] - stats[tier_order[k]][1]
                      for k in range(len(tier_order) - 1)])


def overlap_amount(df, scores, tier_order):
    return np.sum(np.maximum(0.0, -gaps(df, scores, tier_order)))


def strict_objective(w, df, tier_order, width_penalty_weight):
    w_norm = w / (np.linalg.norm(w) + 1e-9)
    scores = compute_scores(df, w_norm)
    g = gaps(df, scores, tier_order)
    stats = tier_minmax(df, scores, tier_order)
    widths = np.array([stats[t][1] - stats[t][0] for t in tier_order])
    return -(g.min()) + width_penalty_weight * np.var(widths)


def strict_constraint(w, df, tier_order):
    w_norm = w / (np.linalg.norm(w) + 1e-9)
    return gaps(df, compute_scores(df, w_norm), tier_order)


def soft_objective(w, df, tier_order, width_penalty_weight, overlap_penalty_weight):
    w_norm = w / (np.linalg.norm(w) + 1e-9)
    scores = compute_scores(df, w_norm)
    g = gaps(df, scores, tier_order)
    overlap = overlap_amount(df, scores, tier_order)
    stats = tier_minmax(df, scores, tier_order)
    widths = np.array([stats[t][1] - stats[t][0] for t in tier_order])
    return -(g.min()) + overlap_penalty_weight * overlap + width_penalty_weight * np.var(widths)


def correlation_seed(df, tier_order):
    tier_to_idx = {t: i for i, t in enumerate(tier_order)}
    y = df["tier"].map(tier_to_idx).to_numpy()
    seed = np.zeros(len(FEATURE_COLUMNS))
    for i, col in enumerate(FEATURE_COLUMNS):
        if df[col].nunique() <= 1:
            continue
        rho, _ = spearmanr(df[col].to_numpy(), y)
        seed[i] = max(rho, 0.0) if np.isfinite(rho) else 0.0
    if seed.sum() == 0:
        seed = np.ones(len(FEATURE_COLUMNS)) * 0.1
    return seed


def fit_strict(df, tier_order, width_penalty_weight, n_restarts, seed):
    rng = np.random.default_rng(seed)
    n_features = len(FEATURE_COLUMNS)
    bounds = [(0.0, None)] * n_features
    cons = [{"type": "ineq", "fun": lambda w: strict_constraint(w, df, tier_order)}]

    starts = [correlation_seed(df, tier_order)]
    starts += [rng.uniform(0.01, 1.0, size=n_features) for _ in range(n_restarts - 1)]

    best = None
    for w0 in starts:
        res = minimize(strict_objective, w0, args=(df, tier_order, width_penalty_weight),
                        method="SLSQP", bounds=bounds, constraints=cons,
                        options={"maxiter": 500, "ftol": 1e-9})
        if not np.isfinite(res.fun):
            continue
        w_norm = res.x / (np.linalg.norm(res.x) + 1e-9)
        if not all(g > 1e-9 for g in gaps(df, compute_scores(df, w_norm), tier_order)):
            continue
        if best is None or res.fun < best.fun:
            best = res
    return None if best is None else best.x / (np.linalg.norm(best.x) + 1e-9)


def fit_soft(df, tier_order, width_penalty_weight, overlap_penalty_weight, n_restarts, seed):
    rng = np.random.default_rng(seed)
    n_features = len(FEATURE_COLUMNS)
    bounds = [(0.0, None)] * n_features
    starts = [correlation_seed(df, tier_order)]
    starts += [rng.uniform(0.01, 1.0, size=n_features) for _ in range(n_restarts - 1)]

    best = None
    for w0 in starts:
        res = minimize(soft_objective, w0,
                        args=(df, tier_order, width_penalty_weight, overlap_penalty_weight),
                        method="SLSQP", bounds=bounds, options={"maxiter": 500, "ftol": 1e-9})
        if best is None or res.fun < best.fun:
            best = res
    return best.x / (np.linalg.norm(best.x) + 1e-9)


def run_linear(args):
    tier_order = args.tier_order.split(",")
    df = load_features(args.features_csv, tier_order)
    data_quality_report(df, tier_order)

    train_idx, test_idx = stratified_split(df, tier_order, seed=args.seed)
    train_df = df.iloc[train_idx].reset_index(drop=True)

    print("\n=== Attempting strict (non-overlapping) separation on TRAIN split ===")
    weights = fit_strict(train_df, tier_order, args.width_penalty, args.restarts, args.seed)

    mode = "strict"
    if weights is None:
        print("\nNo feasible strict separation found. Falling back to SOFT-MARGIN fit.")
        weights = fit_soft(train_df, tier_order, args.width_penalty, args.overlap_penalty,
                            args.restarts, args.seed)
        mode = "soft"

    print(f"\n=== Fitted weights (mode: {mode}) ===")
    for f, w in zip(FEATURE_COLUMNS, weights):
        print(f"  {f:24s} {w:.4f}")

    all_scores = compute_scores(df, weights)
    df["score"] = all_scores
    report_separation(df, all_scores, tier_order, label="FULL data (train+test)")
    report_separation(df.iloc[test_idx].reset_index(drop=True), all_scores[test_idx],
                       tier_order, label="HELD-OUT test split only (the honest check)")

    df.to_csv("scored_puzzles_linear.csv", index=False)
    print("\nWrote scored_puzzles_linear.csv")


# ============================================================
# Symbolic fitting (PySR, custom loss -- lazy-imported)
# ============================================================

OVERLAP_PENALTY_WEIGHT = 1000.0
WIDTH_VARIANCE_WEIGHT = 0.5

JULIA_LOSS_CODE = f"""
function difficulty_separation_loss(tree, dataset::Dataset{{T,L}}, options)::L where {{T,L}}
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    if !flag
        return L(Inf)
    end
    y = dataset.y
    n_tiers = 5
    lo = fill(L(Inf), n_tiers)
    hi = fill(L(-Inf), n_tiers)
    @inbounds for i in eachindex(y)
        t = Int(round(y[i])) + 1
        if prediction[i] < lo[t]
            lo[t] = prediction[i]
        end
        if prediction[i] > hi[t]
            hi[t] = prediction[i]
        end
    end
    for t in 1:n_tiers
        if lo[t] == L(Inf)
            return L(1e6)
        end
    end
    gaps = Vector{{L}}(undef, n_tiers - 1)
    for k in 1:(n_tiers - 1)
        gaps[k] = lo[k + 1] - hi[k]
    end
    min_gap = minimum(gaps)
    overlap_penalty = L(0)
    for g in gaps
        if g < L(0)
            overlap_penalty += -g
        end
    end
    widths = hi .- lo
    mean_width = sum(widths) / n_tiers
    width_var = sum((widths .- mean_width) .^ 2) / n_tiers
    return -min_gap + L({WIDTH_VARIANCE_WEIGHT}) * width_var + L({OVERLAP_PENALTY_WEIGHT}) * overlap_penalty
end
"""


def run_symbolic(args):
    from pysr import PySRRegressor  # lazy import: only needed for this mode

    tier_order = args.tier_order.split(",")
    if len(tier_order) != 5:
        raise ValueError("JULIA_LOSS_CODE hardcodes n_tiers = 5 -- edit it if you have a different number of tiers.")

    df = load_features(args.features_csv, tier_order)
    data_quality_report(df, tier_order)

    tier_to_idx = {t: i for i, t in enumerate(tier_order)}
    y = df["tier"].map(tier_to_idx).to_numpy(dtype=float)  # used only for grouping inside the custom loss
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)

    train_idx, test_idx = stratified_split(df, tier_order, seed=args.seed)

    print(f"\n=== Symbolic search settings ===")
    print(f"  iterations:              {args.iterations}")
    print(f"  populations:             {args.populations}")
    print(f"  population size:        {args.population_size}")
    print(f"  cycles per iteration:    {args.ncycles_per_iteration}")
    print(f"  max equation complexity: {args.maxsize}")
    print(f"  parallel workers:        {args.procs} ({'deterministic, single-threaded' if args.procs == 0 else 'NOT bit-reproducible run-to-run'})")
    print(f"  training puzzles:        {len(train_idx)}  (held out: {len(test_idx)})")
    print("This is a wide search -- expect it to take a while longer than a quick pass.\n")

    model = PySRRegressor(
        niterations=args.iterations,
        populations=args.populations,
        population_size=args.population_size,
        ncycles_per_iteration=args.ncycles_per_iteration,
        binary_operators=["+", "-", "*", "/", "pow"],
        unary_operators=["log", "sqrt", "square", "cube", "abs"],
        full_objective=JULIA_LOSS_CODE,
        maxsize=args.maxsize,
        parsimony=args.parsimony,
        random_state=args.seed,
        deterministic=(args.procs == 0),
        procs=args.procs,
        # Keep the whole complexity/accuracy tradeoff, not just the winner --
        # this is what lets you actually see the "trends" across equation
        # complexity rather than just one final answer.
        model_selection="best",
    )
    model.fit(X[train_idx], y[train_idx], variable_names=FEATURE_COLUMNS)

    print("\n=== Best equation found (best complexity/accuracy tradeoff) ===")
    print(model.sympy())

    all_scores = model.predict(X)
    df["score"] = all_scores
    report_separation(df, all_scores, tier_order, label="FULL data (train+test)")
    report_separation(df.iloc[test_idx].reset_index(drop=True), all_scores[test_idx],
                       tier_order, label="HELD-OUT test split only (the honest check)")

    df.to_csv("scored_puzzles_symbolic.csv", index=False)
    print("\nWrote scored_puzzles_symbolic.csv")

    # Full Pareto front -- every equation PySR found at every complexity
    # level, not just the "best" one. This is where the TRENDS live: does
    # loss keep dropping as complexity increases, or plateau early (meaning
    # a simple equation is about as good as a complicated one)?
    pareto = model.equations_[["complexity", "loss", "score", "equation"]].copy()
    pareto.to_csv("symbolic_pareto_front.csv", index=False)
    print(f"\nWrote symbolic_pareto_front.csv -- {len(pareto)} equations across "
          "the full complexity/accuracy tradeoff curve.")
    print("\nFull Pareto front:")
    print(pareto.to_string(index=False))


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p_lin = sub.add_parser("linear", help="Linear weighted-sum fit (exact separation guarantee if feasible)")
    p_lin.add_argument("features_csv")
    p_lin.add_argument("tier_order", help="comma-separated, easiest first")
    p_lin.add_argument("--restarts", type=int, default=30)
    p_lin.add_argument("--width-penalty", type=float, default=0.5)
    p_lin.add_argument("--overlap-penalty", type=float, default=5.0, help="only used if strict separation is infeasible")
    p_lin.add_argument("--seed", type=int, default=0)

    p_sym = sub.add_parser("symbolic", help="PySR nonlinear fit (needs Julia -- see module docstring)")
    p_sym.add_argument("features_csv")
    p_sym.add_argument("tier_order", help="comma-separated, easiest first -- must be exactly 5 tiers")
    p_sym.add_argument("--iterations", type=int, default=2000,
                        help="outer search iterations -- higher = more thorough (default: 2000, a real overnight-scale search)")
    p_sym.add_argument("--populations", type=int, default=60,
                        help="number of parallel equation populations exploring the search space (default: 60)")
    p_sym.add_argument("--population-size", type=int, default=50,
                        help="equations per population (default: 50) -- populations x population_size = total equations evolving each generation")
    p_sym.add_argument("--ncycles-per-iteration", type=int, default=550,
                        help="mutation/selection cycles per iteration -- higher = more refinement per iteration (default: 550)")
    p_sym.add_argument("--maxsize", type=int, default=30,
                        help="max equation complexity/node count to explore (default: 30, higher than the quick-pass default of 20)")
    p_sym.add_argument("--parsimony", type=float, default=0.005,
                        help="penalty per unit of complexity -- lower value lets more complex equations compete (default: 0.005)")
    p_sym.add_argument("--procs", type=int, default=4,
                        help="parallel worker processes (default: 4). Set to 0 for single-threaded deterministic mode (slower, but bit-reproducible)")
    p_sym.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    if args.command == "linear":
        run_linear(args)
    elif args.command == "symbolic":
        run_symbolic(args)


if __name__ == "__main__":
    main()
