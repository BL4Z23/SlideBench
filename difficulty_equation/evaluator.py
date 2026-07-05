"""
evaluator.py -- takes the held-out 20% (held_out_puzzles.csv, written by
equation.py's save_split()) and checks how each fitted equation actually
performs on data it never saw during fitting.

  python3 evaluator.py linear   held_out_puzzles.csv fitted_weights_linear.csv       "starter,beginner,intermediate,advanced,expert"
  python3 evaluator.py symbolic held_out_puzzles.csv symbolic_pareto_front.csv       "starter,beginner,intermediate,advanced,expert"

`linear` loads the single fitted weight vector and reports separation on
the held-out set.

`symbolic` loads the FULL Pareto front (every equation PySR found, at
every complexity) and evaluates *each one* against the held-out set --
so you can see which equations actually generalize vs. which only looked
good on the 80% they were fit on.
"""

import argparse
import numpy as np
import pandas as pd
import sympy

from equation import FEATURE_COLUMNS, tier_stats, load_features


# ============================================================
# Shared: separation metrics computed against held-out data
# ============================================================

def held_out_report(df, scores, tier_order):
    """Same metrics as equation.py's report_separation(), but returned as a
    dict (no printing) so we can tabulate many equations at once."""
    stats = tier_stats(df, scores, tier_order)
    widths = [stats[t][1] - stats[t][0] for t in tier_order]

    n_violations = 0
    min_gap = np.inf
    for k in range(len(tier_order) - 1):
        gap = stats[tier_order[k + 1]][0] - stats[tier_order[k]][1]
        min_gap = min(min_gap, gap)
        if gap <= 0:
            n_violations += 1

    tier_to_idx = {t: i for i, t in enumerate(tier_order)}
    y_ordinal = df["tier"].map(tier_to_idx).to_numpy()
    from scipy.stats import spearmanr
    rho, _ = spearmanr(scores, y_ordinal)

    return {
        "width_std": float(np.std(widths)),
        "min_gap": float(min_gap),
        "n_violations": int(n_violations),
        "spearman_rho": float(rho),
    }


def print_report(df, scores, tier_order, label=""):
    if label:
        print(f"\n--- {label} ---")
    stats = tier_stats(df, scores, tier_order)
    for t in tier_order:
        lo, hi, mean, std = stats[t]
        print(f"  {t:14s} [{lo:10.4f}, {hi:10.4f}]  width={hi-lo:10.4f}  mean={mean:8.4f}  std={std:8.4f}")
    r = held_out_report(df, scores, tier_order)
    status = "PASS -- fully separated" if r["n_violations"] == 0 else f"FAIL -- {r['n_violations']} overlapping gap(s)"
    print(f"  min adjacent gap: {r['min_gap']:.4f}   spearman rho: {r['spearman_rho']:.4f}   {status}")
    return r


# ============================================================
# Linear: load a single weight vector, verify it
# ============================================================

def run_linear(args):
    tier_order = args.tier_order.split(",")
    df = load_features(args.held_out_csv, tier_order)
    weights_df = pd.read_csv(args.model_csv)
    weights_df = weights_df.set_index("feature").loc[FEATURE_COLUMNS]  # enforce order
    w = weights_df["weight"].to_numpy(dtype=float)

    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    scores = X @ w
    df["held_out_score"] = scores

    print(f"\n=== Verifying linear equation against {len(df)} held-out puzzles ===")
    print_report(df, scores, tier_order, label="HELD-OUT verification")

    df.to_csv("held_out_verified_linear.csv", index=False)
    print("\nWrote held_out_verified_linear.csv")


# ============================================================
# Symbolic: load the full Pareto front, verify every equation
# ============================================================

def run_symbolic(args):
    tier_order = args.tier_order.split(",")
    df = load_features(args.held_out_csv, tier_order)
    pareto = pd.read_csv(args.model_csv)  # symbolic_pareto_front.csv: complexity, loss, score, equation

    symbols = {name: sympy.Symbol(name) for name in FEATURE_COLUMNS}
    X = df[FEATURE_COLUMNS]

    print(f"\n=== Verifying {len(pareto)} equations (full Pareto front) against "
          f"{len(df)} held-out puzzles never used for fitting ===")

    results = []
    for _, row in pareto.iterrows():
        eq_str = row["equation"]
        try:
            expr = sympy.sympify(eq_str, locals=symbols)
            fn = sympy.lambdify([symbols[c] for c in FEATURE_COLUMNS], expr, modules=["numpy"])
            scores = np.asarray(fn(*[X[c].to_numpy(dtype=float) for c in FEATURE_COLUMNS]), dtype=float)
            if scores.ndim == 0:
                scores = np.full(len(df), float(scores))
            r = held_out_report(df, scores, tier_order)
            r.update({"complexity": row["complexity"], "train_loss": row["loss"], "equation": eq_str})
            results.append(r)
        except Exception as e:
            results.append({"complexity": row["complexity"], "train_loss": row["loss"],
                             "equation": eq_str, "width_std": np.nan, "min_gap": np.nan,
                             "n_violations": -1, "spearman_rho": np.nan, "error": str(e)})

    out = pd.DataFrame(results).sort_values("complexity")
    out.to_csv("held_out_verification.csv", index=False)

    print("\nPer-equation held-out verification (sorted by complexity):")
    display_cols = ["complexity", "train_loss", "min_gap", "n_violations", "spearman_rho", "equation"]
    print(out[display_cols].to_string(index=False))

    generalizes = out[(out["n_violations"] == 0)]
    if len(generalizes):
        best = generalizes.sort_values("spearman_rho", ascending=False).iloc[0]
        print(f"\nBest equation that still fully separates tiers on held-out data "
              f"(complexity {int(best['complexity'])}, rho={best['spearman_rho']:.4f}):")
        print(f"  {best['equation']}")
    else:
        print("\nNo equation in the Pareto front achieves full separation on the "
              "held-out set -- the best-on-train equation may be overfit. Check "
              "held_out_verification.csv for the closest candidates (lowest n_violations).")

    print("\nWrote held_out_verification.csv")


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p_lin = sub.add_parser("linear", help="Verify a single fitted linear weight vector")
    p_lin.add_argument("held_out_csv", help="held_out_puzzles.csv from equation.py")
    p_lin.add_argument("model_csv", help="fitted_weights_linear.csv from equation.py")
    p_lin.add_argument("tier_order", help="comma-separated, easiest first")

    p_sym = sub.add_parser("symbolic", help="Verify every equation in a PySR Pareto front")
    p_sym.add_argument("held_out_csv", help="held_out_puzzles.csv from equation.py")
    p_sym.add_argument("model_csv", help="symbolic_pareto_front.csv from equation.py")
    p_sym.add_argument("tier_order", help="comma-separated, easiest first")

    args = ap.parse_args()
    if args.command == "linear":
        run_linear(args)
    elif args.command == "symbolic":
        run_symbolic(args)


if __name__ == "__main__":
    main()
