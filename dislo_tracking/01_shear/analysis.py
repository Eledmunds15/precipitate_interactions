import pandas as pd
import numpy as np

def generate_summary(df_stats, df_interp):
    """Generate summary table of dislocation positions and total line length."""
    if df_interp.empty:
        df_summary = df_stats[["Step","total_line_length"]].copy()
        df_summary["x_pos"] = np.nan
        df_summary["x_std"] = np.nan
        return df_summary

    grouped = df_interp.groupby("Step")["x"]
    means = grouped.mean().values
    stds = grouped.std().values
    steps = grouped.mean().index.values

    lx_val = df_stats["lx"].iloc[0]
    dx = np.diff(means, prepend=means[0])
    jumps = np.round(dx / lx_val)
    unwrapped_x = means - np.cumsum(jumps) * lx_val

    summary_data = []
    for i, step in enumerate(steps):
        row = df_stats[df_stats["Step"] == step].iloc[0]
        summary_data.append({
            "Step": step,
            "x_pos": unwrapped_x[i],
            "x_std": stds[i],
            "total_line_length": row["total_line_length"]
        })
    return pd.DataFrame(summary_data)