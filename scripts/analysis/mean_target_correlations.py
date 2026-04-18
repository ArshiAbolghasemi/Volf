from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.model.common.preprocessing import (
    build_forecasting_design_matrix,
    split_design_matrix_xy,
)
from src.model.har.types import HARFeatureConfig

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute feature correlations with a mean RV target and plot "
            "top correlated features."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/ag/v4.csv",
        help="Path to input CSV.",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="wheat_weekly_rv",
        help="Base RV target column.",
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default="4,8,12,16",
        help="Comma-separated mean target horizons (e.g. '4,8,12,16').",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Number of top features by absolute correlation.",
    )
    parser.add_argument(
        "--target-transform",
        type=str,
        choices=["none", "log"],
        default="none",
        help="Apply transform before mean-target construction.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/analysis/correlations",
        help="Directory for outputs (CSV + PNG).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.top_n < 1:
        msg = "--top-n must be >= 1."
        raise ValueError(msg)

    horizon_tokens = [token.strip() for token in args.horizons.split(",")]
    horizons = sorted({int(token) for token in horizon_tokens if token})
    if not horizons:
        msg = "--horizons must contain at least one integer."
        raise ValueError(msg)
    if any(horizon < 1 for horizon in horizons):
        msg = "--horizons values must be >= 1 for target_mode='mean'."
        raise ValueError(msg)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(input_path)

    numeric_features = [
        col
        for col in data.columns
        if col != args.target_col and pd.api.types.is_numeric_dtype(data[col])
    ]
    if not numeric_features:
        msg = "No numeric feature columns found in input data."
        raise ValueError(msg)

    sns.set_theme(style="whitegrid")

    for horizon in horizons:
        feature_cfg = HARFeatureConfig(
            target_col=args.target_col,
            core_columns=[args.target_col],
            target_horizon=horizon,
            target_mode="mean",
            extra_feature_cols=numeric_features,
        )

        design, _, target_name = build_forecasting_design_matrix(
            data,
            feature_cfg,
            target_transform=args.target_transform,
        )
        x, y = split_design_matrix_xy(design, target_name)

        x_features = x.drop(columns=[args.target_col], errors="ignore")
        x_features = x_features.select_dtypes(include=["number"])

        corr_with_target = x_features.corrwith(y)
        corr_df = (
            corr_with_target.rename("corr")
            .to_frame()
            .assign(abs_corr=lambda frame: frame["corr"].abs())
            .sort_values("abs_corr", ascending=False)
        )

        top_n = min(args.top_n, len(corr_df))
        top_df = corr_df.head(top_n).copy().sort_values("corr", ascending=True)

        csv_path = output_dir / (
            f"mean_target_corr_{args.target_col}_h{horizon}_{args.target_transform}.csv"
        )
        corr_df.to_csv(csv_path, index=True)

        plt.figure(figsize=(10, max(6, 0.45 * top_n)))
        colors = ["#2a9d8f" if value >= 0 else "#e76f51" for value in top_df["corr"]]
        plt.barh(top_df.index, top_df["corr"], color=colors)
        plt.axvline(0.0, color="black", linewidth=1)
        plt.xlabel("Correlation with target")
        plt.ylabel("Feature")
        target_formula = r"$y_t=\frac{1}{h}\sum_{i=1}^{h}\log(\mathrm{RV}_{t+i})$"
        title = f"Top {top_n} features vs {target_formula} (h={horizon})"
        plt.title(title)
        plt.tight_layout()

        plot_path = output_dir / (
            f"mean_target_corr_top{top_n}_{args.target_col}_h{horizon}_{args.target_transform}.png"
        )
        plt.savefig(plot_path, dpi=180)
        plt.close()

        logger.info("Saved full correlation table: %s", csv_path)
        logger.info("Saved top-%d plot: %s", top_n, plot_path)


if __name__ == "__main__":
    main()
