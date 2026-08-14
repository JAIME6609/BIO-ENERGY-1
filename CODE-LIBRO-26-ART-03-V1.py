"""
AI-Optimized Waste-to-Energy Systems for Circular Bioenergy and Emission Governance
=================================================================================

This script generates a reproducible computational experiment aligned with the
chapter architecture developed in the manuscript. It creates a synthetic but
mechanistically informed waste-to-energy data environment, trains predictive
machine-learning models, conducts multi-objective optimization of candidate
waste-to-energy configurations, and produces tables and figures organized in
three folders corresponding to manuscript Sections 5.1, 5.2, and 5.3.

The script is intentionally self-contained. It does not require proprietary data;
instead, it uses transparent process assumptions to build a testbed for methods
that can later be replaced by plant-level SCADA, laboratory, lifecycle inventory,
IoT, and municipal waste-flow records.

Outputs
-------
results/5.1  Predictive intelligence and yield-emission modeling
results/5.2  Multi-objective optimization and Pareto governance
results/5.3  AI circularity score, explainability, and decision accountability

Authoring note
--------------
All variables and formulas are defined inside the script so that the experiment
can be audited, modified, and rerun in a research environment.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import textwrap
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


REQUIRED_PACKAGES: Dict[str, str] = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "sklearn": "scikit-learn",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "shap": "shap",
    "scipy": "scipy",
}


def check_dependencies() -> None:
    """Verify that all required packages are available before running the experiment."""
    missing = []
    for module_name, package_name in REQUIRED_PACKAGES.items():
        if importlib.util.find_spec(module_name) is None:
            missing.append(package_name)
    if missing:
        unique_missing = sorted(set(missing))
        print("The following required packages are missing:\n")
        for package in unique_missing:
            print(f"  - {package}")
        print("\nInstall them with the following command and rerun the script:\n")
        print("  python -m pip install " + " ".join(unique_missing))
        sys.exit(1)


check_dependencies()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from lightgbm import LGBMRegressor
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("PYTHONHASHSEED", "42")


RANDOM_SEED = 42
ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
SECTION_51 = RESULTS_DIR / "5.1"
SECTION_52 = RESULTS_DIR / "5.2"
SECTION_53 = RESULTS_DIR / "5.3"
for directory in (SECTION_51, SECTION_52, SECTION_53):
    directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class RouteParameters:
    """Route-specific constants used in the mechanistic data generator."""

    energy_factor: float
    emission_base: float
    cost_base: float
    material_recovery_base: float
    pollutant_risk: float
    reliability_base: float


ROUTE_PARAMETERS: Dict[str, RouteParameters] = {
    "anaerobic_digestion": RouteParameters(0.82, 165.0, 42.0, 0.66, 0.42, 0.78),
    "gasification": RouteParameters(0.74, 245.0, 69.0, 0.33, 0.61, 0.70),
    "pyrolysis": RouteParameters(0.63, 205.0, 61.0, 0.48, 0.52, 0.72),
    "incineration_energy_recovery": RouteParameters(0.70, 315.0, 58.0, 0.18, 0.82, 0.76),
    "hybrid_biorefinery": RouteParameters(0.91, 138.0, 83.0, 0.74, 0.36, 0.81),
}


FEEDSTOCK_PROFILES: Dict[str, Dict[str, float]] = {
    "food_waste": {"moisture": 72, "organic": 82, "ash": 6, "cn": 18, "lhv": 8.3, "contamination": 4},
    "agricultural_residues": {"moisture": 28, "organic": 71, "ash": 9, "cn": 42, "lhv": 15.2, "contamination": 3},
    "sewage_sludge": {"moisture": 78, "organic": 61, "ash": 18, "cn": 11, "lhv": 6.5, "contamination": 8},
    "msw_organic_fraction": {"moisture": 54, "organic": 68, "ash": 14, "cn": 25, "lhv": 11.6, "contamination": 12},
    "industrial_organic_residues": {"moisture": 46, "organic": 77, "ash": 8, "cn": 23, "lhv": 12.8, "contamination": 7},
    "mixed_biomass_stream": {"moisture": 40, "organic": 64, "ash": 12, "cn": 31, "lhv": 13.7, "contamination": 9},
}


COMPATIBLE_ROUTES: Dict[str, List[str]] = {
    "food_waste": ["anaerobic_digestion", "hybrid_biorefinery", "incineration_energy_recovery"],
    "agricultural_residues": ["gasification", "pyrolysis", "hybrid_biorefinery"],
    "sewage_sludge": ["anaerobic_digestion", "pyrolysis", "incineration_energy_recovery", "hybrid_biorefinery"],
    "msw_organic_fraction": ["incineration_energy_recovery", "gasification", "pyrolysis", "hybrid_biorefinery"],
    "industrial_organic_residues": ["anaerobic_digestion", "gasification", "hybrid_biorefinery"],
    "mixed_biomass_stream": ["gasification", "pyrolysis", "hybrid_biorefinery", "incineration_energy_recovery"],
}


NUMERIC_FEATURES = [
    "moisture_pct",
    "organic_fraction_pct",
    "ash_pct",
    "carbon_nitrogen_ratio",
    "lower_heating_value_mj_kg",
    "contamination_pct",
    "temperature_c",
    "hydraulic_retention_time_d",
    "residence_time_min",
    "oxygen_equivalence_ratio",
    "ph",
    "pretreatment_intensity",
    "sensor_coverage",
    "data_quality_index",
    "collection_distance_km",
    "seasonality_index",
    "local_energy_demand_index",
    "vulnerable_population_index",
    "sorting_efficiency",
]

CATEGORICAL_FEATURES = ["feedstock_type", "conversion_route"]
TARGETS = ["net_energy_mwh_t", "net_emissions_kgco2e_t", "ai_circularity_score"]


def bounded_normal(mean: float, sd: float, low: float, high: float, size: int, rng: np.random.Generator) -> np.ndarray:
    """Draw from a normal distribution and clip values to physically plausible bounds."""
    return np.clip(rng.normal(mean, sd, size), low, high)


def normalize_series(values: pd.Series) -> pd.Series:
    """Min-max normalization with protection against degenerate ranges."""
    values = values.astype(float)
    span = values.max() - values.min()
    if span <= 1e-12:
        return pd.Series(np.ones(len(values)) * 0.5, index=values.index)
    return (values - values.min()) / span


def weighted_choice(rng: np.random.Generator, options: List[str]) -> str:
    """Prefer circular routes while preserving route diversity."""
    base = np.ones(len(options), dtype=float)
    for idx, route in enumerate(options):
        if route == "hybrid_biorefinery":
            base[idx] += 0.75
        elif route == "anaerobic_digestion":
            base[idx] += 0.45
        elif route == "incineration_energy_recovery":
            base[idx] -= 0.20
    probabilities = base / base.sum()
    return str(rng.choice(options, p=probabilities))


def simulate_wte_dataset(n_samples: int = 1600, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate a mechanistically informed synthetic dataset.

    The generator combines feedstock chemistry, route-specific conversion logic,
    operational parameters, sensing quality, emissions, cost, and environmental
    justice variables. It is not intended to replace empirical plant data; it is a
    transparent computational scaffold for methodological development.
    """
    rng = np.random.default_rng(seed)
    feedstocks = np.array(list(FEEDSTOCK_PROFILES.keys()))
    feedstock_probabilities = np.array([0.22, 0.17, 0.16, 0.20, 0.13, 0.12])

    records: List[Dict[str, float | str]] = []
    for i in range(n_samples):
        feedstock = str(rng.choice(feedstocks, p=feedstock_probabilities))
        profile = FEEDSTOCK_PROFILES[feedstock]
        route = weighted_choice(rng, COMPATIBLE_ROUTES[feedstock])
        params = ROUTE_PARAMETERS[route]

        seasonality = float(np.sin(2 * np.pi * (i % 365) / 365) + rng.normal(0, 0.12))
        moisture = float(np.clip(rng.normal(profile["moisture"] + 3.2 * seasonality, 6.0), 8, 88))
        organic = float(np.clip(rng.normal(profile["organic"] - 0.08 * moisture, 5.0), 35, 93))
        ash = float(np.clip(rng.normal(profile["ash"] + 0.05 * moisture, 3.2), 2, 30))
        cn_ratio = float(np.clip(rng.normal(profile["cn"], 5.5), 6, 58))
        lhv = float(np.clip(rng.normal(profile["lhv"] - 0.035 * moisture + 0.055 * organic, 1.3), 3.5, 22.0))
        contamination = float(np.clip(rng.normal(profile["contamination"], 3.0), 0.5, 24.0))

        temperature = float(np.clip(rng.normal(38 if route == "anaerobic_digestion" else 650, 5 if route == "anaerobic_digestion" else 115), 25, 950))
        hrt = float(np.clip(rng.normal(23 if route in ["anaerobic_digestion", "hybrid_biorefinery"] else 4, 5), 1, 45))
        residence_time = float(np.clip(rng.normal(45 if route != "anaerobic_digestion" else 8, 12), 3, 90))
        oxygen_ratio = float(np.clip(rng.normal(0.27 if route == "gasification" else 0.08 if route == "pyrolysis" else 0.55 if route == "incineration_energy_recovery" else 0.18, 0.07), 0.01, 0.95))
        ph = float(np.clip(rng.normal(7.15 if route in ["anaerobic_digestion", "hybrid_biorefinery"] else 6.8, 0.35), 5.5, 8.6))
        pretreatment = float(np.clip(rng.beta(2.2, 2.7), 0.02, 0.98))
        sensor_coverage = float(np.clip(rng.beta(4.2, 1.7), 0.25, 0.99))
        data_quality = float(np.clip(0.55 * sensor_coverage + 0.45 * rng.beta(5.0, 2.0), 0.20, 0.99))
        distance = float(np.clip(rng.gamma(shape=3.2, scale=9.0), 2.0, 88.0))
        demand = float(np.clip(rng.normal(0.68, 0.18), 0.15, 1.0))
        vulnerability = float(np.clip(rng.beta(2.1, 3.4), 0.02, 0.98))
        sorting = float(np.clip(rng.beta(4.0, 2.5) - 0.003 * contamination, 0.18, 0.98))

        anaerobic_efficiency = expit((ph - 6.55) * 2.0) * expit((temperature - 33.5) / 2.7) * (1 - np.exp(-hrt / 17.0))
        thermal_efficiency = expit((lhv - 6.8) / 2.1) * (1 - moisture / 115.0) * (1 - ash / 75.0)
        pretreatment_gain = 1.0 + 0.16 * pretreatment + 0.05 * sorting
        contamination_penalty = 1.0 - 0.018 * contamination
        demand_factor = 0.88 + 0.18 * demand

        if route == "anaerobic_digestion":
            route_energy = 0.0185 * organic * anaerobic_efficiency * pretreatment_gain * contamination_penalty
        elif route == "gasification":
            route_energy = 0.086 * lhv * thermal_efficiency * (1 + 0.12 * oxygen_ratio) * contamination_penalty
        elif route == "pyrolysis":
            route_energy = 0.071 * lhv * thermal_efficiency * (1 + 0.25 * pretreatment) * (1 - 0.40 * oxygen_ratio)
        elif route == "incineration_energy_recovery":
            route_energy = 0.080 * lhv * thermal_efficiency * (1 + 0.05 * sorting) * contamination_penalty
        else:
            biological_component = 0.012 * organic * anaerobic_efficiency
            thermal_component = 0.050 * lhv * thermal_efficiency
            route_energy = (biological_component + thermal_component) * pretreatment_gain * contamination_penalty

        net_energy = float(np.clip(params.energy_factor * route_energy * demand_factor + rng.normal(0, 0.045), 0.05, 1.85))
        avoided_landfill = 82.0 * (organic / 100.0) * (1 - moisture / 120.0) + 52.0 * sorting
        fossil_substitution = 138.0 * net_energy * demand_factor
        transport_emissions = 0.88 * distance
        pollution_term = 2.4 * contamination + 1.2 * ash + 24.0 * params.pollutant_risk
        net_emissions = float(params.emission_base + transport_emissions + pollution_term - avoided_landfill - fossil_substitution + rng.normal(0, 18.0))
        net_emissions = float(np.clip(net_emissions, -165, 520))

        reliability = float(np.clip(params.reliability_base + 0.16 * sensor_coverage + 0.11 * data_quality - 0.008 * contamination - 0.0025 * abs(cn_ratio - 25), 0.25, 0.98))
        cost = float(np.clip(params.cost_base + 0.62 * distance + 1.30 * contamination + 18 * pretreatment + rng.normal(0, 6.0), 25, 180))
        material_recovery = float(np.clip(params.material_recovery_base + 0.25 * sorting + 0.08 * pretreatment - 0.007 * contamination, 0.03, 0.98))
        exposure = float(np.clip(vulnerability * params.pollutant_risk * (0.75 + contamination / 30.0) * (1.15 - 0.55 * sensor_coverage), 0.01, 1.0))
        carbon_avoidance = float(np.clip((220.0 - net_emissions) / 385.0, 0, 1))
        economic_viability = float(np.clip(1.0 - cost / 190.0 + 0.22 * net_energy, 0, 1))
        justice_performance = float(np.clip(1.0 - exposure, 0, 1))

        circularity = 100.0 * (
            0.22 * min(net_energy / 1.45, 1.0)
            + 0.22 * carbon_avoidance
            + 0.18 * material_recovery
            + 0.15 * reliability
            + 0.12 * economic_viability
            + 0.11 * justice_performance
        )
        circularity = float(np.clip(circularity + rng.normal(0, 2.2), 0, 100))

        records.append(
            {
                "feedstock_type": feedstock,
                "conversion_route": route,
                "moisture_pct": moisture,
                "organic_fraction_pct": organic,
                "ash_pct": ash,
                "carbon_nitrogen_ratio": cn_ratio,
                "lower_heating_value_mj_kg": lhv,
                "contamination_pct": contamination,
                "temperature_c": temperature,
                "hydraulic_retention_time_d": hrt,
                "residence_time_min": residence_time,
                "oxygen_equivalence_ratio": oxygen_ratio,
                "ph": ph,
                "pretreatment_intensity": pretreatment,
                "sensor_coverage": sensor_coverage,
                "data_quality_index": data_quality,
                "collection_distance_km": distance,
                "seasonality_index": seasonality,
                "local_energy_demand_index": demand,
                "vulnerable_population_index": vulnerability,
                "sorting_efficiency": sorting,
                "net_energy_mwh_t": net_energy,
                "net_emissions_kgco2e_t": net_emissions,
                "operating_cost_usd_t": cost,
                "material_recovery_index": material_recovery,
                "reliability_index": reliability,
                "exposure_index": exposure,
                "carbon_avoidance_index": carbon_avoidance,
                "economic_viability_index": economic_viability,
                "environmental_justice_index": justice_performance,
                "ai_circularity_score": circularity,
            }
        )

    data = pd.DataFrame.from_records(records)
    return data


def build_preprocessor() -> ColumnTransformer:
    """Create a preprocessing graph for mixed numerical and categorical variables."""
    try:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # Compatibility with older scikit-learn versions.
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", one_hot, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def make_models(seed: int = RANDOM_SEED) -> Dict[str, object]:
    """Define a suite of tree-based regressors suitable for nonlinear process modeling."""
    return {
        "Random Forest": RandomForestRegressor(n_estimators=120, random_state=seed, min_samples_leaf=3, n_jobs=1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=150, random_state=seed, min_samples_leaf=2, n_jobs=1),
        "XGBoost": XGBRegressor(
            n_estimators=160,
            max_depth=5,
            learning_rate=0.045,
            subsample=0.90,
            colsample_bytree=0.90,
            reg_lambda=1.4,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=seed,
            n_jobs=1,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=180,
            learning_rate=0.035,
            num_leaves=31,
            subsample=0.92,
            colsample_bytree=0.92,
            min_child_samples=18,
            random_state=seed,
            n_jobs=1,
            verbose=-1,
        ),
    }


def fit_predictive_models(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Pipeline], pd.DataFrame, pd.DataFrame]:
    """Train separate model pipelines for energy yield and net emissions."""
    X = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_energy = data["net_energy_mwh_t"]
    y_emissions = data["net_emissions_kgco2e_t"]
    X_train, X_test, y_energy_train, y_energy_test, y_em_train, y_em_test = train_test_split(
        X, y_energy, y_emissions, test_size=0.25, random_state=RANDOM_SEED
    )

    metrics = []
    fitted: Dict[str, Pipeline] = {}
    predictions = pd.DataFrame(index=X_test.index)
    predictions["observed_energy"] = y_energy_test
    predictions["observed_emissions"] = y_em_test

    for target_name, y_train, y_test, observed_col, unit in [
        ("net_energy_mwh_t", y_energy_train, y_energy_test, "observed_energy", "MWh t^-1"),
        ("net_emissions_kgco2e_t", y_em_train, y_em_test, "observed_emissions", "kg CO2e t^-1"),
    ]:
        for model_name, model in make_models().items():
            pipeline = Pipeline(steps=[("preprocess", build_preprocessor()), ("model", model)])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            rmse = float(math.sqrt(mean_squared_error(y_test, y_pred)))
            mae = float(mean_absolute_error(y_test, y_pred))
            r2 = float(r2_score(y_test, y_pred))
            metrics.append(
                {
                    "target": target_name,
                    "model": model_name,
                    "R2": r2,
                    "RMSE": rmse,
                    "MAE": mae,
                    "unit": unit,
                }
            )
            fitted[f"{target_name}::{model_name}"] = pipeline
            predictions[f"{target_name}::{model_name}"] = y_pred

    metrics_df = pd.DataFrame(metrics).sort_values(["target", "R2"], ascending=[True, False])
    best_energy_model_name = metrics_df[metrics_df["target"] == "net_energy_mwh_t"].iloc[0]["model"]
    best_em_model_name = metrics_df[metrics_df["target"] == "net_emissions_kgco2e_t"].iloc[0]["model"]
    best_summary = pd.DataFrame(
        [
            {"target": "net_energy_mwh_t", "best_model": best_energy_model_name},
            {"target": "net_emissions_kgco2e_t", "best_model": best_em_model_name},
        ]
    )
    return metrics_df, fitted, best_summary, predictions


def pareto_front(df: pd.DataFrame, objective_columns: List[str]) -> pd.Series:
    """Return a Boolean mask for non-dominated solutions.

    All objective columns must be expressed as quantities to be maximized. The
    implementation is quadratic but efficient enough for the candidate population
    used in this reproducible experiment.
    """
    values = df[objective_columns].to_numpy(dtype=float)
    n = values.shape[0]
    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_efficient[i]:
            continue
        dominates_i = np.all(values >= values[i], axis=1) & np.any(values > values[i], axis=1)
        if np.any(dominates_i):
            is_efficient[i] = False
    return pd.Series(is_efficient, index=df.index)


def perform_optimization(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate candidate scenarios and extract a Pareto-efficient decision set."""
    candidates = simulate_wte_dataset(n_samples=2200, seed=RANDOM_SEED + 100)
    candidates["energy_objective"] = normalize_series(candidates["net_energy_mwh_t"])
    candidates["emission_objective"] = 1.0 - normalize_series(candidates["net_emissions_kgco2e_t"])
    candidates["cost_objective"] = 1.0 - normalize_series(candidates["operating_cost_usd_t"])
    candidates["exposure_objective"] = 1.0 - normalize_series(candidates["exposure_index"])
    candidates["reliability_objective"] = normalize_series(candidates["reliability_index"])
    candidates["circularity_objective"] = normalize_series(candidates["ai_circularity_score"])
    objectives = [
        "energy_objective",
        "emission_objective",
        "cost_objective",
        "exposure_objective",
        "reliability_objective",
        "circularity_objective",
    ]
    candidates["pareto_efficient"] = pareto_front(candidates, objectives)
    candidates["composite_governance_score"] = 100.0 * (
        0.23 * candidates["energy_objective"]
        + 0.22 * candidates["emission_objective"]
        + 0.14 * candidates["cost_objective"]
        + 0.15 * candidates["exposure_objective"]
        + 0.12 * candidates["reliability_objective"]
        + 0.14 * candidates["circularity_objective"]
    )
    pareto = candidates[candidates["pareto_efficient"]].copy()
    pareto = pareto.sort_values("composite_governance_score", ascending=False)
    return candidates, pareto


def summarize_by_route(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate governance-relevant indicators by conversion route."""
    summary = (
        df.groupby("conversion_route")
        .agg(
            mean_energy_mwh_t=("net_energy_mwh_t", "mean"),
            mean_net_emissions_kgco2e_t=("net_emissions_kgco2e_t", "mean"),
            mean_operating_cost_usd_t=("operating_cost_usd_t", "mean"),
            mean_circularity_score=("ai_circularity_score", "mean"),
            mean_reliability_index=("reliability_index", "mean"),
            mean_environmental_justice_index=("environmental_justice_index", "mean"),
            mean_material_recovery_index=("material_recovery_index", "mean"),
            n=("conversion_route", "size"),
        )
        .reset_index()
        .sort_values("mean_circularity_score", ascending=False)
    )
    return summary


def get_feature_names(pipeline: Pipeline) -> List[str]:
    """Recover transformed feature names from a fitted preprocessing pipeline."""
    preprocessor = pipeline.named_steps["preprocess"]
    numeric_names = list(NUMERIC_FEATURES)
    categorical_encoder = preprocessor.named_transformers_["cat"]
    categorical_names = list(categorical_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    return numeric_names + categorical_names


def compute_shap_importance(pipeline: Pipeline, data: pd.DataFrame, sample_size: int = 220) -> pd.DataFrame:
    """Compute SHAP feature importance for the selected tree-based model."""
    X = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    sample = X.sample(n=min(sample_size, len(X)), random_state=RANDOM_SEED)
    transformed = pipeline.named_steps["preprocess"].transform(sample)
    model = pipeline.named_steps["model"]
    feature_names = get_feature_names(pipeline)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(transformed)
    shap_array = np.asarray(shap_values)
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": np.abs(shap_array).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    return importance


def save_table_png(df: pd.DataFrame, path: Path, title: str, max_rows: int = 10) -> None:
    """Save a compact table as a high-resolution PNG for direct manuscript insertion."""
    table_df = df.head(max_rows).copy()
    fig_height = max(2.2, 0.38 * len(table_df) + 1.2)
    fig_width = max(8.0, 1.25 * len(table_df.columns))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(title, pad=12, fontsize=11)
    display_df = table_df.copy()
    for col in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[col]):
            display_df[col] = display_df[col].map(lambda v: f"{v:.3f}")
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_figures_and_tables(
    data: pd.DataFrame,
    metrics_df: pd.DataFrame,
    best_summary: pd.DataFrame,
    predictions: pd.DataFrame,
    fitted: Dict[str, Pipeline],
    candidates: pd.DataFrame,
    pareto: pd.DataFrame,
    shap_importance: pd.DataFrame,
) -> None:
    """Create all article-ready tables and figures in the section folders."""
    # ------------------------- Section 5.1 -------------------------
    metrics_df.to_csv(SECTION_51 / "table_1_predictive_model_metrics.csv", index=False)
    best_summary.to_csv(SECTION_51 / "best_model_summary.csv", index=False)
    save_table_png(
        metrics_df.round(4),
        SECTION_51 / "table_1_predictive_model_metrics.png",
        "Table 1. Predictive model performance for energy yield and net emissions",
        max_rows=8,
    )

    best_energy = best_summary.loc[best_summary["target"] == "net_energy_mwh_t", "best_model"].iloc[0]
    best_em = best_summary.loc[best_summary["target"] == "net_emissions_kgco2e_t", "best_model"].iloc[0]

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    y_obs = predictions["observed_energy"]
    y_pred = predictions[f"net_energy_mwh_t::{best_energy}"]
    ax.scatter(y_obs, y_pred, alpha=0.55, s=22)
    lims = [min(y_obs.min(), y_pred.min()), max(y_obs.max(), y_pred.max())]
    ax.plot(lims, lims, linewidth=1.4)
    ax.set_xlabel("Observed net energy yield (MWh t$^{-1}$)")
    ax.set_ylabel("Predicted net energy yield (MWh t$^{-1}$)")
    ax.set_title(f"Figure 1. Observed and predicted energy yield using {best_energy}")
    fig.tight_layout()
    fig.savefig(SECTION_51 / "figure_1_observed_predicted_energy.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    y_obs = predictions["observed_emissions"]
    y_pred = predictions[f"net_emissions_kgco2e_t::{best_em}"]
    ax.scatter(y_obs, y_pred, alpha=0.55, s=22)
    lims = [min(y_obs.min(), y_pred.min()), max(y_obs.max(), y_pred.max())]
    ax.plot(lims, lims, linewidth=1.4)
    ax.set_xlabel("Observed net emissions (kg CO$_2$e t$^{-1}$)")
    ax.set_ylabel("Predicted net emissions (kg CO$_2$e t$^{-1}$)")
    ax.set_title(f"Figure 2. Observed and predicted net emissions using {best_em}")
    fig.tight_layout()
    fig.savefig(SECTION_51 / "figure_2_observed_predicted_emissions.png", dpi=300)
    plt.close(fig)

    # ------------------------- Section 5.2 -------------------------
    # A diversified representative set is more informative for manuscript interpretation
    # than simply listing the global top-ranked candidates, because Pareto dominance can
    # concentrate many entries in one technologically dominant route.
    route_quotas = {
        "hybrid_biorefinery": 6,
        "anaerobic_digestion": 6,
        "pyrolysis": 3,
        "gasification": 3,
        "incineration_energy_recovery": 2,
    }
    representative_blocks = []
    for route, quota in route_quotas.items():
        block = pareto[pareto["conversion_route"] == route].head(quota)
        if not block.empty:
            representative_blocks.append(block)
    pareto_representative = pd.concat(representative_blocks, axis=0) if representative_blocks else pareto.head(18)
    pareto_top = pareto_representative[
        [
            "feedstock_type",
            "conversion_route",
            "net_energy_mwh_t",
            "net_emissions_kgco2e_t",
            "operating_cost_usd_t",
            "exposure_index",
            "reliability_index",
            "ai_circularity_score",
            "composite_governance_score",
        ]
    ].copy()
    pareto_top.to_csv(SECTION_52 / "table_2_pareto_optimal_scenarios.csv", index=False)
    save_table_png(
        pareto_top.round(3),
        SECTION_52 / "table_2_pareto_optimal_scenarios.png",
        "Table 2. Representative Pareto-optimal waste-to-energy configurations",
        max_rows=9,
    )

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    sample = candidates.sample(n=min(1000, len(candidates)), random_state=RANDOM_SEED)
    ax.scatter(sample["net_energy_mwh_t"], sample["net_emissions_kgco2e_t"], alpha=0.20, s=16, label="Candidate scenarios")
    ax.scatter(pareto["net_energy_mwh_t"], pareto["net_emissions_kgco2e_t"], alpha=0.85, s=34, label="Pareto-efficient scenarios")
    ax.set_xlabel("Net energy yield (MWh t$^{-1}$)")
    ax.set_ylabel("Net emissions (kg CO$_2$e t$^{-1}$)")
    ax.set_title("Figure 3. Pareto frontier for energy recovery and emissions governance")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(SECTION_52 / "figure_3_pareto_front_energy_emissions.png", dpi=300)
    plt.close(fig)

    route_scores = summarize_by_route(pareto)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.bar(route_scores["conversion_route"], route_scores["mean_circularity_score"], label="Circularity score")
    ax.plot(route_scores["conversion_route"], 100 * route_scores["mean_reliability_index"], marker="o", label="Reliability index × 100")
    ax.set_ylabel("Score")
    ax.set_title("Figure 4. Circularity-reliability trade-off across Pareto-efficient routes")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(SECTION_52 / "figure_4_route_tradeoff_circularity_reliability.png", dpi=300)
    plt.close(fig)

    # ------------------------- Section 5.3 -------------------------
    governance_summary = summarize_by_route(data)
    governance_summary.to_csv(SECTION_53 / "table_3_governance_indicator_summary.csv", index=False)
    save_table_png(
        governance_summary.round(3),
        SECTION_53 / "table_3_governance_indicator_summary.png",
        "Table 3. Governance indicators aggregated by conversion pathway",
        max_rows=8,
    )

    shap_importance.to_csv(SECTION_53 / "shap_feature_importance.csv", index=False)
    feature_labels = {
        "lower_heating_value_mj_kg": "Lower heating value",
        "hydraulic_retention_time_d": "Hydraulic retention time",
        "conversion_route_hybrid_biorefinery": "Hybrid biorefinery route",
        "moisture_pct": "Moisture content",
        "ph": "pH",
        "contamination_pct": "Contamination level",
        "temperature_c": "Process temperature",
        "organic_fraction_pct": "Organic fraction",
        "feedstock_type_agricultural_residues": "Agricultural-residue feedstock",
        "feedstock_type_sewage_sludge": "Sewage-sludge feedstock",
        "ash_pct": "Ash content",
        "local_energy_demand_index": "Local energy demand",
    }
    top_shap = shap_importance.head(12).copy()
    top_shap["feature_label"] = top_shap["feature"].map(feature_labels).fillna(
        top_shap["feature"].str.replace("_", " ").str.title()
    )
    top_shap = top_shap.sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    ax.barh(top_shap["feature_label"], top_shap["mean_abs_shap"])
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_ylabel("Predictor")
    ax.set_title("Feature importance for net-energy prediction")
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(SECTION_53 / "figure_5_shap_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    components = governance_summary.copy()
    components["energy_component"] = 100 * normalize_series(components["mean_energy_mwh_t"])
    components["carbon_component"] = 100 * (1 - normalize_series(components["mean_net_emissions_kgco2e_t"]))
    components["material_component"] = 100 * components["mean_material_recovery_index"]
    components["justice_component"] = 100 * components["mean_environmental_justice_index"]
    components["reliability_component"] = 100 * components["mean_reliability_index"]
    component_cols = ["energy_component", "carbon_component", "material_component", "justice_component", "reliability_component"]
    component_table = components[["conversion_route"] + component_cols]
    component_table.to_csv(SECTION_53 / "circularity_score_components.csv", index=False)

    route_labels = {
        "hybrid_biorefinery": "Hybrid\nbiorefinery",
        "anaerobic_digestion": "Anaerobic\ndigestion",
        "pyrolysis": "Pyrolysis",
        "gasification": "Gasification",
        "incineration_energy_recovery": "Incineration\nwith energy\nrecovery",
    }
    fig, ax = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
    x = np.arange(len(components))
    bottom = np.zeros(len(components))
    for col in component_cols:
        values = components[col].to_numpy()
        ax.bar(x, values, bottom=bottom, label=col.replace("_component", "").replace("_", " ").title())
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels(components["conversion_route"].map(route_labels).fillna(components["conversion_route"]))
    ax.set_ylabel("Component contribution to composite score")
    ax.set_title("AI circularity score decomposition")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(SECTION_53 / "figure_6_circularity_score_decomposition.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_readme(data: pd.DataFrame, pareto: pd.DataFrame) -> None:
    """Create a concise reproducibility note for the results archive."""
    readme = f"""
    AI-Optimized Waste-to-Energy Computational Results
    ==================================================

    This archive contains the tables and figures generated by ai_optimized_wte_framework.py.
    The computational experiment uses a synthetic but mechanistically structured data environment
    with {len(data):,} baseline scenarios and {len(pareto):,} Pareto-efficient candidate solutions.

    Folder mapping:
    - 5.1: Predictive intelligence and yield-emission modeling.
    - 5.2: Multi-objective optimization and Pareto governance.
    - 5.3: AI circularity score, explainability, and decision accountability.

    To reproduce the complete output, run:
        python ai_optimized_wte_framework.py

    Dependency installation command if needed:
        python -m pip install numpy pandas matplotlib scikit-learn xgboost lightgbm shap scipy
    """
    (RESULTS_DIR / "README.txt").write_text(textwrap.dedent(readme).strip() + "\n", encoding="utf-8")


def main() -> None:
    print("Generating mechanistically informed waste-to-energy dataset...")
    data = simulate_wte_dataset()
    data.to_csv(RESULTS_DIR / "synthetic_wte_dataset.csv", index=False)

    print("Training predictive models for energy yield and net emissions...")
    metrics_df, fitted, best_summary, predictions = fit_predictive_models(data)

    print("Running multi-objective optimization and Pareto screening...")
    candidates, pareto = perform_optimization(data)
    candidates.to_csv(SECTION_52 / "candidate_scenario_population.csv", index=False)
    pareto.to_csv(SECTION_52 / "pareto_efficient_scenarios_full.csv", index=False)

    best_energy_model_name = best_summary.loc[best_summary["target"] == "net_energy_mwh_t", "best_model"].iloc[0]
    best_energy_pipeline = fitted[f"net_energy_mwh_t::{best_energy_model_name}"]

    print("Computing SHAP explanations for the best energy-yield model...")
    shap_importance = compute_shap_importance(best_energy_pipeline, data)

    print("Writing article-ready tables and figures...")
    create_figures_and_tables(data, metrics_df, best_summary, predictions, fitted, candidates, pareto, shap_importance)
    write_readme(data, pareto)

    print("Done. Results have been written to:")
    print(f"  {RESULTS_DIR}")


if __name__ == "__main__":
    main()
