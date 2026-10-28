"""
Assign new bank customers to a segment using the saved K-Means pipeline.

Usage:
    from segment_assigner import assign_segment
    result = assign_segment(new_customers_df, "segmentation_kmeans_v1.joblib")
Input columns (raw format, same as the source file):
    client_age, customer_since, bank_account, credit_card, mortgage,
    total_number_of_products, customer_income, customer_debt,
    use_of_digital_channels, use_of_branches
"""
import joblib
import numpy as np
import pandas as pd

BINARY_MAP = {"no": 0, "yes": 1}
ORDINAL_MAP = {"low": 0, "medium": 1, "high": 2}


def build_features(raw: pd.DataFrame, reference_year: int) -> pd.DataFrame:
    """Replicate the cleaning and feature engineering of Modules 2 and 4."""
    d = raw.copy()
    for c in ["bank_account", "credit_card", "mortgage", "use_of_digital_channels", "use_of_branches"]:
        d[c] = d[c].astype(str).str.strip().str.lower()
    for c in ["bank_account", "credit_card", "mortgage"]:
        d[f"{c}_bin"] = d[c].map(BINARY_MAP)
    d["digital_use_ord"] = d["use_of_digital_channels"].map(ORDINAL_MAP)
    d["branch_use_ord"] = d["use_of_branches"].map(ORDINAL_MAP)
    d["n_core_products"] = d[["bank_account_bin", "credit_card_bin", "mortgage_bin"]].sum(axis=1)
    # Rule 2.6b: total products can never be lower than the core products held
    d["total_number_of_products"] = np.maximum(d["total_number_of_products"], d["n_core_products"])
    d["extra_products"] = d["total_number_of_products"] - d["n_core_products"]
    d["tenure_years"] = reference_year - d["customer_since"]
    d["log_debt_to_income"] = np.log1p(d["customer_debt"] / d["customer_income"])
    return d


def assign_segment(raw: pd.DataFrame, model_path: str) -> pd.DataFrame:
    """Return the input with segment id, segment name and recommended offer."""
    artifact = joblib.load(model_path)
    d = build_features(raw, artifact["reference_year"])
    missing = d[artifact["features"]].isna().any(axis=1)
    if missing.any():
        raise ValueError(f"{missing.sum()} rows have missing or unmapped values")
    ids = artifact["pipeline"].predict(d[artifact["features"]].astype(float))
    out = raw.copy()
    out["segment_id"] = ids
    out["segment"] = pd.Series(ids, index=raw.index).map(artifact["segment_names"])
    out["recommended_offer"] = out["segment"].map(artifact["offers"])
    return out
