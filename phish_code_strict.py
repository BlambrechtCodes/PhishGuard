"""
Phishing detector strict-mode integration helpers.

- Loads model artifacts if available (model.pkl, scaler.pkl, training_columns.pkl, top_features.pkl)
- Provides strict_predict_one for applying strict policy on a single-row dataframe
- If a dataset CSV is present, you can run a quick test from CLI

Strict defaults (from strict_policy):
    legit_hi = 0.75
    legit_lo = 0.55
    rule_penalty = 0.15
"""

import os
import pickle
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from strict_policy import (
    strict_hard_rules,
    strict_mode_decision,
    DEFAULT_LEGIT_HI,
    DEFAULT_LEGIT_LO,
    DEFAULT_RULE_PENALTY,
)

ARTIFACTS_DIR = Path(os.getenv("PHISH_ARTIFACTS_DIR", "."))

def _load_pickle(path: Path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_artifacts(artifacts_dir: Path = ARTIFACTS_DIR) -> Dict[str, Any]:
    """Load model/scaler/columns/features if present. Missing items are omitted."""
    out = {}
    for name in ["model.pkl", "scaler.pkl", "training_columns.pkl", "top_features.pkl"]:
        p = artifacts_dir / name
        if p.exists():
            out[name] = _load_pickle(p)
    return out

def ensure_columns(df: pd.DataFrame, training_columns: list, filler: pd.Series) -> pd.DataFrame:
    for col in training_columns:
        if col not in df.columns:
            df[col] = filler.get(col, 0)
    return df[training_columns].copy()

def strict_predict_one(raw_row_df: pd.DataFrame,
                       artifacts: Dict[str, Any],
                       legit_hi: float = DEFAULT_LEGIT_HI,
                       legit_lo: float = DEFAULT_LEGIT_LO,
                       rule_penalty: float = DEFAULT_RULE_PENALTY) -> Dict[str, Any]:
    """Apply strict policy to a single-row feature dataframe."""
    assert len(raw_row_df) == 1, "raw_row_df must be a single row DataFrame"

    model = artifacts.get("model.pkl")
    scaler = artifacts.get("scaler.pkl")
    training_columns = artifacts.get("training_columns.pkl")
    top_features = artifacts.get("top_features.pkl")

    if model is None or scaler is None or training_columns is None or top_features is None:
        raise RuntimeError("Missing artifacts: need model.pkl, scaler.pkl, training_columns.pkl, top_features.pkl")

    filler = pd.Series({c: 0 for c in training_columns})
    row_full = ensure_columns(raw_row_df.copy(), training_columns, filler=filler)

    scaled = scaler.transform(row_full)
    sel = pd.DataFrame(scaled, columns=training_columns)[top_features]

    proba = model.predict_proba(sel)[0]
    p_legit = float(proba[1])

    rule_hits, rule_details = strict_hard_rules(row_full.iloc[0])

    decision, adjusted = strict_mode_decision(
        p_legit=p_legit,
        rule_hits=rule_hits,
        legit_hi=legit_hi,
        legit_lo=legit_lo,
        rule_penalty=rule_penalty
    )

    return {
        "base_prediction": "LEGIT" if p_legit >= 0.5 else "PHISH",
        "p_legit": round(p_legit, 6),
        "strict_decision": decision,
        "adjusted_p_legit": round(adjusted, 6),
        "rule_hits": int(rule_hits),
        "rule_details": rule_details,
        "probs": {"phish": round(float(proba[0]), 6), "legit": round(float(proba[1]), 6)},
        "thresholds": {"legit_hi": legit_hi, "legit_lo": legit_lo, "rule_penalty": rule_penalty}
    }

if __name__ == "__main__":
    # Optional quick demo if the dataset is present.
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    csv_path = Path("PhiUSIIL_Phishing_URL_Dataset.csv")
    if not csv_path.exists():
        print("No dataset found at", csv_path, "- module ready for import.")
    else:
        df = pd.read_csv(csv_path)
        ycol = [c for c in df.columns if c.lower() in ("type","label","target","is_phish")][0]
        y = df[ycol].astype(int)
        X = df.select_dtypes(include=[np.number]).drop(columns=[ycol], errors="ignore").fillna(0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        rf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42, class_weight={0:2.0,1:1.0})
        rf.fit(X_train_s, y_train)

        p_legit = rf.predict_proba(X_test_s)[:,1]

        def get_col(name, default=0):
            return X_test[name] if name in X_test.columns else pd.Series(default, index=X_test.index)

        hits = (
            (get_col('URLLength') >= 100).astype(int) +
            (get_col('NoOfSubDomain') >= 3).astype(int) +
            (get_col('SpecialCharRatioInURL') >= 0.08).astype(int) +
            (get_col('DigitRatioInURL') >= 0.35).astype(int) +
            (get_col('NoOfAmpersandInURL') >= 2).astype(int) +
            (get_col('IsDomainIP').fillna(0).astype(int) == 1).astype(int)
        ).to_numpy()

        legit_hi, legit_lo, penalty = DEFAULT_LEGIT_HI, DEFAULT_LEGIT_LO, DEFAULT_RULE_PENALTY
        adjusted = np.maximum(0.0, p_legit - hits * penalty)
        y_pred = np.where(adjusted >= legit_hi, 1, np.where(adjusted < legit_lo, 0, 0))

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        print("Strict policy metrics:")
        print(f"  accuracy : {acc:.6f}")
        print(f"  precision: {prec:.6f}")
        print(f"  recall   : {rec:.6f}")
        print(f"  f1-score : {f1:.6f}")
        print("  confusion_matrix [rows=true, cols=pred] (Phish=0, Legit=1):\n", cm)
