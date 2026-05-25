import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

SEED = 42
TARGET_COL = "target"
ID_COL = "ID"


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def sanitize_column(name: str) -> str:
    clean = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower())
    return re.sub(r"_+", "_", clean).strip("_")


def load_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(os.path.join(data_dir, "Train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "Test.csv"))
    fred = pd.read_csv(os.path.join(data_dir, "economic_indicators.csv"))
    return train, test, fred


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["disbursement_date"] = pd.to_datetime(out["disbursement_date"], errors="coerce")
    out["due_date"] = pd.to_datetime(out["due_date"], errors="coerce")

    numeric_cols = [
        "Total_Amount",
        "Total_Amount_to_Repay",
        "duration",
        "Amount_Funded_By_Lender",
        "Lender_portion_Funded",
        "Lender_portion_to_be_repaid",
    ]
    for c in numeric_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["loan_year"] = out["disbursement_date"].dt.year
    out["loan_month"] = out["disbursement_date"].dt.month
    out["loan_weekday"] = out["disbursement_date"].dt.weekday
    out["days_to_due"] = (out["due_date"] - out["disbursement_date"]).dt.days
    out["days_to_due"] = out["days_to_due"].fillna(out["duration"])

    amount = out["Total_Amount"].replace(0, np.nan)
    out["repay_ratio"] = out["Total_Amount_to_Repay"] / amount
    out["interest_rate_proxy"] = (out["Total_Amount_to_Repay"] - out["Total_Amount"]) / amount
    out["amount_per_day"] = out["Total_Amount"] / out["duration"].replace(0, np.nan)
    out["interest_x_duration"] = out["interest_rate_proxy"] * out["duration"]
    out["lender_share_interest"] = out["Lender_portion_Funded"] * out["interest_rate_proxy"]

    out["loan_key"] = out["customer_id"].astype(str) + "_" + out["tbl_loan_id"].astype(str)
    return out


def transform_fred(fred: pd.DataFrame) -> pd.DataFrame:
    f = fred.copy()
    value_cols = [c for c in f.columns if c.startswith("YR")]
    long = f.melt(id_vars=["Country", "Indicator"], value_vars=value_cols, var_name="year", value_name="value")
    long["year"] = long["year"].str.replace("YR", "", regex=False).astype(int)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    wide = long.pivot_table(index=["Country", "year"], columns="Indicator", values="value", aggfunc="mean").reset_index()
    rename_map = {c: f"fred_{sanitize_column(c)}" for c in wide.columns if c not in {"Country", "year"}}
    wide = wide.rename(columns=rename_map)
    return wide


def add_fred_features(df: pd.DataFrame, fred_wide: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["loan_year"] = out["loan_year"].fillna(-1).astype(int)
    merged = out.merge(
        fred_wide,
        how="left",
        left_on=["country_id", "loan_year"],
        right_on=["Country", "year"],
    )
    merged = merged.drop(columns=["Country", "year"], errors="ignore")
    return merged


def aggregate_to_loan_level(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    work = df.copy()
    group_cols = ["loan_key"]

    agg_map = {
        "ID": "first",
        "customer_id": "first",
        "country_id": "first",
        "tbl_loan_id": "first",
        "loan_type": "first",
        "New_versus_Repeat": "first",
        "disbursement_date": "first",
        "due_date": "first",
        "duration": "first",
        "loan_year": "first",
        "loan_month": "first",
        "loan_weekday": "first",
        "days_to_due": "first",
        "Total_Amount": "first",
        "Total_Amount_to_Repay": "first",
        "repay_ratio": "first",
        "interest_rate_proxy": "first",
        "amount_per_day": "first",
        "interest_x_duration": "first",
    }

    numeric_agg_cols = [
        "Amount_Funded_By_Lender",
        "Lender_portion_Funded",
        "Lender_portion_to_be_repaid",
        "lender_share_interest",
    ]
    for col in numeric_agg_cols:
        agg_map[col] = ["sum", "mean", "max", "min", "std"]

    loan = work.groupby(group_cols).agg(agg_map)
    loan.columns = [
        f"{c[0]}_{c[1]}" if isinstance(c, tuple) and c[1] else c[0] if isinstance(c, tuple) else c
        for c in loan.columns.to_flat_index()
    ]
    loan = loan.reset_index()

    rename_simple = {
        "ID_first": "ID",
        "customer_id_first": "customer_id",
        "country_id_first": "country_id",
        "tbl_loan_id_first": "tbl_loan_id",
        "loan_type_first": "loan_type",
        "New_versus_Repeat_first": "New_versus_Repeat",
        "disbursement_date_first": "disbursement_date",
        "due_date_first": "due_date",
        "duration_first": "duration",
        "loan_year_first": "loan_year",
        "loan_month_first": "loan_month",
        "loan_weekday_first": "loan_weekday",
        "days_to_due_first": "days_to_due",
        "Total_Amount_first": "Total_Amount",
        "Total_Amount_to_Repay_first": "Total_Amount_to_Repay",
        "repay_ratio_first": "repay_ratio",
        "interest_rate_proxy_first": "interest_rate_proxy",
        "amount_per_day_first": "amount_per_day",
        "interest_x_duration_first": "interest_x_duration",
    }
    loan = loan.rename(columns=rename_simple)

    lender_stats = work.groupby("loan_key").agg(
        lender_count=("lender_id", "count"), lender_unique_count=("lender_id", "nunique")
    )
    loan = loan.merge(lender_stats, on="loan_key", how="left")

    loan["funded_to_total_ratio"] = loan["Amount_Funded_By_Lender_sum"] / loan["Total_Amount"].replace(0, np.nan)
    loan["repay_portion_to_total_ratio"] = loan["Lender_portion_to_be_repaid_sum"] / loan[
        "Total_Amount_to_Repay"
    ].replace(0, np.nan)
    loan["lender_amount_concentration"] = loan["Amount_Funded_By_Lender_max"] / loan[
        "Amount_Funded_By_Lender_sum"
    ].replace(0, np.nan)

    if is_train:
        target_map = work.groupby("loan_key")[TARGET_COL].max()
        loan[TARGET_COL] = loan["loan_key"].map(target_map)

    return loan


def add_customer_history_features(train_loan: pd.DataFrame, test_loan: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = train_loan.copy().sort_values(["customer_id", "disbursement_date", "tbl_loan_id"]).reset_index(drop=True)
    test = test_loan.copy().sort_values(["customer_id", "disbursement_date", "tbl_loan_id"]).reset_index(drop=True)

    grp = train.groupby("customer_id", sort=False)
    train["cust_prev_loan_count"] = grp.cumcount()
    train["cust_prev_defaults"] = grp[TARGET_COL].cumsum().shift(1).fillna(0)
    train["cust_prev_default_rate"] = np.where(
        train["cust_prev_loan_count"] > 0,
        train["cust_prev_defaults"] / train["cust_prev_loan_count"],
        np.nan,
    )

    prev_amount_sum = grp["Total_Amount"].cumsum().shift(1)
    train["cust_prev_amount_mean"] = np.where(
        train["cust_prev_loan_count"] > 0,
        prev_amount_sum / train["cust_prev_loan_count"],
        np.nan,
    )
    train["cust_prev_amount_max"] = grp["Total_Amount"].shift(1).groupby(train["customer_id"]).cummax()
    train["cust_last_loan_default"] = grp[TARGET_COL].shift(1)
    train["cust_days_since_last_loan"] = (
        train["disbursement_date"] - grp["disbursement_date"].shift(1)
    ).dt.days

    cust_stats = train.groupby("customer_id").agg(
        cust_train_loan_count=(TARGET_COL, "count"),
        cust_train_default_count=(TARGET_COL, "sum"),
        cust_train_default_rate=(TARGET_COL, "mean"),
        cust_train_amount_mean=("Total_Amount", "mean"),
        cust_train_amount_max=("Total_Amount", "max"),
        cust_last_train_disb=("disbursement_date", "max"),
        cust_last_train_default=(TARGET_COL, "last"),
    )

    test = test.merge(cust_stats, on="customer_id", how="left")
    test["cust_prev_loan_count"] = test["cust_train_loan_count"].fillna(0)
    test["cust_prev_defaults"] = test["cust_train_default_count"].fillna(0)
    test["cust_prev_default_rate"] = test["cust_train_default_rate"]
    test["cust_prev_amount_mean"] = test["cust_train_amount_mean"]
    test["cust_prev_amount_max"] = test["cust_train_amount_max"]
    test["cust_last_loan_default"] = test["cust_last_train_default"]
    test["cust_days_since_last_loan"] = (test["disbursement_date"] - test["cust_last_train_disb"]).dt.days

    train["loan_to_cust_avg_amount_ratio"] = train["Total_Amount"] / train["cust_prev_amount_mean"].replace(0, np.nan)
    test["loan_to_cust_avg_amount_ratio"] = test["Total_Amount"] / test["cust_prev_amount_mean"].replace(0, np.nan)

    drop_cols = [
        "cust_train_loan_count",
        "cust_train_default_count",
        "cust_train_default_rate",
        "cust_train_amount_mean",
        "cust_train_amount_max",
        "cust_last_train_disb",
        "cust_last_train_default",
    ]
    test = test.drop(columns=drop_cols, errors="ignore")

    return train, test


def add_customer_lender_features(train_loan: pd.DataFrame, test_loan: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = train_loan.copy()
    test = test_loan.copy()

    lender_train_stats = train.groupby("customer_id").agg(
        cust_train_unique_loan_type=("loan_type", "nunique"),
        cust_train_unique_country=("country_id", "nunique"),
        cust_train_unique_loans=("tbl_loan_id", "nunique"),
    )
    train = train.merge(lender_train_stats, on="customer_id", how="left")
    test = test.merge(lender_train_stats, on="customer_id", how="left")

    return train, test


def add_target_encoding(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: List[str],
    target_col: str,
    seed: int,
    n_splits: int = 5,
    smoothing: float = 20.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = train_df.copy()
    test = test_df.copy()

    y = train[target_col].astype(float).values
    global_mean = float(np.mean(y))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for col in cols:
        tr_col = train[col].astype(str).fillna("__NA__")
        te_col = test[col].astype(str).fillna("__NA__")

        enc_train = np.zeros(len(train), dtype=float)

        for tr_idx, val_idx in skf.split(train, train[target_col]):
            fold_tr = pd.DataFrame({col: tr_col.iloc[tr_idx].values, target_col: y[tr_idx]})
            stats = fold_tr.groupby(col)[target_col].agg(["mean", "count"])
            smooth = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
            enc_train[val_idx] = tr_col.iloc[val_idx].map(smooth).fillna(global_mean).values

        full_stats = pd.DataFrame({col: tr_col.values, target_col: y}).groupby(col)[target_col].agg(["mean", "count"])
        full_smooth = (full_stats["mean"] * full_stats["count"] + global_mean * smoothing) / (
            full_stats["count"] + smoothing
        )

        train[f"{col}_te"] = enc_train
        test[f"{col}_te"] = te_col.map(full_smooth).fillna(global_mean).values

    return train, test


def label_encode(train_df: pd.DataFrame, test_df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = train_df.copy()
    test = test_df.copy()

    for col in cols:
        all_vals = pd.concat([train[col], test[col]], axis=0).astype(str).fillna("__NA__")
        uniques = pd.Series(all_vals.unique())
        mapping = {v: i for i, v in enumerate(uniques)}
        train[f"{col}_idx"] = train[col].astype(str).fillna("__NA__").map(mapping).astype(float)
        test[f"{col}_idx"] = test[col].astype(str).fillna("__NA__").map(mapping).astype(float)

    return train, test


def choose_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    best_thr = 0.5
    best_f1 = -1.0
    for thr in np.linspace(0.01, 0.99, 197):
        pred = (y_prob >= thr).astype(int)
        score = f1_score(y_true, pred)
        if score > best_f1:
            best_f1 = score
            best_thr = float(thr)
    return best_thr, best_f1


@dataclass
class CVResult:
    oof_pred: np.ndarray
    test_pred: np.ndarray
    model_names: List[str]


def train_and_blend(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    seed: int,
    n_splits: int,
    enable_xgb: bool,
    enable_cat: bool,
) -> CVResult:
    import lightgbm as lgb

    xgb_available = False
    cat_available = False

    if enable_xgb:
        try:
            from xgboost import XGBClassifier

            xgb_available = True
        except Exception:
            xgb_available = False

    if enable_cat:
        try:
            from catboost import CatBoostClassifier

            cat_available = True
        except Exception:
            cat_available = False

    y = train_df[target_col].astype(int).values
    X = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    X = X.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    med = X.median(numeric_only=True)
    X = X.fillna(med)
    X_test = X_test.fillna(med)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof = np.zeros(len(train_df), dtype=float)
    preds = np.zeros(len(test_df), dtype=float)

    model_names: List[str] = ["lightgbm"]
    if xgb_available:
        model_names.append("xgboost")
    if cat_available:
        model_names.append("catboost")

    n_models = float(len(model_names))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr = X.iloc[tr_idx]
        y_tr = y[tr_idx]
        X_va = X.iloc[va_idx]
        y_va = y[va_idx]

        pos = float(np.sum(y_tr == 1))
        neg = float(np.sum(y_tr == 0))
        scale_pos_weight = max(1.0, neg / max(1.0, pos))

        fold_oof = np.zeros(len(va_idx), dtype=float)
        fold_test = np.zeros(len(test_df), dtype=float)

        lgb_model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=1500,
            learning_rate=0.03,
            num_leaves=64,
            max_depth=-1,
            min_child_samples=80,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            scale_pos_weight=scale_pos_weight,
            random_state=seed + fold,
            n_jobs=-1,
        )
        lgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(stopping_rounds=120, verbose=False)],
        )
        fold_oof += lgb_model.predict_proba(X_va)[:, 1]
        fold_test += lgb_model.predict_proba(X_test)[:, 1]

        if xgb_available:
            xgb_model = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=1400,
                learning_rate=0.03,
                max_depth=6,
                min_child_weight=2,
                subsample=0.85,
                colsample_bytree=0.8,
                reg_lambda=3.0,
                gamma=0.0,
                scale_pos_weight=scale_pos_weight,
                random_state=seed + fold,
                n_jobs=-1,
                tree_method="hist",
            )
            xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            fold_oof += xgb_model.predict_proba(X_va)[:, 1]
            fold_test += xgb_model.predict_proba(X_test)[:, 1]

        if cat_available:
            cat_model = CatBoostClassifier(
                loss_function="Logloss",
                eval_metric="Logloss",
                iterations=1600,
                learning_rate=0.03,
                depth=8,
                l2_leaf_reg=4.0,
                random_seed=seed + fold,
                class_weights=[1.0, scale_pos_weight],
                verbose=False,
            )
            cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
            fold_oof += cat_model.predict_proba(X_va)[:, 1]
            fold_test += cat_model.predict_proba(X_test)[:, 1]

        fold_oof = fold_oof / n_models
        fold_test = fold_test / n_models

        oof[va_idx] = fold_oof
        preds += fold_test / n_splits

        fold_f1 = f1_score(y_va, (fold_oof >= 0.5).astype(int))
        print(f"Fold {fold}/{n_splits} | baseline F1@0.5={fold_f1:.5f} | pos_weight={scale_pos_weight:.2f}")

    return CVResult(oof_pred=oof, test_pred=preds, model_names=model_names)


def make_submission(
    test_rows: pd.DataFrame, test_loan: pd.DataFrame, loan_pred_binary: pd.Series, output_path: str, target_name: str
) -> pd.DataFrame:
    out = test_rows[[ID_COL, "loan_key"]].copy()
    out = out.merge(
        test_loan[["loan_key"]].assign(pred=loan_pred_binary.values),
        on="loan_key",
        how="left",
    )
    out["pred"] = out["pred"].fillna(0).astype(int)
    submission = out[[ID_COL, "pred"]].rename(columns={"pred": target_name})
    submission.to_csv(output_path, index=False)
    return submission


def build_datasets(train_raw: pd.DataFrame, test_raw: pd.DataFrame, fred_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_base = add_base_features(train_raw)
    test_base = add_base_features(test_raw)

    fred_wide = transform_fred(fred_raw)
    train_base = add_fred_features(train_base, fred_wide)
    test_base = add_fred_features(test_base, fred_wide)

    train_loan = aggregate_to_loan_level(train_base, is_train=True)
    test_loan = aggregate_to_loan_level(test_base, is_train=False)

    train_loan, test_loan = add_customer_history_features(train_loan, test_loan)
    train_loan, test_loan = add_customer_lender_features(train_loan, test_loan)

    te_cols = ["customer_id", "loan_type", "tbl_loan_id", "country_id", "New_versus_Repeat"]
    train_loan, test_loan = add_target_encoding(train_loan, test_loan, te_cols, TARGET_COL, seed=SEED)

    cat_cols = ["customer_id", "country_id", "tbl_loan_id", "loan_type", "New_versus_Repeat"]
    train_loan, test_loan = label_encode(train_loan, test_loan, cat_cols)

    return train_loan, test_loan


def get_feature_columns(train_loan: pd.DataFrame) -> List[str]:
    drop_cols = {
        TARGET_COL,
        ID_COL,
        "loan_key",
        "disbursement_date",
        "due_date",
    }
    feature_cols = [c for c in train_loan.columns if c not in drop_cols]
    numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train_loan[c])]
    return numeric_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train robust loan-default pipeline and create submission")
    parser.add_argument("--data-dir", type=str, default=".", help="Directory containing Train.csv/Test.csv/economic_indicators.csv")
    parser.add_argument("--output", type=str, default="submission_blend.csv", help="Submission output CSV path")
    parser.add_argument("--target-name", type=str, default="Target", help="Output target column name")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--disable-xgb", action="store_true", help="Disable XGBoost in blend")
    parser.add_argument("--disable-cat", action="store_true", help="Disable CatBoost in blend")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    train_raw, test_raw, fred_raw = load_data(args.data_dir)
    print(f"Loaded train={train_raw.shape}, test={test_raw.shape}, fred={fred_raw.shape}")

    train_loan, test_loan = build_datasets(train_raw, test_raw, fred_raw)
    print(f"Loan-level train={train_loan.shape}, test={test_loan.shape}")

    feature_cols = get_feature_columns(train_loan)
    print(f"Using {len(feature_cols)} numeric features")

    cv_result = train_and_blend(
        train_df=train_loan,
        test_df=test_loan,
        target_col=TARGET_COL,
        feature_cols=feature_cols,
        seed=args.seed,
        n_splits=args.n_splits,
        enable_xgb=not args.disable_xgb,
        enable_cat=not args.disable_cat,
    )

    best_thr, best_f1 = choose_threshold(train_loan[TARGET_COL].values, cv_result.oof_pred)
    print(f"Models used: {', '.join(cv_result.model_names)}")
    print(f"Best global threshold={best_thr:.3f} | OOF F1={best_f1:.6f}")

    country_thresholds: Dict[str, float] = {}
    if train_loan["country_id"].nunique() > 1:
        for country, g in train_loan.groupby("country_id"):
            thr, f1 = choose_threshold(g[TARGET_COL].values, cv_result.oof_pred[g.index.values])
            country_thresholds[country] = thr
            print(f"Country {country} threshold={thr:.3f} | OOF F1={f1:.6f}")

    if country_thresholds:
        test_binary = pd.Series(index=test_loan.index, dtype=int)
        for country, idx in test_loan.groupby("country_id").groups.items():
            thr = country_thresholds.get(country, best_thr)
            test_binary.loc[list(idx)] = (cv_result.test_pred[list(idx)] >= thr).astype(int)
        test_binary = test_binary.fillna((cv_result.test_pred >= best_thr).astype(int)).astype(int)
    else:
        test_binary = pd.Series((cv_result.test_pred >= best_thr).astype(int), index=test_loan.index)

    submission = make_submission(
        test_rows=add_base_features(test_raw),
        test_loan=test_loan,
        loan_pred_binary=test_binary,
        output_path=args.output,
        target_name=args.target_name,
    )
    print(f"Wrote submission to {args.output} with shape {submission.shape}")


if __name__ == "__main__":
    main()
