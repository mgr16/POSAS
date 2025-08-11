import pandas as pd
import numpy as np

def variance_threshold(df: pd.DataFrame, thr: float = 1e-8):
    keep = [c for c in df.columns if df[c].var() > thr]
    return df[keep], keep

def correlation_prune(df: pd.DataFrame, corr_thr: float = 0.9):
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = [column for column in upper.columns if any(upper[column] > corr_thr)]
    keep = [c for c in df.columns if c not in drop]
    return df[keep], keep
