import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, List

class TabularPreprocessor:
    """
    - Escala numéricas con StandardScaler.
    - Categóricas con índices enteros, reservando 0 para UNK (desconocido).
      Mapeo: {UNK: 0, cat1:1, cat2:2, ...}.
    """
    def __init__(self, numeric: List[str], categorical: List[str]):
        self.numeric_base = numeric
        self.numeric = list(numeric)
        self.categorical = categorical
        self.scaler = StandardScaler()
        self.cat_maps: Dict[str, Dict[str, int]] = {}
        self.engineered_features: List[str] = []

    def _engineer_features(self, df: pd.DataFrame, update_numeric: bool = False) -> pd.DataFrame:
        df = df.copy()
        if "90 s" in df.columns:
            denom = df["90 s"].replace(0, np.nan)
            for col in ["Gls", "Ass", "PrgC"]:
                if col in df.columns:
                    df[col] = (df[col] / denom).fillna(0.0)

        engineered = []
        if "Gls" in df.columns and "xG" in df.columns:
            ratio_col = "Gls_xG_ratio"
            df[ratio_col] = (df["Gls"] / df["xG"].replace(0, np.nan)).fillna(0.0)
            engineered.append(ratio_col)
        if "PrgP" in df.columns and "PrgC" in df.columns:
            ratio_col = "PrgP_PrgC_ratio"
            df[ratio_col] = (df["PrgP"] / df["PrgC"].replace(0, np.nan)).fillna(0.0)
            engineered.append(ratio_col)

        if update_numeric:
            self.engineered_features = engineered
            self.numeric = list(self.numeric_base) + engineered
        return df

    def fit(self, df: pd.DataFrame) -> None:
        # Ajusta scaler SOLO con train (hazlo por fold en train.py).
        df_eng = self._engineer_features(df, update_numeric=True)
        self.scaler.fit(df_eng[self.numeric])

        # Construye mapas con UNK=0 y categorías a partir de 1
        for col in self.categorical:
            uniques = sorted(df[col].dropna().astype(str).unique())
            mapping = {"__UNK__": 0}
            for i, v in enumerate(uniques, start=1):
                mapping[v] = i
            self.cat_maps[col] = mapping

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        df_eng = self._engineer_features(df, update_numeric=False)
        # Aplica scaler
        df_num = pd.DataFrame(self.scaler.transform(df_eng[self.numeric]),
                              columns=self.numeric, index=df.index)

        # Indexa categóricas con UNK=0
        for col in self.categorical:
            mapping = self.cat_maps[col]
            df_num[col + "__idx"] = df[col].astype(str).map(mapping).fillna(0).astype(int)

        # Cardinalidades para embeddings (incluye UNK)
        emb_cards = {k: max(mapping.values()) + 1 for k, mapping in self.cat_maps.items()}
        return df_num, emb_cards

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._engineer_features(df, update_numeric=False)

    def save(self, path_scaler: str, path_cats: str) -> None:
        import pickle
        with open(path_scaler, 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(path_cats, 'w') as f:
            json.dump(self.cat_maps, f, indent=2)
