import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, List

class TabularPreprocessor:
    """
    - Escala numéricas con StandardScaler.
    - Categóricas con índices enteros, reservando 0 para UNK (desconocido).
      Mapeo: {UNK: 0, cat1:1, cat2:2, ...}.
    """
    def __init__(self, numeric: List[str], categorical: List[str]):
        self.numeric = numeric
        self.categorical = categorical
        self.scaler = StandardScaler()
        self.cat_maps: Dict[str, Dict[str, int]] = {}

    def fit(self, df: pd.DataFrame) -> None:
        # Ajusta scaler SOLO con train (hazlo por fold en train.py).
        self.scaler.fit(df[self.numeric])

        # Construye mapas con UNK=0 y categorías a partir de 1
        for col in self.categorical:
            uniques = sorted(df[col].dropna().astype(str).unique())
            mapping = {"__UNK__": 0}
            for i, v in enumerate(uniques, start=1):
                mapping[v] = i
            self.cat_maps[col] = mapping

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        # Aplica scaler
        df_num = pd.DataFrame(self.scaler.transform(df[self.numeric]),
                              columns=self.numeric, index=df.index)

        # Indexa categóricas con UNK=0
        for col in self.categorical:
            mapping = self.cat_maps[col]
            df_num[col + "__idx"] = df[col].astype(str).map(mapping).fillna(0).astype(int)

        # Cardinalidades para embeddings (incluye UNK)
        emb_cards = {k: max(mapping.values()) + 1 for k, mapping in self.cat_maps.items()}
        return df_num, emb_cards

    def save(self, path_scaler: str, path_cats: str) -> None:
        import pickle
        with open(path_scaler, 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(path_cats, 'w') as f:
            json.dump(self.cat_maps, f, indent=2)
