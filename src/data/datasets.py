from typing import Dict, Any, Optional
import os, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset
from .transforms import HeatmapTransforms

class MultimodalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg_image: Dict[str, Any], heatmaps_dir: str, features_num: list, cat_idx_cols: list, target_col: Optional[str]):
        self.df = df.reset_index(drop=True)
        self.hm_dir = heatmaps_dir
        self.features_num = features_num
        self.cat_idx_cols = cat_idx_cols
        self.target_col = target_col
        self.tfm = HeatmapTransforms(
            size=cfg_image['size'],
            normalize=cfg_image.get('normalize', True),
            shift_pixels=cfg_image.get('augment', {}).get('shift_pixels', 0),
            gaussian_blur_sigma=cfg_image.get('augment', {}).get('gaussian_blur_sigma', 0.0),
            random_affine_degrees=cfg_image.get('augment', {}).get('random_affine_degrees', 0.0),
            coarse_dropout_prob=cfg_image.get('augment', {}).get('coarse_dropout_prob', 0.0),
            mean=cfg_image.get('mean', None),
            std=cfg_image.get('std', None),
        )

    def _load_heatmap(self, fname: str) -> np.ndarray:
        p_npy = os.path.join(self.hm_dir, fname.replace('.jpg', '.npy'))
        p_jpg = os.path.join(self.hm_dir, fname)
        if os.path.exists(p_npy):
            arr = np.load(p_npy, mmap_mode='r')
            return np.array(arr)
        elif os.path.exists(p_jpg):
            import cv2
            img = cv2.imread(p_jpg, cv2.IMREAD_GRAYSCALE)
            return img
        else:
            raise FileNotFoundError(f"Heatmap no encontrado: {fname}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        hm = self._load_heatmap(row['heatmap_filename'])
        hm = self.tfm(hm)[None, ...]  # (1, H, W)
        x_img = torch.from_numpy(hm)
        x_tab = torch.tensor(row[self.features_num].values.astype('float32'))
        # Construir dict de categorías a partir de columnas *_idx
        cats_dict = {c.replace('__idx', ''): int(row[c]) for c in self.cat_idx_cols}
        # Target opcional (para inferencia)
        if self.target_col is not None and self.target_col in self.df.columns:
            y = torch.tensor(float(row[self.target_col]))
        else:
            y = torch.tensor(0.0)
        return {"img": x_img, "tab": x_tab, "cats": cats_dict, "y": y}
