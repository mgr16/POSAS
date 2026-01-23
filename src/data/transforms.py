import numpy as np
import cv2

class HeatmapTransforms:
    def __init__(
        self,
        size: int = 64,
        normalize: bool = True,
        shift_pixels: int = 0,
        gaussian_blur_sigma: float = 0.0,
        random_affine_degrees: float = 0.0,
        coarse_dropout_prob: float = 0.0,
        mean: float = None,
        std: float = None,
    ):
        self.size = size
        self.normalize = normalize
        self.shift_pixels = shift_pixels
        self.sigma = gaussian_blur_sigma
        self.random_affine_degrees = random_affine_degrees
        self.coarse_dropout_prob = coarse_dropout_prob
        self.mean = mean
        self.std = std

    def _apply_random_affine(self, hm: np.ndarray) -> np.ndarray:
        if not self.random_affine_degrees and not self.shift_pixels:
            return hm
        angle = np.random.uniform(-self.random_affine_degrees, self.random_affine_degrees) if self.random_affine_degrees else 0.0
        tx = np.random.randint(-self.shift_pixels, self.shift_pixels + 1) if self.shift_pixels else 0
        ty = np.random.randint(-self.shift_pixels, self.shift_pixels + 1) if self.shift_pixels else 0
        center = (self.size / 2, self.size / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        M[0, 2] += tx
        M[1, 2] += ty
        return cv2.warpAffine(hm, M, (self.size, self.size), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)

    def _apply_coarse_dropout(self, hm: np.ndarray) -> np.ndarray:
        if self.coarse_dropout_prob <= 0 or np.random.rand() > self.coarse_dropout_prob:
            return hm
        h, w = hm.shape[:2]
        drop_h = np.random.randint(max(1, int(0.1 * h)), max(2, int(0.3 * h)))
        drop_w = np.random.randint(max(1, int(0.1 * w)), max(2, int(0.3 * w)))
        y1 = np.random.randint(0, h - drop_h + 1)
        x1 = np.random.randint(0, w - drop_w + 1)
        hm = hm.copy()
        hm[y1:y1 + drop_h, x1:x1 + drop_w] = 0
        return hm

    def __call__(self, img: np.ndarray) -> np.ndarray:
        # img: 2D (H,W) float32 o uint8
        hm = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        hm = self._apply_random_affine(hm)
        hm = self._apply_coarse_dropout(hm)
        if self.sigma and self.sigma > 0:
            k = max(1, int(2 * round(3 * self.sigma) + 1))
            hm = cv2.GaussianBlur(hm, (k, k), self.sigma)
        hm = hm.astype('float32')
        if self.normalize:
            if self.mean is not None and self.std is not None:
                hm = (hm - float(self.mean)) / (float(self.std) + 1e-6)
            else:
                m, s = hm.mean(), hm.std() + 1e-6
                hm = (hm - m) / s
        return hm
