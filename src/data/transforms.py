import numpy as np
import cv2

class HeatmapTransforms:
    def __init__(self, size: int = 64, normalize: bool = True, shift_pixels: int = 0, gaussian_blur_sigma: float = 0.0, mean: float = None, std: float = None):
        self.size = size
        self.normalize = normalize
        self.shift_pixels = shift_pixels
        self.sigma = gaussian_blur_sigma
        self.mean = mean
        self.std = std

    def __call__(self, img: np.ndarray) -> np.ndarray:
        # img: 2D (H,W) float32 o uint8
        hm = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        if self.shift_pixels:
            M = np.float32([[1, 0, np.random.randint(-self.shift_pixels, self.shift_pixels + 1)],
                            [0, 1, np.random.randint(-self.shift_pixels, self.shift_pixels + 1)]])
            hm = cv2.warpAffine(hm, M, (self.size, self.size), flags=cv2.INTER_NEAREST)
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
