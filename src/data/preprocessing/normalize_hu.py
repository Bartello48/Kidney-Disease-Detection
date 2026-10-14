import numpy as np


def normalize_hu(
    image: np.ndarray,
    hu_min: int,
    hu_max: int
) -> np.ndarray:
    image = np.clip(image, hu_min, hu_max)
    image = (image - hu_min) / (hu_max - hu_min)
    return image.astype(np.float32)
