import numpy as np


def get_crop(
    margin: int,
    image: np.ndarray,
    mask: np.ndarray,
    hu_min: int
) -> tuple[np.ndarray, np.ndarray]:

    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        raise ValueError("No voxels with relevant mapping found")

    max_y = max(ys)
    min_y = min(ys)
    max_x = max(xs)
    min_x = min(xs)

    size_y = max_y - min_y + 1
    size_x = max_x - min_x + 1

    size = max(size_x, size_y)
    size += 2 * margin

    center = (
        (max_y + min_y) / 2,
        (max_x + min_x) / 2
    )

    y1 = int(round(center[0] - size / 2))
    y2 = y1 + size
    x1 = int(round(center[1] - size / 2))
    x2 = x1 + size

    h, w = image.shape
    pad_top = max(0, -y1)
    pad_bot = max(0, y2 - h)
    pad_left = max(0, -x1)
    pad_right = max(0, w - x2)

    if pad_top or pad_bot or pad_left or pad_right:
        image = np.pad(
            image,
            (
                (pad_top, pad_bot),
                (pad_left, pad_right),
            ),
            mode="constant",
            constant_values=hu_min,
        )

        mask = np.pad(
            mask,
            (
                (pad_top, pad_bot),
                (pad_left, pad_right),
            ),
            mode="constant",
            constant_values=0,
        )

        y1 += pad_top
        y2 += pad_top
        x1 += pad_left
        x2 += pad_left

    image = image[y1:y2, x1:x2]
    mask = mask[y1:y2, x1:x2]

    return image, mask
