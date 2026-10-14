import numpy as np
import torch
import torch.nn.functional as funct


def resize(
    image: np.ndarray,
    mask: np.ndarray,
    size: int
) -> tuple[torch.Tensor, torch.Tensor]:

    image = torch.from_numpy(image)
    image = image.unsqueeze(0).unsqueeze(0)

    image = funct.interpolate(
        image,
        size=(size, size),
        mode='bilinear',
        align_corners=False
    )
    image = image.squeeze(0)

    mask = torch.from_numpy(mask)
    mask = mask.unsqueeze(0).unsqueeze(0).float()
    mask = funct.interpolate(
        mask,
        size=(size, size),
        mode='nearest'
    )
    mask = mask.squeeze(0).squeeze(0).to(torch.uint8)

    return image, mask
