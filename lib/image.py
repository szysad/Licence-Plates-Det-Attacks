import requests
from PIL import Image
import torchvision.transforms.functional as TF
import torch
import torchvision
import numpy as np


def download_img(url: str, path: str) -> None:
    with open(path, "wb") as handle:
        response = requests.get(url, stream=True)
        for block in response.iter_content(1024):
            if not block:
                break

            handle.write(block)


def load_image_torch(path: str) -> np.ndarray:
    img = Image.open(path)
    img = TF.to_tensor(img).unsqueeze(0)
    return img


def preprocess_img(img: torch.Tensor, hw: int = 640, half: bool = True) -> torch.Tensor:
    img = torchvision.transforms.Resize((hw, hw))(img)
    if half is True:
        img = img.half()
    return img


def img_to_numpy(img: torch.Tensor) -> np.ndarray:
    return img.permute(1, 2, 0).numpy()


def img_float_to_uint(img: np.ndarray):
    return (255 * img).astype(np.uint8)
