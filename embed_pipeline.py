"""
embed_pipeline.py — 浮水印系統：僅保留嵌入過程
"""
from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
from torchvision import transforms

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

# 僅保留嵌入階段所需的依賴模組
from Tong_class_pythonCodes import LSB2_Embed_process_test5 as embed_lsb
from Tong_class_pythonCodes import read_model_predict_bottleneck_test2 as predict_bottleneck
from Tong_class_pythonCodes import upset_ofKey_test2 as upset_ofKey

LATENT_DIM = 256
IMG_SIZE = 128
DEFAULT_SEED = 42

MODEL_CANDIDATES = [
    "Tong_autoencoder_bottleneck256_batch32",
    "Tong_autoencoder_fusion2",
    "Tong_autoencoder_256diffmodel",
    "Tong_autoencoder_256diffmodel32batch",
    "Tong_autoencoder_fusion381difmodel",
]

@dataclass
class CryptoBlockInfo:
    block_id: int
    row: int
    col: int
    arnold_iterations: int
    original_matrix: np.ndarray
    xor_matrix: np.ndarray
    arnold_matrix: np.ndarray
    patch_before: np.ndarray | None = None
    patch_after: np.ndarray | None = None

@dataclass
class EmbedResult:
    original: np.ndarray
    embedded: np.ndarray
    bottleneck: np.ndarray
    block_size: int
    n_keys: int
    random_keys: list[int]
    crypto_blocks: list[CryptoBlockInfo] = field(default_factory=list)
    diff_map: np.ndarray | None = None

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_gray_image_from_array_or_path(source) -> np.ndarray:
    if isinstance(source, np.ndarray):
        img = source.copy()
    else:
        data = np.fromfile(str(source), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("無法讀取影像")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img

def generate_unique_random_numbers(seed: int, count: int, start: int = 1, end: int = 100) -> list[int]:
    rng = random.Random(seed)
    pool = list(range(start, end + 1))
    rng.shuffle(pool)
    return pool[:count]

def find_model_paths(base_dir: str | None = None) -> tuple[str | None, str | None]:
    base = base_dir or _BASE_DIR
    model_dir = os.path.join(base, "VGG11_model_prefusion")
    for name in MODEL_CANDIDATES:
        pt = os.path.join(model_dir, f"{name}.pt")
        pth = os.path.join(model_dir, f"{name}.pth")
        if os.path.isfile(pt) or os.path.isfile(pth):
            return (pt if os.path.isfile(pt) else None, pth if os.path.isfile(pth) else None)
    return None, None

def load_model(device: torch.device, base_dir: str | None = None) -> torch.nn.Module:
    from model_arch import build_model
    pt_path, pth_path = find_model_paths(base_dir)
    
    if pth_path and os.path.isfile(pth_path):
        model = build_model(latent_dim=LATENT_DIM, img_size=IMG_SIZE)
        state = torch.load(pth_path, map_location=device, weights_only=False)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return model

    if pt_path and os.path.isfile(pt_path):
        model = torch.load(pt_path, map_location=device, weights_only=False)
        model.to(device)
        model.eval()
        return model

    raise FileNotFoundError("找不到 bn256 模型，請確認模型檔案路徑。")

def get_transform():
    return transforms.Compose([transforms.ToTensor(), transforms.Resize((IMG_SIZE, IMG_SIZE))])

def embed_watermark(
    image: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    *,
    seed: int = DEFAULT_SEED,
    capture_crypto: bool = True,
    demo_block_id: int = 0,
) -> EmbedResult:
    gray = load_gray_image_from_array_or_path(image)
    original = gray.copy()

    predictor = predict_bottleneck.read_model_predict_bottleneck()
    bottleneck_out = predictor.infer_single_image(gray, model, get_transform(), device)
    bottleneck = np.array(bottleneck_out[0], dtype=np.uint8)

    block_size = int(np.sqrt(len(bottleneck)) * 2)
    n_keys = int((gray.shape[0] // block_size) * (gray.shape[1] // block_size))
    random_keys = generate_unique_random_numbers(seed, count=n_keys)

    tong_ofkey = upset_ofKey.Disruption_operation()
    tong_embed = embed_lsb.imageAndBottleneck_ToWatermarkingImage()

    embedded = gray.copy()
    crypto_blocks: list[CryptoBlockInfo] = []
    block_id = 0

    for i in range(0, gray.shape[0], block_size):
        for j in range(0, gray.shape[1], block_size):
            patch = embedded[i : i + block_size, j : j + block_size].copy()
            orig_matrix = tong_ofkey.Help_change_traits(bottleneck.tolist())
            xor_list = tong_ofkey.apply_xor_binding(bottleneck.tolist(), block_id)
            xor_matrix = tong_ofkey.Help_change_traits(xor_list)
            arnold_list = tong_ofkey.arnold_transform(xor_list, random_keys[block_id])
            arnold_matrix = tong_ofkey.Help_change_traits(arnold_list)

            patch_embedded = tong_embed.Tong_calculate(patch, arnold_list)
            embedded[i : i + block_size, j : j + block_size] = patch_embedded

            if capture_crypto:
                crypto_blocks.append(
                    CryptoBlockInfo(
                        block_id=block_id,
                        row=i // block_size,
                        col=j // block_size,
                        arnold_iterations=random_keys[block_id],
                        original_matrix=orig_matrix.copy(),
                        xor_matrix=xor_matrix.copy(),
                        arnold_matrix=arnold_matrix.copy(),
                        patch_before=patch.copy(),
                        patch_after=patch_embedded.copy(),
                    )
                )
            block_id += 1

    diff_map = np.abs(embedded.astype(np.int16) - original.astype(np.int16)).astype(np.uint8)
    return EmbedResult(
        original=original,
        embedded=embedded,
        bottleneck=bottleneck,
        block_size=block_size,
        n_keys=n_keys,
        random_keys=random_keys,
        crypto_blocks=crypto_blocks,
        diff_map=diff_map,
    )