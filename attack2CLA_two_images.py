"""
attack2CLA_two_images.py
雙圖拼貼攻擊：使用另一張合法的浮水印影像作為來源，
將其區塊依據竄改率覆蓋到目標影像的對應位置。
"""

from __future__ import annotations

import cv2
import numpy as np

BLOCK_SIZE = 16

def compute_tamper_blocks(total_blocks: int, ratio_percent: int) -> int:
    """依整數竄改率計算要篡改的區塊數。"""
    n = round(total_blocks * ratio_percent / 100.0)
    return max(1, min(total_blocks, n))

def actual_ratio_percent(tampered_blocks: int, block_size: int, total_pixels: int) -> int:
    """實際像素竄改比例（四捨五入為整數 %）。"""
    tampered_pixels = tampered_blocks * (block_size ** 2)
    return int(round(tampered_pixels / total_pixels * 100))

def collage_attack_two_images(
    img_watermarked: np.ndarray,
    source_image: np.ndarray,
    ratio_percent: int,
    *,
    block_size: int = BLOCK_SIZE,
) -> tuple[np.ndarray, int, int]:
    """
    對目標影像執行雙圖拼貼攻擊。
    回傳：(篡改圖, 篡改區塊數, 實際整數竄改率%)
    """
    # 確保兩張圖尺寸一致
    if img_watermarked.shape != source_image.shape:
        source_image = cv2.resize(source_image, (img_watermarked.shape[1], img_watermarked.shape[0]))

    h, w = img_watermarked.shape[:2]
    total_pixels = h * w
    total_blocks_x = w // block_size
    total_blocks_y = h // block_size
    total_blocks = total_blocks_x * total_blocks_y

    num_tamper_blocks = compute_tamper_blocks(total_blocks, ratio_percent)
    tampering_image = img_watermarked.copy()
    tampered_count = 0

    # 依序由左至右、由上至下進行區塊置換
    for i in range(total_blocks_y):
        for j in range(total_blocks_x):
            if tampered_count >= num_tamper_blocks:
                break

            y_target = i * block_size
            x_target = j * block_size
            
            # 從來源影像相同位置取區塊
            tampering_image[
                y_target : y_target + block_size,
                x_target : x_target + block_size,
            ] = source_image[
                y_target : y_target + block_size,
                x_target : x_target + block_size,
            ]
            tampered_count += 1
        if tampered_count >= num_tamper_blocks:
            break

    actual_percent = actual_ratio_percent(tampered_count, block_size, total_pixels)
    return tampering_image, tampered_count, actual_percent