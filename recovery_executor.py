from __future__ import annotations
import cv2
import numpy as np
import torch
import random
from dataclasses import dataclass

# 匯入專案中原有的浮水印與密碼學處理模組
from Tong_class_pythonCodes import Class_extract_the_2LSB_values_bn256 as extract_bn256
from Tong_class_pythonCodes import reconstruction_image_Frombottleneck_test1 as recon_module
from Tong_class_pythonCodes import reduction_image_test2 as reduction_module
from Tong_class_pythonCodes import upset_ofKey_test2 as upset_ofKey

LATENT_DIM = 256
IMG_SIZE = 128
DEFAULT_SEED = 42

@dataclass
class RecoverResult:
    tampered: np.ndarray
    detection_v1: np.ndarray
    detection_v2: np.ndarray
    detection_step2: np.ndarray
    detection_step3: np.ndarray
    detection_mask: np.ndarray
    bottleneck_recon: np.ndarray
    recovered: np.ndarray
    true_block: list[int]
    pattern_count: int
    top_freq: int

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

def recover_watermark(
    tampered: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    *,
    seed: int = DEFAULT_SEED,
    bottleck_len: int = LATENT_DIM,
) -> RecoverResult:
    """
    執行影像竄改偵測與浮水印修復流程
    """
    tampering_image = load_gray_image_from_array_or_path(tampered)

    Disruption_operation = upset_ofKey.Disruption_operation()
    extractImage_To_Bottleneck = extract_bn256.extractImage_To_Bottleneck()
    Calculate_the_most_identical = extract_bn256.Calculate_the_most_identical(bottleck_len=bottleck_len)
    reconstruction_image_Frombottleneck = recon_module.read_model_bottleneck_recImage()
    reduction = reduction_module.reconstructed_LateFusion()

    info = extract_bn256.compute_block_and_key_counts(tampering_image.shape, bottleck_len)
    n_keys = info["n_keys"]
    random_numbers_list = generate_unique_random_numbers(seed, count=n_keys)

    extract_bottleneck = extractImage_To_Bottleneck.Tong_calculate(tampering_image, bottleck_len)
    extract_bottleneck_np = np.array(extract_bottleneck)
    extract_bottleneck_np_reshape = extract_bottleneck_np.reshape(
        extract_bottleneck_np.shape[0],
        extract_bottleneck_np.shape[1],
        extract_bottleneck_np.shape[2] * extract_bottleneck_np.shape[3],
    )

    counting = 0
    for i in range(extract_bottleneck_np_reshape.shape[0]):
        for j in range(extract_bottleneck_np_reshape.shape[1]):
            unscrambled = Disruption_operation.inverse_arnold_transform(
                bottleneck_lsit=extract_bottleneck_np_reshape[i, j],
                iterations=random_numbers_list[counting],
            )
            extract_bottleneck_np_reshape[i, j] = Disruption_operation.remove_xor_binding(
                bottleneck_lsit=unscrambled,
                block_id=counting,
            )
            counting += 1

    most_common = Calculate_the_most_identical.Tong_calculate(extract_bottleneck_np_reshape)
    true_block, top_freq = most_common[0]

    extract_list = extract_bottleneck_np_reshape.reshape(
        extract_bottleneck_np.shape[0],
        extract_bottleneck_np.shape[1],
        extract_bottleneck_np.shape[2],
        extract_bottleneck_np.shape[3],
    ).tolist()

    calcuation = extract_bn256.Calculate_Correct_Position_Of_the_picture(true_block, extract_list)
    v1, v2 = calcuation.Tong_calculate_Step1()
    step2 = calcuation.Tampering_point_detectio_recovery_OfKEY(
        random_numbers_list, v2, Disruption_operation
    )
    step3 = calcuation.Tong_calculate_Step3(step2)
    step4 = extract_bn256.Mathematical_morphology().Tong_calculate_Step4(step3, 5, 2)

    bottleneck_recon = reconstruction_image_Frombottleneck.rec_single_image(
        bottleneck_lsit=true_block, model=model, device=device
    )
    recovered = reduction.Tong_calculate_reconstructed_LateFusion(
        tampering_image=tampering_image,
        binary_image=step4,
        reconstructed_image_step1=bottleneck_recon,
    )

    return RecoverResult(
        tampered=tampering_image,
        detection_v1=v1,
        detection_v2=v2,
        detection_step2=step2,
        detection_step3=step3,
        detection_mask=step4,
        bottleneck_recon=bottleneck_recon,
        recovered=recovered,
        true_block=true_block,
        pattern_count=len(most_common),
        top_freq=top_freq,
    )