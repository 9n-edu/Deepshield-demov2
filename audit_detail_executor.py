import cv2
import numpy as np
import base64
import random
from torchvision import transforms

# 直接引入底層核心模組，不依賴 demo_pipeline
from Tong_class_pythonCodes import (
    read_model_predict_bottleneck_test2 as predict_bottleneck,
    upset_ofKey_test2 as upset_ofKey,
    Class_extract_the_2LSB_values_bn256 as extract_bn256,
    reconstruction_image_Frombottleneck_test1 as recon_module,
    reduction_image_test2 as reduction_module,
    LSB2_Embed_process_test5 as embed_lsb
)

LATENT_DIM = 256
DEFAULT_SEED = 42

def _to_b64(arr: np.ndarray) -> str:
    """將影像陣列轉換為 Base64 字串"""
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    _, buf = cv2.imencode(".png", arr)
    return base64.b64encode(buf).decode("utf-8")

def get_audit_details_data(img: np.ndarray, model, device, seed: int = DEFAULT_SEED) -> dict:
    """
    專門用於管理員後台的詳細過程提取。
    回傳 Embed(加密)、Recover(解密) 各區塊矩陣以及 6 階段修復影像。
    """
    # 1. 模組初始化
    disruption_op = upset_ofKey.Disruption_operation()
    extract_module = extract_bn256.extractImage_To_Bottleneck()
    calc_most_common = extract_bn256.Calculate_the_most_identical(bottleck_len=LATENT_DIM)
    recon_mod = recon_module.read_model_bottleneck_recImage()
    reduction = reduction_module.reconstructed_LateFusion()
    predictor = predict_bottleneck.read_model_predict_bottleneck()

    # 產生與影像對應的 16 把金鑰
    rng = random.Random(seed)
    pool = list(range(1, 101))
    rng.shuffle(pool)
    keys = pool[:16]

    # ==========================================
    # 階段 A：模擬加密過程 (擷取矩陣資料)
    # ==========================================
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((128, 128))])
    bottleneck_out = predictor.infer_single_image(img, model, transform, device)
    bottleneck = np.array(bottleneck_out[0], dtype=np.uint8)

    embed_blocks = {}
    for b_id in range(16):
        orig_matrix = disruption_op.Help_change_traits(bottleneck.tolist())
        xor_list = disruption_op.apply_xor_binding(bottleneck.tolist(), b_id)
        xor_matrix = disruption_op.Help_change_traits(xor_list)
        arnold_list = disruption_op.arnold_transform(xor_list, keys[b_id])
        arnold_matrix = disruption_op.Help_change_traits(arnold_list)

        embed_blocks[b_id] = {
            "arnold_iterations": keys[b_id],
            "orig_matrix": orig_matrix.tolist(),
            "xor_matrix": xor_matrix.tolist(),
            "arnold_matrix": arnold_matrix.tolist()
        }

    # ==========================================
    # 階段 B：執行修復與解密過程
    # ==========================================
    raw_ext = np.array(extract_module.Tong_calculate(img, LATENT_DIM))
    ext_reshaped = raw_ext.reshape(raw_ext.shape[0], raw_ext.shape[1], -1)

    recover_blocks = {}
    counting = 0
    for r in range(ext_reshaped.shape[0]):
        for c in range(ext_reshaped.shape[1]):
            b_id = counting
            iters = keys[b_id]
            ext_list = ext_reshaped[r, c].tolist()
            
            # 空間反轉與解碼
            unscrambled = disruption_op.inverse_arnold_transform(bottleneck_lsit=ext_list, iterations=iters)
            restored = disruption_op.remove_xor_binding(bottleneck_lsit=unscrambled, block_id=b_id)

            recover_blocks[b_id] = {
                "arnold_iterations": iters,
                "matrix_2lsb": disruption_op.Help_change_traits(ext_list).tolist(),
                "matrix_inv_arnold": disruption_op.Help_change_traits(unscrambled).tolist(),
                "matrix_restored": disruption_op.Help_change_traits(restored).tolist()
            }
            # 覆寫回陣列以供多數決使用
            ext_reshaped[r, c] = restored
            counting += 1

    # ==========================================
    # 階段 C：多數決與 6 階段遮罩定位
    # ==========================================
    most_common = calc_most_common.Tong_calculate(ext_reshaped)
    true_block, _ = most_common[0]
    
    extract_list = ext_reshaped.reshape(raw_ext.shape[0], raw_ext.shape[1], raw_ext.shape[2], raw_ext.shape[3]).tolist()
    calc_pos = extract_bn256.Calculate_Correct_Position_Of_the_picture(true_block, extract_list)

    v1, v2 = calc_pos.Tong_calculate_Step1()
    step2 = calc_pos.Tampering_point_detectio_recovery_OfKEY(keys, v2, disruption_op)
    step3 = calc_pos.Tong_calculate_Step3(step2)
    step4_mask = extract_bn256.Mathematical_morphology().Tong_calculate_Step4(step3, 5, 2)

    # 重建修復影像
    bottleneck_recon = recon_mod.rec_single_image(bottleneck_lsit=true_block, model=model, device=device)
    recovered_img = reduction.Tong_calculate_reconstructed_LateFusion(
        tampering_image=img,
        binary_image=step4_mask,
        reconstructed_image_step1=bottleneck_recon
    )

    return {
        "embed_blocks": {0: embed_blocks},
        "recover_blocks": recover_blocks,
        "images": {
            "tampered": _to_b64(img),
            "v1": _to_b64(v1),
            "v2": _to_b64(v2),
            "step2": _to_b64(step2),
            "step3": _to_b64(step3),
            "mask": _to_b64(step4_mask),
            "recovered": _to_b64(recovered_img)
        }
    }