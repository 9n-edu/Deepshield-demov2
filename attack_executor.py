from __future__ import annotations
import cv2
import random
import numpy as np

# 使用 try-except 進行安全匯入
try:
    from attack2CLA_two_images import ca_attack
except ImportError:
    ca_attack = None

try:
    from deletion_attack import deletion_attack
except ImportError:
    deletion_attack = None

try:
    from attack2CLA import collage_attack # 單圖平移
except ImportError:
    collage_attack = None

# 新增雙圖拼貼的匯入
try:
    from attack2CLA_two_images import collage_attack_two_images
except ImportError:
    collage_attack_two_images = None

try:
    from DRAWattack70 import doodle_attack_top_down
except ImportError:
    doodle_attack_top_down = None

try:
    from make_tamper_block import tamper_random_blocks
except ImportError:
    tamper_random_blocks = None

try:
    from make_tamper_image import run_manual_paint_array
except ImportError:
    run_manual_paint_array = None

IMG_SIZE = 128

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
    # 強制轉成 128x128
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img

def apply_attack(
    embedded: np.ndarray,
    attack_type: str,
    ratio_percent: int,
    *,
    source_image: np.ndarray | None = None,
    ca_side: str = "right",
    deletion_fill: int = 255,
    block_seed: int = 42,
    block_style: str = "fill",
    block_fill: int = 0,
) -> np.ndarray | tuple[np.ndarray, list[int]]:
    
    img = embedded.copy()
    h, w = img.shape[:2]
    attack_type = attack_type.lower()

    # 1. 塗鴉攻擊 (Doodle)
    if attack_type == "doodle":
        if doodle_attack_top_down is not None:
            try:
                res = doodle_attack_top_down(img, ratio_percent)
                return res[0] if isinstance(res, (tuple, list)) else res
            except Exception:
                pass
        # 備用機制
        out = img.copy()
        cut = int(h * (ratio_percent / 100.0))
        out[:cut, :] = 0
        return out

    # 2. 刪除攻擊 (Deletion)
    if attack_type == "deletion":
        if deletion_attack is not None:
            try:
                res = deletion_attack(img, ratio_percent, fill_value=deletion_fill)
                return res[0] if isinstance(res, (tuple, list)) else res
            except Exception:
                pass
        # 備用機制
        out = img.copy()
        scale = np.sqrt(max(0.05, min(0.95, ratio_percent / 100.0)))
        bh, bw = int(h * scale), int(w * scale)
        y1, x1 = (h - bh) // 2, (w - bw) // 2
        out[y1 : y1 + bh, x1 : x1 + bw] = 255
        return out

    # 3. 單圖區塊複製貼上 (copy_paste: 單張網格平移拼貼)
    if attack_type == "copy_paste":
        if collage_attack is not None:
            try:
                res = collage_attack(img, ratio_percent)
                return res[0] if isinstance(res, (tuple, list)) else res
            except Exception:
                pass
        # 備用機制
        out = img.copy()
        shift = max(2, int(w * (ratio_percent / 200.0)))
        out[:, shift:] = img[:, :-shift]
        return out

    # 4. 雙圖連續區塊置換 (ca / ca_attack)
    if attack_type in ["ca", "ca_attack"]:
        if source_image is None:
            src_u8 = np.rot90(img, 2)
        else:
            src_u8 = load_gray_image_from_array_or_path(source_image)
            
        if src_u8.shape != img.shape:
            src_u8 = cv2.resize(src_u8, (w, h))
            
        if ca_attack is not None:
            try:
                res = ca_attack(img, src_u8, ratio_percent, side=ca_side)
                return res[0] if isinstance(res, (tuple, list)) else res
            except Exception as e:
                print(f"CA Attack 失敗，進入備用機制: {e}")
                pass
        # 備用機制
        out = img.copy()
        cut = int(w * (ratio_percent / 100.0))
        out[:, w - cut :] = src_u8[:, w - cut :]
        return out

    # 4.5 雙圖網格拼貼 (collage)
    if attack_type == "collage":
        if source_image is None:
            src_u8 = np.rot90(img, 2)
        else:
            src_u8 = load_gray_image_from_array_or_path(source_image)
            
        if collage_attack_two_images is not None:
            try:
                res = collage_attack_two_images(img, src_u8, ratio_percent)
                return res[0] if isinstance(res, (tuple, list)) else res
            except Exception as e:
                print(f"雙圖 collage 失敗，進入備用機制: {e}")
                pass
        # 備用機制：簡單的網格置換
        out = img.copy()
        src_u8 = cv2.resize(src_u8, (w, h))
        block_size = 16
        n_blocks = round((w // block_size) * (h // block_size) * (ratio_percent / 100.0))
        count = 0
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                if count >= n_blocks:
                    break
                out[i:i+block_size, j:j+block_size] = src_u8[i:i+block_size, j:j+block_size]
                count += 1
            if count >= n_blocks:
                break
        return out

    # 5. 隨機區塊攻擊 (Random Block)
    if attack_type in ["random_block", "block"]:
        n_blocks = max(1, min(16, round(float(ratio_percent) / 100.0 * 16)))
        if tamper_random_blocks is not None:
            try:
                return tamper_random_blocks(
                    img,
                    n_blocks,
                    block_size=32,
                    seed=block_seed,
                    style=block_style, 
                    fill_value=block_fill,
                )
            except Exception:
                pass
        # 備用機制
        out = img.copy()
        rng = random.Random(block_seed)
        block_indices = list(range(16))
        rng.shuffle(block_indices)
        for b in block_indices[:n_blocks]:
            r = (b // 4) * 32
            c = (b % 4) * 32
            out[r : r + 32, c : c + 32] = 0
        return out, block_indices[:n_blocks]

    # 6. 手動塗鴉 (Manual Paint)
    if attack_type in ["manual_paint", "manual"]:
        if run_manual_paint_array is not None:
            try:
                painted = run_manual_paint_array(img)
                if painted is not None:
                    return painted
            except Exception:
                pass
        # 備用機制：預設打叉
        out = img.copy()
        cv2.line(out, (12, 12), (116, 116), 0, 8)
        cv2.line(out, (12, 116), (116, 12), 0, 8)
        return out

    # 7. 不破壞
    if attack_type == "none":
        return img

    raise ValueError(f"未知攻擊類型: {attack_type}")

def unpack_attack_result(result) -> tuple[np.ndarray, list[int] | None]:
    """統一解包函數，提取影像與被更動的區塊 ID"""
    if isinstance(result, tuple) and len(result) == 2:
        return result[0], result[1]
    return result, None