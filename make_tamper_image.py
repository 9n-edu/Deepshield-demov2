from __future__ import annotations
import cv2
import numpy as np

def run_manual_paint_array(img: np.ndarray) -> np.ndarray | None:
    """
    開啟 OpenCV 視窗供使用者手動塗鴉攻擊。
    - 滑鼠左鍵拖曳：畫圖 (預設為黑色筆刷)
    - 按 's' 或 'Enter'：完成塗鴉並送出
    - 按 'r'：清除畫布重來
    - 按 'ESC'：取消攻擊 (回傳 None)
    """
    # 建立圖片副本，避免直接覆蓋原圖
    img_display = img.copy()
    img_original = img.copy()
    
    # 繪圖狀態變數
    drawing = False
    ix, iy = -1, -1

    # 滑鼠事件回呼函式
    def draw_doodle(event, x, y, flags, param):
        nonlocal drawing, ix, iy, img_display
        
        # 按下滑鼠左鍵開始畫
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
            cv2.circle(img_display, (x, y), 2, 0, -1)  # 點一下也有痕跡 (顏色0=黑)
            
        # 拖曳滑鼠時畫線
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                cv2.line(img_display, (ix, iy), (x, y), 0, 5) # 筆刷粗細為5
                ix, iy = x, y
                
        # 放開滑鼠左鍵停止畫
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.line(img_display, (ix, iy), (x, y), 0, 5)

    window_name = 'Manual Tamper (Press S: Save, R: Reset, ESC: Cancel)'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_doodle)

    print(">> 彈出塗鴉視窗：請用滑鼠在圖片上塗鴉。按 's' 儲存，按 'ESC' 取消。")

    while True:
        cv2.imshow(window_name, img_display)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # 按下 ESC 鍵取消
            print(">> 已取消手動塗鴉。")
            cv2.destroyWindow(window_name)
            return None
            
        elif key == ord('s') or key == 13:  # 按下 's' 鍵或 Enter 鍵儲存
            print(">> 塗鴉完成，準備覆蓋影像！")
            cv2.destroyWindow(window_name)
            return img_display
            
        elif key == ord('r'):  # 按下 'r' 鍵重置圖片
            print(">> 畫布已重置。")
            img_display = img_original.copy()

    cv2.destroyWindow(window_name)
    return None

if __name__ == "__main__":
    # 簡單的獨立測試邏輯 (直接執行此檔案時才會跑)
    dummy_img = np.ones((256, 256), dtype=np.uint8) * 200
    res = run_manual_paint_array(dummy_img)
    if res is not None:
        cv2.imshow("Result", res)
        cv2.waitKey(0)