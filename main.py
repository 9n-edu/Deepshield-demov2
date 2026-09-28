import cv2
import numpy as np
import base64
import firebase_admin
import requests
from fastapi.middleware.cors import CORSMiddleware
from attack_executor import apply_attack, unpack_attack_result  # 新增這行匯入攻擊模組
from recovery_executor import recover_watermark
from audit_detail_executor import get_audit_details_data
from datetime import datetime
from firebase_admin import credentials, auth, firestore
from fastapi import FastAPI, Form, File, UploadFile
from embed_pipeline import get_device, load_model, embed_watermark
import os
import firebase_admin
from firebase_admin import credentials
from dotenv import load_dotenv

# 1. 載入 .env 檔案內的環境變數
load_dotenv()

# 2. 取得環境變數設定的路徑
cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH")

# 3. 檢查設定是否存在且檔案是否存在
if not cred_path or not os.path.exists(cred_path):
    raise FileNotFoundError(
        f"找不到 Firebase 金鑰檔案：'{cred_path}'。"
        "請確認專案目錄下已建立 .env 並設定正確的金鑰檔案路徑。"
    )

# 4. 初始化 Firebase Admin SDK
cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred)

app = FastAPI()

# 新增這段 CORS 設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = firestore.client()

# 預載入浮水印模型
device = get_device()
watermark_model = load_model(device)

@app.post("/api/add_account")
async def add_account(
    email: str = Form(...),
    password: str = Form(...),
    name: str = Form(...),
    gender: str = Form(...),
    department: str = Form(...),
    position: str = Form(...),
    photo: UploadFile = File(...)
):
    # 1. 讀取並轉換前端上傳的照片 (強制轉為灰階，以符合浮水印模型需求)
    contents = await photo.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # 2. 進行浮水印加密 [呼叫拆分後的 Pipeline]
    result = embed_watermark(image, watermark_model, device)
    encrypted_image = result.embedded

    # 3. 將加密後的影像轉為 PNG，再轉成 Base64 字串
    _, encoded_image = cv2.imencode('.png', encrypted_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    base64_bytes = base64.b64encode(encoded_image)
    base64_string = base64_bytes.decode('utf-8')

    # 4. 產生從 0000 開始的流水號員工編號
    dept_map = {"研發部": "RD", "業務部": "SA", "人資部": "HR"}
    pos_map = {"主管": "M", "一般員工": "E"}
    
    dept_code = dept_map.get(department, "XX")
    pos_code = pos_map.get(position, "X")
    
    # 查詢目前資料庫中 employees 集合的資料筆數
    existing_employees = db.collection('employees').get()
    current_count = len(existing_employees)
    
    # 將目前的總數補齊 4 位數 (0 -> "0000", 1 -> "0001", 12 -> "0012")
    serial_num = f"{current_count:04d}"
    
    emp_id = f"{dept_code}{pos_code}-{serial_num}"

    hire_date = datetime.now().strftime("%Y-%m-%d")

    # 5. 在 Firebase Auth 建立帳號 (處理登入用)
    user_record = auth.create_user(
        email=email,
        password=password,
        display_name=name
    )

    # 6. 將所有資料（包含照片的 Base64 字串與員工編號）寫入 Firestore
    doc_ref = db.collection('employees').document(user_record.uid)
    doc_ref.set({
        'emp_id': emp_id,
        'email': email,
        'name': name,
        'gender': gender,
        'department': department,
        'position': position,
        'hire_date': hire_date,
        'encrypted_photo_base64': f"data:image/png;base64,{base64_string}",
        'role_level': 'admin' if position == '主管' else 'user'
    })

    return {
        "status": "success", 
        "uid": user_record.uid, 
        "emp_id": emp_id, 
        "message": "帳戶建立成功，圖片已轉存為 Base64 存入 Firestore"
    }

@app.get("/api/get_employee/{emp_id}")
async def get_employee(emp_id: str):
    # 從 Firestore 的 employees 集合中，尋找符合該 emp_id (員工編號) 的文件
    docs = db.collection('employees').where('emp_id', '==', emp_id).stream()
    
    for doc in docs:
        data = doc.to_dict()
        return {"status": "success", "data": data}
        
    return {"status": "error", "message": "找不到該員工編號的資料"}

FIREBASE_WEB_API_KEY = "AIzaSyCSsfq9mDmK39YCRiwiTJyW77oNcCMmQAs"

@app.post("/api/login")
async def login(emp_id: str = Form(...), password: str = Form(...)):
    docs = db.collection('employees').where('emp_id', '==', emp_id).stream()
    
    user_data = None
    for doc in docs:
        user_data = doc.to_dict()
        break 

    if not user_data:
        return {"status": "error", "message": "找不到此員工編號，請確認輸入是否正確"}

    email = user_data.get('email')

    # 2. 呼叫 Firebase REST API 進行密碼驗證
    try:
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_WEB_API_KEY}"
        payload = {
            "email": email,
            "password": password,
            "returnSecureToken": True
        }
        response = requests.post(url, json=payload)
        result = response.json()

        # 如果 Firebase 回傳錯誤 (例如密碼錯誤)
        if "error" in result:
            return {"status": "error", "message": "密碼錯誤，請再試一次"}

        # 登入成功
        return {
            "status": "success", 
            "message": "登入成功", 
            "emp_id": emp_id,
            "role_level": user_data.get('role_level'),
            "department": user_data.get('department')
        }
        
    except Exception as e:
        return {"status": "error", "message": f"驗證系統連線失敗: {str(e)}"}

@app.get("/api/get_department_employees/{department}")
async def get_department_employees(department: str):
    docs = db.collection('employees').where('department', '==', department).stream()
    
    employees = []
    for doc in docs:
        employees.append(doc.to_dict())
        
    return {"status": "success", "data": employees}

@app.get("/api/get_all_employees")
async def get_all_employees():
    try:
        # 直接撈取 employees 集合內的所有文件 (不設條件)
        docs = db.collection('employees').stream()
        
        employees = []
        for doc in docs:
            data = doc.to_dict()
            if 'emp_id' in data and 'name' in data:
                employees.append({
                    "emp_id": data['emp_id'],
                    "name": data['name'],
                    "department": data.get('department', '未知')
                })
                
        # 依照員工編號排序，讓選單比較整齊
        employees = sorted(employees, key=lambda x: x['emp_id'])
        
        return {"status": "success", "data": employees}
    except Exception as e:
        return {"status": "error", "message": f"無法取得名單: {str(e)}"}

@app.post("/api/execute_attack")
async def execute_attack(
    emp_id: str = Form(...),
    attack_type: str = Form(...),
    ratio: int = Form(...)
):
    # 1. 尋找目標員工
    docs = db.collection('employees').where('emp_id', '==', emp_id).stream()
    user_doc = None
    for doc in docs:
        user_doc = doc
        break

    if not user_doc:
        return {"status": "error", "message": "資料庫中找不到此員工編號！"}

    # 2. 解碼資料庫中的 Base64 影像還原成 OpenCV 格式
    data = user_doc.to_dict()
    base64_str = data['encrypted_photo_base64'].split(",")[1]
    img_bytes = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    try:
        # 3. 呼叫 attack_executor 進行影像攻擊
        result = apply_attack(img, attack_type, ratio)
        tampered_img, _ = unpack_attack_result(result)

        # 4. 將竄改後的影像無壓縮轉回 PNG Base64
        _, encoded_image = cv2.imencode('.png', tampered_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        new_base64_string = base64.b64encode(encoded_image).decode('utf-8')
        new_b64 = f"data:image/png;base64,{new_base64_string}"

        # 5. 強行覆蓋 Firestore 中的原始影像 (模擬駭客竄改)
        user_doc.reference.update({
            'encrypted_photo_base64': new_b64
        })

        return {
            "status": "success", 
            "message": f"成功以 {attack_type} ({ratio}%) 竄改並覆蓋資料庫影像！",
            "new_image": new_b64
        }
    except Exception as e:
        return {"status": "error", "message": f"攻擊腳本執行失敗: {str(e)}"}

@app.post("/api/audit_and_recover")
async def audit_and_recover(emp_id: str = Form(...)):
    # 1. 取得資料庫中目前的影像 (可能是被駭客竄改過的)
    docs = db.collection('employees').where('emp_id', '==', emp_id).stream()
    user_doc = None
    for doc in docs:
        user_doc = doc
        break

    if not user_doc:
        return {"status": "error", "message": "找不到此員工資料"}

    data = user_doc.to_dict()
    base64_str = data['encrypted_photo_base64'].split(",")[1]
    img_bytes = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_bytes, np.uint8)
    tampered_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    try:
        # 2. 呼叫修復模型進行偵測與還原
        result = recover_watermark(tampered_img, watermark_model, device)
        
        # result.detection_mask 是偵測到的竄改區域 (黑白圖)
        # result.recovered 是修復後的乾淨影像
        
        # 3. 將偵測遮罩轉為 Base64 (供前端顯示哪裡被改過)
        _, mask_encoded = cv2.imencode('.png', result.detection_mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        mask_b64 = f"data:image/png;base64,{base64.b64encode(mask_encoded).decode('utf-8')}"

        # 4. 將修復後的影像轉為 Base64
        _, recovered_encoded = cv2.imencode('.png', result.recovered, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        recovered_b64 = f"data:image/png;base64,{base64.b64encode(recovered_encoded).decode('utf-8')}"

        # 5. 【資安行動】將修復後的乾淨影像覆蓋回資料庫，完成資料救援
        user_doc.reference.update({
            'encrypted_photo_base64': recovered_b64
        })

        return {
            "status": "success", 
            "message": "偵測與修復完成，已將乾淨影像存回資料庫！",
            "mask_image": mask_b64,
            "recovered_image": recovered_b64
        }
    except Exception as e:
        return {"status": "error", "message": f"修復失敗: {str(e)}"}

@app.get("/api/auto_verify/{emp_id}")
async def auto_verify(emp_id: str):
    docs = db.collection('employees').where('emp_id', '==', emp_id).stream()
    user_doc = None
    for doc in docs:
        user_doc = doc
        break

    if not user_doc:
        return {"status": "error", "message": "找不到員工"}

    data = user_doc.to_dict()
    base64_str = data['encrypted_photo_base64'].split(",")[1]
    img_bytes = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_bytes, np.uint8)
    current_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    try:
        from recovery_executor import recover_watermark
        result = recover_watermark(current_img, watermark_model, device)
        
        # 計算遮罩中是否有代表竄改的白色像素 (設定一個微小的容錯閾值如 10)
        tampered_pixels = cv2.countNonZero(result.detection_mask)
        is_tampered = tampered_pixels > 10

        if is_tampered:
            # 將修復後的影像無損轉為 Base64
            _, recovered_encoded = cv2.imencode('.png', result.recovered, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            recovered_b64 = f"data:image/png;base64,{base64.b64encode(recovered_encoded).decode('utf-8')}"

            # 自動修復：將乾淨影像覆蓋回資料庫
            user_doc.reference.update({
                'encrypted_photo_base64': recovered_b64
            })

            return {
                "status": "success", 
                "is_tampered": True, 
                "image": recovered_b64
            }
        else:
            return {
                "status": "success", 
                "is_tampered": False, 
                "image": data['encrypted_photo_base64']
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/reupload_photo")
async def reupload_photo(
    emp_id: str = Form(...),
    photo: UploadFile = File(...)
):
    # 1. 在資料庫中尋找該員工
    docs = db.collection('employees').where('emp_id', '==', emp_id).stream()
    user_doc = None
    for doc in docs:
        user_doc = doc
        break

    if not user_doc:
        return {"status": "error", "message": "找不到此員工資料"}

    try:
        # 2. 讀取前端傳來的新影像並轉為灰階
        contents = await photo.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        # 3. 為新影像重新嵌入浮水印
        result = embed_watermark(image, watermark_model, device)
        
        # 4. 轉為 Base64 字串
        _, encoded_image = cv2.imencode('.png', result.embedded, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        new_b64 = f"data:image/png;base64,{base64.b64encode(encoded_image).decode('utf-8')}"

        # 5. 更新資料庫：覆蓋新影像，並加入「隱藏的竄改歷史標記」
        user_doc.reference.update({
            'encrypted_photo_base64': new_b64,
            'has_tampered_history': True  # 後端專用的隱藏標記
        })

        return {
            "status": "success", 
            "message": "影像重新上傳成功，系統已重置您的相片！",
            "new_image": new_b64
        }
    except Exception as e:
        return {"status": "error", "message": f"影像處理失敗: {str(e)}"}

@app.get("/api/admin_audit_details/{emp_id}")
async def admin_audit_details(emp_id: str):
    # 1. 從資料庫取得目標員工影像
    docs = db.collection('employees').where('emp_id', '==', emp_id).stream()
    user_doc = next((doc for doc in docs), None)
    if not user_doc:
        return {"status": "error", "message": "找不到員工資料"}

    data = user_doc.to_dict()
    base64_str = data['encrypted_photo_base64'].split(",")[1]
    img_bytes = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_bytes, np.uint8)
    db_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    try:
        # 2. 直接呼叫獨立檔案獲取所有稽核細節資料 (完全不依賴 demo_pipeline)
        result_data = get_audit_details_data(db_img, watermark_model, device)
        result_data["status"] = "success"
        
        return result_data
    except Exception as e:
        return {"status": "error", "message": str(e)}