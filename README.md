當你的組員從 GitHub 下載（Clone）此專案時，他們的流程會是：

安裝依賴：

pip install -r requirements.txt
複製範本產生自己的 .env：

Copy-Item .env.example .env
放入私下取得的金鑰檔案：
把向你取得的 firebase_key.json 丟到專案目錄。

檢查 .env：
確認 .env 中的 FIREBASE_CREDENTIALS_PATH 對應的是該金鑰檔名。
