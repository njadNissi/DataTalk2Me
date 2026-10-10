$ErrorActionPreference = "Stop"

python -m pip install -r requirements.txt
python -m pip install pyinstaller

python -m PyInstaller `
    --clean `
    --noconfirm `
    --onefile `
    --windowed `
    --name DataTalk2Me `
    --add-data "app.py;." `
    --add-data "src;src" `
    --add-data "data;data" `
    --collect-all streamlit `
    --collect-all streamlit_cropper `
    --collect-all pandas `
    --collect-all numpy `
    --collect-all sklearn `
    --collect-all scipy `
    --collect-all matplotlib `
    --collect-all seaborn `
    --collect-all plotly `
    --collect-all PIL `
    --collect-all cv2 `
    --collect-all rembg `
    --collect-all onnxruntime `
    --collect-all firebase_admin `
    --collect-all google.cloud.firestore `
    --collect-all bcrypt `
    --collect-all pydotplus `
    --collect-all graphviz `
    launcher.py
