# Build the standalone Windows executable

The Windows executable includes the Python runtime, application dependencies,
Streamlit app source, and the bundled example datasets and logos. Firebase
credentials are not included. Without configured Firebase credentials, the app
uses its existing offline-auth behavior.

## Build locally on Windows

Install Python 3.11, open PowerShell in the project directory, and run:

```powershell
./build_windows_exe.ps1
```

The standalone executable is written to `dist/DataTalk2Me.exe`. It starts the
Streamlit app and opens it in the default browser.

## Build with GitHub Actions

Run the **Build Streamlit Windows EXE** workflow using the Actions
`workflow_dispatch` trigger. On completion, download the `DataTalk2Me` artifact
from the workflow run; it contains `DataTalk2Me.exe`.

The first run of AI background removal may need an internet connection to
download its ONNX model.
