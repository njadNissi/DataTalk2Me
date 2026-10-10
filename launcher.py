# launcher.py
import os
import sys
from pathlib import Path

from streamlit.web import cli


def main():
    app_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    app_path = app_root / "app.py"
    if not app_path.is_file():
        raise FileNotFoundError(f"Could not find the bundled Streamlit app: {app_path}")

    os.chdir(app_root)
    sys.path.insert(0, str(app_root))
    sys.argv = [
        "streamlit", "run", "app.py",
        "--global.developmentMode=false",
        "--server.headless=false"
    ]
    sys.exit(cli.main())


if __name__ == "__main__":
    main()
