# launcher.py
import sys
from streamlit.web import cli

def main():
    sys.argv = [
        "streamlit", "run", "app.py",
        "--global.developmentMode=false",
        "--server.headless=true"
    ]
    sys.exit(cli.main())

if __name__ == "__main__":
    main()
