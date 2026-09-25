import os
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv

def launch_label_studio():
    # 1. Locate and load the .env file dynamically
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    dotenv_path = project_root / ".env"

    if load_dotenv(dotenv_path=dotenv_path):
        print(f"✅ Environment variables loaded from: {dotenv_path}")
    else:
        print(f"⚠️ Warning: No .env file found at {dotenv_path}.")

    # 2. Check and validate the local storage path
    doc_root = os.getenv("LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT")
    if not doc_root:
        print("❌ Error: LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT is not set in your .env file.", file=sys.stderr)
        return
        
    local_path = Path(doc_root)
    if not local_path.exists():
        print(f"❌ Error: Local storage directory does NOT exist at:\n   {local_path}", file=sys.stderr)
        print("   Please check your .env path configuration or create the folder structure.", file=sys.stderr)
        return
    else:
        print(f"📂 Verified Local Storage Path: {local_path.resolve()}")

    # 3. Read host and port configuration
    host = os.getenv("HOST", "localhost")
    port = os.getenv("PORT", "8080")
    print(f"🚀 Starting Label Studio on http://{host}:{port} ...\n")

    # 4. Construct execution command with --no-browser flag
    command = ["label-studio", "start", "--host", host, "--port", port, "--no-browser"]

    try:
        # Run label-studio with the loaded environment variables
        subprocess.run(command, env=os.environ, check=True)
    except KeyboardInterrupt:
        print("\n👋 Label Studio server stopped by user.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error launching Label Studio: {e}", file=sys.stderr)
    except FileNotFoundError:
        print("\n❌ Error: 'label-studio' command not found. Ensure it is installed in your active virtual environment.", file=sys.stderr)

if __name__ == "__main__":
    launch_label_studio()

