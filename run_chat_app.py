#!/usr/bin/env python3
"""
Simple launcher script for the Bank of England Docs Assistant Streamlit app.
Run this script to start the chat interface.
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit chat application"""
    print("🏦 Starting Bank of England Docs Assistant...")
    print("=" * 50)
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(script_dir, "streamlit_chat_app.py")
    
    # Check if the app file exists
    if not os.path.exists(app_path):
        print(f"❌ Error: {app_path} not found!")
        sys.exit(1)
    
    try:
        # Launch Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run", app_path,
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ]
        
        print("🚀 Launching Streamlit app...")
        print(f"📂 App location: {app_path}")
        print("🌐 The app will open in your default browser at: http://localhost:8501")
        print("\nPress Ctrl+C to stop the application")
        print("=" * 50)
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down the application...")
    except Exception as e:
        print(f"❌ Error launching the application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
