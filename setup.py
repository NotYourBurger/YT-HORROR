import subprocess
import sys

def setup_environment():
    """Install required dependencies for the Horror Story Generator"""
    print("Installing required packages...")
    
    # Install requirements
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    print("Setup complete! You can now run the app with 'python app.py'")

if __name__ == "__main__":
    setup_environment() 