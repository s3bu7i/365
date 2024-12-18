import os
import subprocess
from datetime import datetime

# Düzəliş ediləcək fayl adı
FILE_NAME = "dummy_file.txt"

# Mesaj üçün tarix
def get_commit_message():
    return f"Auto-commit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

# Faylı redaktə etmək
def modify_file():
    with open(FILE_NAME, "a") as file:
        file.write(f"Update at {datetime.now()}\n")

# Git komutları işlətmək
def git_commit_and_push():
    try:
        # Faylı redaktə et
        modify_file()
        
        # Git add və commit
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", get_commit_message()], check=True)
        
        # Git push
        subprocess.run(["git", "push"], check=True)
        print("Changes pushed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during Git operation: {e}")

# Skripti işlət
if __name__ == "__main__":
    git_commit_and_push()
