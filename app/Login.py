from huggingface_hub import login
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def validate_login():
    token = os.getenv("HUGGINGFACE_TOKENW")
    if not token:
        raise ValueError("HUGGINGFACE_TOKEN not found in .env file.")
    try:
        login(token=token, add_to_git_credential=True)
        print("Successfully logged in to Hugging Face.")
    except Exception as e:
        print(f"Failed to log in: {e}")

if __name__ == "__main__":
    validate_login()

