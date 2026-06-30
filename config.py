import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./outputs"))

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Training configurations
IMG_SIZE = {"efficientnet_b3": (300, 300), 
            "vit_b_16": (224, 224),
            "convnextv2_tiny": (300, 300)}
BATCH_SIZE = 32

# Ensure we deal with absolute paths if they are relative
if not DATA_DIR.is_absolute():
    DATA_DIR = (BASE_DIR / DATA_DIR).resolve()
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = (BASE_DIR / OUTPUT_DIR).resolve()

