import os

# Classes and mapping
CLASSES = [
    "accident",
    "call",
    "doctor",
    "help",
    "hot",
    "lose",
    "pain",
    "thief",
]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
ID_TO_CLASS = {i: c for c, i in CLASS_TO_ID.items()}
NUM_CLASSES = len(CLASSES)

# Frame extraction and image settings
FRAMES_PER_VIDEO = 5
IMG_H, IMG_W = 150, 150
IMG_C = 3

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
# Use relative paths from project root
RAW_VIDEOS_DIR = os.path.join(PROJECT_ROOT, "Dataset", "Raw_Data")
CROPPED_VIDEOS_DIR = os.path.join(PROJECT_ROOT, "Dataset", "Cropped_Data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
FRAMES_DIR = os.path.join(PROCESSED_DIR, "frames")  # per-class folders with images
SPLIT_DIR = os.path.join(PROCESSED_DIR, "splits")   # train/val/test txt lists
ANNOTATIONS_DIR = os.path.join(DATA_DIR, "annotations")  # YOLO labels

# Splits
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

# Training hyperparameters (classification)
LR = 1e-3
BATCH_SIZE = 32
EPOCHS = 50

# YOLOv5 specific configuration
YOLO_IMG_SIZE = 640  # Standard YOLOv5 input size
YOLO_BATCH_SIZE = 16
YOLO_EPOCHS = 100
YOLO_CONF_THRESHOLD = 0.25
YOLO_IOU_THRESHOLD = 0.45

# Model and artifacts
ARTIFACTS_DIR = os.path.join(PROCESSED_DIR, "artifacts")
MODEL_DIR = os.path.join(ARTIFACTS_DIR, "models")
LOGS_DIR = os.path.join(ARTIFACTS_DIR, "logs")
REPORTS_DIR = os.path.join(ARTIFACTS_DIR, "reports")

os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(SPLIT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
