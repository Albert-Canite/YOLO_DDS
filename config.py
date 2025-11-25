"""
Global configuration for DDS YOLO-tiny training.
"""
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).parent

# Accept both underscored and space-separated dataset folder names for flexibility
PREFERRED_DATA_DIRS = [
    REPO_ROOT / "Dental_OPG_XRAY_Dataset",
    REPO_ROOT / "Dental OPG XRAY Dataset",
]
for candidate in PREFERRED_DATA_DIRS:
    if candidate.exists():
        DATA_ROOT = candidate
        break
else:
    DATA_ROOT = PREFERRED_DATA_DIRS[0]

# DDS release contains both augmented and original data. Default to augmented structure:
# Dental_OPG_XRAY_Dataset/
#   Augmented_Data/
#     train/{images,labels}
#     valid/{images,labels}
#     test/{images,labels}
AUGMENTED_ROOT = DATA_ROOT / "Augmented_Data"
ORIGINAL_ROOT = DATA_ROOT / "Original_Data"
DATASET_VARIANT = AUGMENTED_ROOT  # switch to ORIGINAL_ROOT if needed

IMAGES_SUBDIR = "images"
LABELS_SUBDIR = "labels"

SPLITS = ["train", "valid", "test"]

IMAGES_DIRS = {
    split: DATASET_VARIANT / split / IMAGES_SUBDIR for split in SPLITS
}
LABELS_DIRS = {
    split: DATASET_VARIANT / split / LABELS_SUBDIR for split in SPLITS
}

# Optional split files (if present). If absent, dataset loader will infer IDs from labels.
SPLITS_DIR = DATASET_VARIANT / "splits"
TRAIN_SPLIT_FILE = SPLITS_DIR / "train.txt"
VAL_SPLIT_FILE = SPLITS_DIR / "valid.txt"
TEST_SPLIT_FILE = SPLITS_DIR / "test.txt"

# Hyperparameters
INPUT_SIZE = 640
BATCH_SIZE = 4
NUM_WORKERS = 4
EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
WARMUP_EPOCHS = 3
MOMENTUM = 0.9

# Model settings
# Anchors are defined in pixels for the given INPUT_SIZE (width, height)
ANCHORS = [
    (32, 32),
    (64, 64),
    (128, 128),
]
STRIDE = 32
NUM_CLASSES = 3  # Update if DDS uses a different number of categories
CLASS_NAMES = [
    "tooth",  # placeholder classes; adjust to dataset specification
    "filling",
    "caries",
]

# Training
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
LOG_DIR = REPO_ROOT / "logs"
DEVICE = "cuda"

# Inference
CONF_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.5
MAX_DETECTIONS = 100

# Random seeds
SEED = 42
