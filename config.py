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
    (40, 110),
    (56, 140),
    (72, 180),
]
STRIDE = 16

# The DDS annotations in this repo contain 6 classes (ids 0–5). Update the
# names if you have a different taxonomy, but keep the length in sync with
# NUM_CLASSES to avoid silently dropping annotations.
NUM_CLASSES = 6
CLASS_NAMES = [f"class_{i}" for i in range(NUM_CLASSES)]

# Training
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
LOG_DIR = REPO_ROOT / "logs"
DEVICE = "cuda"

# Inference
# Use a slightly lower validation threshold than inference to keep recall, but
# avoid flooding metrics with extremely low-score boxes that swamp IoU.
CONF_THRESHOLD = 0.35
VAL_CONF_THRESHOLD = 0.20
NMS_IOU_THRESHOLD = 0.5
MAX_DETECTIONS = 30
OBJ_IGNORE_IOU = 0.5

# Random seeds
SEED = 42
