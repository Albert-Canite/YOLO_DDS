"""
Global configuration for DDS YOLO-tiny training.
"""
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).parent
DATA_ROOT = REPO_ROOT / "Dental_OPG_XRAY_Dataset"
IMAGES_DIR = DATA_ROOT / "images"
ANNOTATIONS_DIR = DATA_ROOT / "annotations"
SPLITS_DIR = DATA_ROOT / "splits"

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

# Splits expected to exist in SPLITS_DIR with filenames train.txt, val.txt, test.txt
TRAIN_SPLIT_FILE = SPLITS_DIR / "train.txt"
VAL_SPLIT_FILE = SPLITS_DIR / "val.txt"
TEST_SPLIT_FILE = SPLITS_DIR / "test.txt"

# Random seeds
SEED = 42
