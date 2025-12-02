"""
Configuration overrides for training YOLO DDS model on PASCAL VOC2012.
This keeps the original DDS config intact while providing VOC-specific
paths and class definitions.
"""
import os
from pathlib import Path

# Base paths
REPO_ROOT = Path(__file__).parent
VOC_ROOT = Path(os.getenv("VOC_ROOT", "E:/VOC"))
TRAINVAL_DIR = VOC_ROOT / "VOC2012_train_val"
TEST_DIR = VOC_ROOT / "VOC2012_test"

# Dataset structure mirrors standard VOC2012 layout
IMAGES_SUBDIR = "JPEGImages"
ANNOTATIONS_SUBDIR = "Annotations"
IMAGESETS_DIRNAME = "ImageSets/Main"

SPLITS = ["train", "val", "test"]
DATASET_DIRS = {
    "train": TRAINVAL_DIR,
    "val": TRAINVAL_DIR,
    "test": TEST_DIR,
}

IMAGES_DIRS = {split: DATASET_DIRS[split] / IMAGES_SUBDIR for split in SPLITS}
ANNOTATION_DIRS = {split: DATASET_DIRS[split] / ANNOTATIONS_SUBDIR for split in SPLITS}
SPLIT_FILES = {
    split: DATASET_DIRS[split] / IMAGESETS_DIRNAME / f"{split}.txt" for split in ["train", "val"]
}
# There is no official test annotation list for VOC2012_test; inference users can
# point to their own list if needed.
SPLIT_FILES["test"] = DATASET_DIRS["test"] / IMAGESETS_DIRNAME / "test.txt"

# Training hyperparameters (can be tuned as needed)
INPUT_SIZE = 640
BATCH_SIZE = 4
NUM_WORKERS = 4
EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
WARMUP_EPOCHS = 3
MOMENTUM = 0.9

# Model settings
ANCHORS = [
    (40, 110),
    (56, 140),
    (72, 180),
]
STRIDE = 16

CLASS_NAMES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
NUM_CLASSES = len(CLASS_NAMES)

# Training artifacts
CHECKPOINT_DIR = REPO_ROOT / "checkpoints_voc"
LOG_DIR = REPO_ROOT / "logs_voc"
EVAL_DEBUG_DIR = REPO_ROOT / "debug_eval_voc"
DEVICE = "cuda"

# Inference thresholds
CONF_THRESHOLD = 0.35
VAL_CONF_THRESHOLD = 0.20
NMS_IOU_THRESHOLD = 0.5
MAX_DETECTIONS = 30
OBJ_IGNORE_IOU = 0.5

# Random seeds
SEED = 42
