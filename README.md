# YOLO DDS

A lightweight PyTorch implementation of a YOLO-tiny style detector for the Dental OPG XRAY Dataset (DDS). The repository includes data loading, model definition, training, evaluation, and qualitative inference utilities.

## Repository structure
- `config.py`: global paths, hyperparameters, anchors, and class names.
- `datasets/dental_dds.py`: dataset loader, target builder, and visualization helper.
- `models/backbone.py`: tiny CNN backbone.
- `models/head.py`: YOLO-tiny style detection head.
- `models/yolotiny.py`: complete model with decoding and NMS utilities.
- `losses/detection_loss.py`: GIoU-based detection loss with objectness and classification terms.
- `train.py`: training loop with validation and checkpointing.
- `eval.py`: evaluation on validation or test splits.
- `infer_visualize.py`: inference and qualitative visualization on test images.
- `utils/boxes.py`: box conversions and GIoU helper.
- `utils/metrics.py`: mAP and IoU computation utilities.

## Dataset setup
1. Download the official Dental OPG XRAY Dataset from the authors.
2. Place it under the repository root as `./Dental_OPG_XRAY_Dataset` with the same folder names as provided by the authors. Expected folders:
   - `images/{train,val,test}` containing `.jpg` or `.png` files.
   - `annotations/{train,val,test}` containing Pascal VOC-style `.xml` annotations.
   - `splits/train.txt`, `splits/val.txt`, `splits/test.txt` listing image basenames (without extensions) per split.
3. Update `config.CLASS_NAMES` if your downloaded annotations contain different class labels.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision matplotlib pillow
```

## Training
```bash
python train.py --epochs 50
```
The script prints one line per epoch with training loss, validation loss, validation mAP@0.5, and mean IoU. The best checkpoint (by validation mAP) is saved to `checkpoints/best.pt`; the final epoch is saved to `checkpoints/last.pt`.

## Evaluation
```bash
python eval.py --split test --checkpoint checkpoints/best.pt
```
Outputs overall mAP@0.5, mean IoU, and per-class AP on the requested split.

## Inference and visualization
```bash
python infer_visualize.py --checkpoint checkpoints/best.pt --output inference_outputs --num 8
```
Ground-truth boxes are drawn in green, predicted boxes and scores in red. Results are saved as PNGs in the specified output directory.

## Notes
- All box coordinates are treated in a normalized 0-1 space internally to ensure consistency across training, decoding, evaluation, and visualization.
- Anchors in `config.py` are defined in pixels for the configured `INPUT_SIZE`; adjust them if you change the input resolution or want to better fit DDS statistics.
- The code defaults to CUDA when available; set `config.DEVICE = "cpu"` if you want to force CPU execution.
