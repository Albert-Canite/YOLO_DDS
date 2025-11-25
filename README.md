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
1. 下载官方 Dental OPG XRAY Dataset 后，解压到仓库根目录，主目录可以是 `./Dental_OPG_XRAY_Dataset` **或** `./Dental OPG XRAY Dataset`（包含空格）；代码会自动识别两种名称。
2. 数据包通常包含 `Augmented_Data`（含 `train/valid/test` 三个子目录）和 `Original_Data`。默认代码使用 `Augmented_Data`，预期结构：
   - `Dental_OPG_XRAY_Dataset/Augmented_Data/train/{images,labels}`
   - `Dental_OPG_XRAY_Dataset/Augmented_Data/valid/{images,labels}`
   - `Dental_OPG_XRAY_Dataset/Augmented_Data/test/{images,labels}`
   - 每个 split 下的 `labels/` 目录应包含与图片同名的 `.txt` 文件（YOLO 标注格式），如 `train/labels/xxx.txt` 对应 `train/images/xxx.jpg`。
   - 如果提供 `splits/train.txt|valid.txt|test.txt`（位于 `Augmented_Data/splits`），也会自动读取；否则直接按各 split 下的 `.txt` 标签文件推断列表。
   若要切换到原始数据，可将 `config.DATASET_VARIANT` 改为 `config.ORIGINAL_ROOT`，并保证结构类似。
3. 如果标注类别与默认不同，请在 `config.CLASS_NAMES` 中修改类别名称。

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
