# TSdet-UAV

Official code release for the TSdet-UAV project.

## Dataset

- Hugging Face:
  [https://huggingface.co/datasets/huaddd/tea_shoot](https://huggingface.co/datasets/huaddd/tea_shoot)

## Model

- Model config:
  `ultralytics/cfg/models/11/yolo11-CARAFE-BI-LSCD.yaml`
- Main custom modules:
  - `C2PSA_BinaryAttn`
  - `CARAFE`
  - `Detect_LSCD`

## Sliced Inference

- Original script:
  `scripts/sw.py`
- Hash-bucket + local NMS script:
  `scripts/sw_hash_local_nms.py`

## Quick Start

Train:

```bash
yolo detect train model=ultralytics/cfg/models/11/yolo11-CARAFE-BI-LSCD.yaml data=your_data.yaml epochs=300 imgsz=640
```

Sliced inference:

```bash
python scripts/sw_hash_local_nms.py --model-path runs/detect/train/weights/best.pt --source test-data/images --output-dir output/sw_hash_local_nms --device cuda:0 --imgsz 640 --confidence 0.01 --iou-thres 0.5 --window-size 640 640 --step-percent 90 90 --merge-method hash_nms --hash-cell-size 64 --hash-neighbor-radius 1
```

## Notes

- A short project note is provided in `MODEL_SLICE.md`.
- This repository is built on top of Ultralytics YOLO.
