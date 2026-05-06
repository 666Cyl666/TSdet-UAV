# Model and Sliced Inference Notes

## 1. Model

- Model config:
  `ultralytics/cfg/models/11/yolo11-CARAFE-BI-LSCD.yaml`
- Custom modules used by this model:
  - `C2PSA_BinaryAttn`
  - `CARAFE`
  - `Detect_LSCD`

Implementation locations:

- `ultralytics/nn/modules/block.py`
  - `C2PSA_BinaryAttn`
  - `CARAFE`
- `ultralytics/nn/modules/head.py`
  - `Detect_LSCD`
- `ultralytics/nn/modules/__init__.py`
  - module export
- `ultralytics/nn/tasks.py`
  - model parsing and registration

## 2. Sliced Inference

Script locations:

- `scripts/sw.py`
  - original sliced inference script
  - uses global `NMS/NMW`
- `scripts/sw_hash_local_nms.py`
  - sliced inference with hash buckets and local NMS
  - recommended script for the released code

Outputs:

- one subfolder per image
- each subfolder contains:
  - `xxx_predictions.json`
  - `xxx_result.jpg`
- the output root also contains:
  - `summary.json`

## 3. How to Use

### 3.1 Training

Run from the repository root:

```bash
yolo detect train model=ultralytics/cfg/models/11/yolo11-CARAFE-BI-LSCD.yaml data=your_data.yaml epochs=300 imgsz=640
```

### 3.2 Original sliced inference

```bash
python scripts/sw.py
```

`sw.py` uses fixed local paths inside the script, so it is intended for quick local use after editing the paths.

### 3.3 Hash-bucket + local NMS sliced inference

Recommended command:

```bash
python scripts/sw_hash_local_nms.py --model-path runs/detect/train/weights/best.pt --source test-data/images --output-dir output/sw_hash_local_nms --device cuda:0 --imgsz 640 --confidence 0.01 --iou-thres 0.5 --window-size 640 640 --step-percent 90 90 --merge-method hash_nms --hash-cell-size 64 --hash-neighbor-radius 1
```

