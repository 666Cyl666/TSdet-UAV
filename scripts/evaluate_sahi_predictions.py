from __future__ import annotations

import argparse
import json
from statistics import mean
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IOU_THRESHOLDS = [round(0.50 + i * 0.05, 2) for i in range(10)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SAHI prediction json files against YOLO-format labels.")
    parser.add_argument(
        "--images-dir",
        default=r"D:\project\slicing_windows\new_slice_windows\test-data\images",
        help="Directory containing original images.",
    )
    parser.add_argument(
        "--labels-dir",
        default=r"D:\project\slicing_windows\new_slice_windows\test-data\labels",
        help="Directory containing YOLO txt labels.",
    )
    parser.add_argument(
        "--pred-dir",
        default=r"D:\project\slicing_windows\new_slice_windows\output\sahi\yolo26n_sw",
        help="Directory containing SAHI per-image prediction folders.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.2,
        help="Only predictions with score >= this threshold are evaluated.",
    )
    parser.add_argument(
        "--output-json",
        default=r"D:\project\slicing_windows\new_slice_windows\output\sahi\yolo26n_sw\eval_metrics.json",
        help="Path to save evaluation metrics as JSON.",
    )
    return parser.parse_args()


def list_images(images_dir: Path) -> list[Path]:
    images = [p for p in sorted(images_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    if not images:
        raise RuntimeError(f"No images found in: {images_dir}")
    return images


def yolo_to_xyxy(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int) -> list[float]:
    bw = w * img_w
    bh = h * img_h
    x1 = (xc * img_w) - bw / 2.0
    y1 = (yc * img_h) - bh / 2.0
    x2 = x1 + bw
    y2 = y1 + bh
    return [x1, y1, x2, y2]


def compute_iou(box1: list[float], box2: list[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0.0 else 0.0


def compute_ap(recall: list[float], precision: list[float]) -> float:
    mrec = [0.0, *recall, 1.0]
    mpre = [1.0, *precision, 0.0]

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    samples = [i / 100 for i in range(101)]
    interp_precision = []
    for sample in samples:
        idx = 0
        while idx < len(mrec) and mrec[idx] < sample:
            idx += 1
        if idx >= len(mpre):
            interp_precision.append(mpre[-1])
        else:
            interp_precision.append(mpre[idx])

    area = 0.0
    for i in range(1, len(samples)):
        area += (samples[i] - samples[i - 1]) * (interp_precision[i] + interp_precision[i - 1]) / 2.0
    return area


def load_ground_truth(images_dir: Path, labels_dir: Path) -> tuple[dict[str, list[dict]], dict[int, str]]:
    gt_by_image: dict[str, list[dict]] = {}
    class_ids: set[int] = set()

    for image_path in list_images(images_dir):
        stem = image_path.stem
        label_path = labels_dir / f"{stem}.txt"
        with Image.open(image_path) as image:
            img_w, img_h = image.size

        gts: list[dict] = []
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError(f"Invalid YOLO label line in {label_path}: {line}")
                cls = int(float(parts[0]))
                xc, yc, w, h = map(float, parts[1:])
                gts.append({"class_id": cls, "bbox": yolo_to_xyxy(xc, yc, w, h, img_w, img_h)})
                class_ids.add(cls)
        gt_by_image[stem] = gts

    class_names = {class_id: f"class_{class_id}" for class_id in sorted(class_ids)}
    return gt_by_image, class_names


def load_predictions(pred_dir: Path, score_threshold: float) -> tuple[dict[str, list[dict]], dict[int, str]]:
    pred_by_image: dict[str, list[dict]] = {}
    class_names: dict[int, str] = {}

    for image_dir in sorted(p for p in pred_dir.iterdir() if p.is_dir()):
        stem = image_dir.name
        json_path = image_dir / f"{stem}_predictions.json"
        if not json_path.exists():
            continue

        raw_predictions = json.loads(json_path.read_text(encoding="utf-8"))
        predictions: list[dict] = []
        for pred in raw_predictions:
            score = float(pred["score"])
            if score < score_threshold:
                continue
            class_id = int(pred["category_id"])
            class_names.setdefault(class_id, str(pred.get("category_name", f"class_{class_id}")))
            predictions.append(
                {
                    "class_id": class_id,
                    "score": score,
                    "bbox": [float(v) for v in pred["bbox_xyxy"]],
                }
            )
        pred_by_image[stem] = predictions

    return pred_by_image, class_names


def collect_class_stats(
    gt_by_image: dict[str, list[dict]],
    pred_by_image: dict[str, list[dict]],
    class_id: int,
    iou_threshold: float,
) -> tuple[int, list[tuple[float, int]]]:
    gt_count = 0
    matched = {
        image_id: [False] * sum(1 for gt in gts if gt["class_id"] == class_id)
        for image_id, gts in gt_by_image.items()
    }
    gt_boxes_by_image: dict[str, list[list[float]]] = {}

    for image_id, gts in gt_by_image.items():
        cls_boxes = [gt["bbox"] for gt in gts if gt["class_id"] == class_id]
        gt_boxes_by_image[image_id] = cls_boxes
        gt_count += len(cls_boxes)

    scored_matches: list[tuple[float, int]] = []
    for image_id, preds in pred_by_image.items():
        cls_preds = [pred for pred in preds if pred["class_id"] == class_id]
        if not cls_preds:
            continue

        gt_boxes = gt_boxes_by_image.get(image_id, [])
        used = matched.get(image_id, [])
        cls_preds.sort(key=lambda item: item["score"], reverse=True)
        for pred in cls_preds:
            best_iou = 0.0
            best_gt_idx = -1
            for gt_idx, gt_box in enumerate(gt_boxes):
                if used[gt_idx]:
                    continue
                iou = compute_iou(pred["bbox"], gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            if best_gt_idx >= 0 and best_iou >= iou_threshold:
                used[best_gt_idx] = True
                scored_matches.append((pred["score"], 1))
            else:
                scored_matches.append((pred["score"], 0))

    scored_matches.sort(key=lambda item: item[0], reverse=True)
    return gt_count, scored_matches


def precision_recall_from_matches(gt_count: int, scored_matches: list[tuple[float, int]]) -> tuple[float, float, float]:
    if not scored_matches:
        return 0.0, 0.0, 0.0

    tp = sum(match for _, match in scored_matches)
    fp = len(scored_matches) - tp
    fn = gt_count - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def evaluate_class(
    gt_by_image: dict[str, list[dict]],
    pred_by_image: dict[str, list[dict]],
    class_id: int,
) -> dict:
    ap_by_threshold: dict[str, float] = {}
    recall_by_threshold: dict[str, float] = {}
    ap_values: list[float] = []
    recall_values: list[float] = []
    prf_at_50 = (0.0, 0.0, 0.0)
    gt_count_50 = 0
    pred_count = sum(1 for preds in pred_by_image.values() for pred in preds if pred["class_id"] == class_id)

    for iou_threshold in IOU_THRESHOLDS:
        gt_count, scored_matches = collect_class_stats(gt_by_image, pred_by_image, class_id, float(iou_threshold))
        precision, recall, f1 = precision_recall_from_matches(gt_count, scored_matches)
        matches = [match for _, match in scored_matches]
        if gt_count == 0:
            ap = 0.0
        elif not matches:
            ap = 0.0
        else:
            tpc = []
            fpc = []
            tp_running = 0.0
            fp_running = 0.0
            for match in matches:
                if match:
                    tp_running += 1.0
                else:
                    fp_running += 1.0
                tpc.append(tp_running)
                fpc.append(fp_running)
            recall_curve = [tp / max(gt_count, 1) for tp in tpc]
            precision_curve = [tp / max(tp + fp, 1e-12) for tp, fp in zip(tpc, fpc)]
            ap = compute_ap(recall_curve, precision_curve)

        key = f"{iou_threshold:.2f}"
        ap_by_threshold[key] = round(ap, 6)
        recall_by_threshold[key] = round(recall, 6)
        ap_values.append(ap)
        recall_values.append(recall)

        if abs(iou_threshold - 0.50) < 1e-9:
            prf_at_50 = (precision, recall, f1)
            gt_count_50 = gt_count

    return {
        "class_id": class_id,
        "gt_count": gt_count_50,
        "pred_count": pred_count,
        "precision": round(prf_at_50[0], 6),
        "recall": round(prf_at_50[1], 6),
        "f1": round(prf_at_50[2], 6),
        "ap50": round(ap_by_threshold["0.50"], 6),
        "ap50_95": round(float(mean(ap_values)) if ap_values else 0.0, 6),
        "ar50_95": round(float(mean(recall_values)) if recall_values else 0.0, 6),
        "ap_by_iou": ap_by_threshold,
        "recall_by_iou": recall_by_threshold,
    }


def evaluate_image(
    image_id: str,
    gt_by_image: dict[str, list[dict]],
    pred_by_image: dict[str, list[dict]],
) -> dict:
    image_gt = gt_by_image.get(image_id, [])
    image_pred = pred_by_image.get(image_id, [])
    class_ids = sorted({item["class_id"] for item in image_gt} | {item["class_id"] for item in image_pred})

    if not class_ids:
        return {
            "image": image_id,
            "gt_boxes": 0,
            "pred_boxes": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mAP50": 0.0,
            "mAP50_95": 0.0,
            "mAR50_95": 0.0,
            "per_class": [],
        }

    gt_single = {image_id: image_gt}
    pred_single = {image_id: image_pred}
    class_results = [evaluate_class(gt_single, pred_single, class_id) for class_id in class_ids]
    return {
        "image": image_id,
        "gt_boxes": len(image_gt),
        "pred_boxes": len(image_pred),
        "precision": round(float(mean([item["precision"] for item in class_results])), 6),
        "recall": round(float(mean([item["recall"] for item in class_results])), 6),
        "f1": round(float(mean([item["f1"] for item in class_results])), 6),
        "mAP50": round(float(mean([item["ap50"] for item in class_results])), 6),
        "mAP50_95": round(float(mean([item["ap50_95"] for item in class_results])), 6),
        "mAR50_95": round(float(mean([item["ar50_95"] for item in class_results])), 6),
        "per_class": class_results,
    }


def main() -> None:
    args = parse_args()

    images_dir = Path(args.images_dir).resolve()
    labels_dir = Path(args.labels_dir).resolve()
    pred_dir = Path(args.pred_dir).resolve()
    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    gt_by_image, gt_class_names = load_ground_truth(images_dir, labels_dir)
    pred_by_image, pred_class_names = load_predictions(pred_dir, args.score_threshold)

    class_ids = sorted(set(gt_class_names) | set(pred_class_names))
    if not class_ids:
        raise RuntimeError("No classes found in labels or predictions.")

    class_results = []
    for class_id in class_ids:
        class_result = evaluate_class(gt_by_image, pred_by_image, class_id)
        class_result["class_name"] = pred_class_names.get(class_id, gt_class_names.get(class_id, f"class_{class_id}"))
        class_results.append(class_result)

    summary = {
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "pred_dir": str(pred_dir),
        "score_threshold": args.score_threshold,
        "num_images": len(gt_by_image),
        "num_classes": len(class_results),
        "total_gt_boxes": sum(item["gt_count"] for item in class_results),
        "total_pred_boxes": sum(item["pred_count"] for item in class_results),
        "precision": round(float(mean([item["precision"] for item in class_results])), 6),
        "recall": round(float(mean([item["recall"] for item in class_results])), 6),
        "f1": round(float(mean([item["f1"] for item in class_results])), 6),
        "mAP50": round(float(mean([item["ap50"] for item in class_results])), 6),
        "mAP50_95": round(float(mean([item["ap50_95"] for item in class_results])), 6),
        "mAR50_95": round(float(mean([item["ar50_95"] for item in class_results])), 6),
        "per_class": class_results,
    }

    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nDataset-level aggregate:")
    print(f"images={summary['num_images']}")
    print(f"classes={summary['num_classes']}")
    print(f"gt_boxes={summary['total_gt_boxes']}")
    print(f"pred_boxes={summary['total_pred_boxes']}")
    print(f"precision={summary['precision']:.6f}")
    print(f"recall={summary['recall']:.6f}")
    print(f"f1={summary['f1']:.6f}")
    print(f"mAP50={summary['mAP50']:.6f}")
    print(f"mAP50-95={summary['mAP50_95']:.6f}")
    print(f"mAR50-95={summary['mAR50_95']:.6f}")
    print(f"saved={output_json}")


if __name__ == "__main__":
    main()
