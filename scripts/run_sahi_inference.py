from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAHI sliced inference with an Ultralytics YOLO model.")
    parser.add_argument("--model-path", required=True, help="Path to YOLO weights, e.g. best.pt")
    parser.add_argument("--source", required=True, help="Image path or directory path")
    parser.add_argument("--output-dir", default="sahi_outputs", help="Directory to save visualizations and json")
    parser.add_argument("--device", default="cpu", help="Inference device, e.g. cpu, cuda:0")
    parser.add_argument("--confidence", type=float, default=0.01, help="Confidence threshold")
    parser.add_argument("--slice-height", type=int, default=640, help="SAHI slice height")
    parser.add_argument("--slice-width", type=int, default=640, help="SAHI slice width")
    parser.add_argument("--overlap-height-ratio", type=float, default=0.1, help="Slice overlap ratio on height")
    parser.add_argument("--overlap-width-ratio", type=float, default=0.1, help="Slice overlap ratio on width")
    parser.add_argument(
        "--postprocess-type",
        default="GREEDYNMM",
        help="SAHI postprocess type, e.g. GREEDYNMM, NMM, NMS",
    )
    parser.add_argument(
        "--postprocess-match-threshold",
        type=float,
        default=0.6,
        help="Postprocess match threshold",
    )
    parser.add_argument(
        "--postprocess-match-metric",
        default="IOU",
        help="Postprocess match metric, e.g. IOU or IOS",
    )
    return parser.parse_args()


def collect_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Source not found: {source}")

    images = [p for p in sorted(source.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    if not images:
        raise RuntimeError(f"No images found in directory: {source}")
    return images


def serialize_prediction(pred) -> dict:
    bbox = pred.bbox
    score = pred.score
    category = pred.category
    return {
        "bbox_xyxy": [float(bbox.minx), float(bbox.miny), float(bbox.maxx), float(bbox.maxy)],
        "bbox_xywh": [
            float(bbox.minx),
            float(bbox.miny),
            float(bbox.maxx - bbox.minx),
            float(bbox.maxy - bbox.miny),
        ],
        "score": float(score.value),
        "category_id": int(category.id),
        "category_name": str(category.name),
    }


def save_box_only_visual(image_path: Path, predictions: list[dict], output_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    for pred in predictions:
        minx, miny, maxx, maxy = pred["bbox_xyxy"]
        cv2.rectangle(
            image,
            (int(minx), int(miny)),
            (int(maxx), int(maxy)),
            (0, 255, 0),
            2,
        )

    cv2.imwrite(str(output_path), image)


def extract_timing(result, total_seconds: float) -> tuple[float, float]:
    """Split SAHI runtime into slice/merge and model inference."""
    durations = getattr(result, "durations_in_seconds", None) or {}
    inference_seconds = float(durations.get("prediction", 0.0) or 0.0)

    # Slice creation + postprocess matching are the closest representation of "slice and stitch".
    slice_seconds = float(durations.get("slice", 0.0) or 0.0)
    postprocess_seconds = float(durations.get("postprocess", 0.0) or 0.0)
    slice_merge_seconds = slice_seconds + postprocess_seconds

    # Fallback when SAHI does not expose detailed timings.
    if inference_seconds <= 0.0 and slice_merge_seconds <= 0.0:
        return max(total_seconds, 0.0), 0.0
    if slice_merge_seconds <= 0.0:
        slice_merge_seconds = max(total_seconds - inference_seconds, 0.0)
    if inference_seconds <= 0.0:
        inference_seconds = max(total_seconds - slice_merge_seconds, 0.0)

    return slice_merge_seconds, inference_seconds


def main() -> None:
    args = parse_args()

    try:
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
    except ImportError as exc:
        raise SystemExit("SAHI is not installed. Run: pip install sahi ultralytics") from exc

    source = Path(args.source).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(source)

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=args.model_path,
        confidence_threshold=args.confidence,
        device=args.device,
    )

    summary = []
    total_start = time.perf_counter()
    for image_path in images:
        image_start = time.perf_counter()
        result = get_sliced_prediction(
            str(image_path),
            detection_model,
            slice_height=args.slice_height,
            slice_width=args.slice_width,
            overlap_height_ratio=args.overlap_height_ratio,
            overlap_width_ratio=args.overlap_width_ratio,
            postprocess_type=args.postprocess_type,
            postprocess_match_threshold=args.postprocess_match_threshold,
            postprocess_match_metric=args.postprocess_match_metric,
        )

        stem = image_path.stem
        image_output_dir = output_dir / stem
        image_output_dir.mkdir(parents=True, exist_ok=True)

        predictions = [serialize_prediction(pred) for pred in result.object_prediction_list]
        visual_path = image_output_dir / f"{stem}_boxes.jpg"
        save_box_only_visual(image_path, predictions, visual_path)

        with (image_output_dir / f"{stem}_predictions.json").open("w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)

        elapsed_seconds = time.perf_counter() - image_start
        slice_merge_seconds, inference_seconds = extract_timing(result, elapsed_seconds)
        other_seconds = max(elapsed_seconds - slice_merge_seconds - inference_seconds, 0.0)
        summary.append(
            {
                "image": str(image_path),
                "num_predictions": len(predictions),
                "elapsed_seconds": round(elapsed_seconds, 6),
                "slice_merge_seconds": round(slice_merge_seconds, 6),
                "inference_seconds": round(inference_seconds, 6),
                "other_seconds": round(other_seconds, 6),
                "visual_dir": str(image_output_dir),
                "visual_image": str(visual_path),
                "prediction_json": str(image_output_dir / f"{stem}_predictions.json"),
            }
        )
        print(
            f"{image_path.name}: {len(predictions)} predictions, "
            f"slice+merge={slice_merge_seconds:.3f}s, "
            f"infer={inference_seconds:.3f}s, "
            f"other={other_seconds:.3f}s, "
            f"total={elapsed_seconds:.3f}s"
        )

    total_elapsed = time.perf_counter() - total_start
    total_slice_merge = sum(item["slice_merge_seconds"] for item in summary)
    total_inference = sum(item["inference_seconds"] for item in summary)
    avg_source = summary[1:] if len(summary) > 1 else summary
    avg_count = len(avg_source)
    avg_elapsed = sum(item["elapsed_seconds"] for item in avg_source) / avg_count if avg_count else 0.0
    avg_slice_merge = sum(item["slice_merge_seconds"] for item in avg_source) / avg_count if avg_count else 0.0
    avg_inference = sum(item["inference_seconds"] for item in avg_source) / avg_count if avg_count else 0.0
    avg_other = sum(item["other_seconds"] for item in avg_source) / avg_count if avg_count else 0.0
    payload = {
        "num_images": len(summary),
        "total_elapsed_seconds": round(total_elapsed, 6),
        "avg_elapsed_seconds": round(avg_elapsed, 6),
        "total_slice_merge_seconds": round(total_slice_merge, 6),
        "avg_slice_merge_seconds": round(avg_slice_merge, 6),
        "total_inference_seconds": round(total_inference, 6),
        "avg_inference_seconds": round(avg_inference, 6),
        "avg_other_seconds": round(avg_other, 6),
        "avg_excludes_first_image": len(summary) > 1,
        "results": summary,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Results saved to: {output_dir}")
    print(f"Total: {total_elapsed:.3f}s for {len(summary)} image(s), avg {payload['avg_elapsed_seconds']:.3f}s/image")
    print(
        f"Split avg: slice+merge={payload['avg_slice_merge_seconds']:.3f}s/image, "
        f"infer={payload['avg_inference_seconds']:.3f}s/image, "
        f"other={payload['avg_other_seconds']:.3f}s/image"
    )


if __name__ == "__main__":
    main()
