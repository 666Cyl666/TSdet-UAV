from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import torch
import torchvision
from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sliding-window inference with hash-bucket local NMS.")
    parser.add_argument("--model-path", required=True, help="Path to YOLO weights, e.g. best.pt")
    parser.add_argument("--source", required=True, help="Image path or directory path")
    parser.add_argument("--output-dir", default="sw_hash_outputs", help="Directory to save results")
    parser.add_argument("--device", default="cpu", help="Inference device, e.g. cpu or cuda:0")
    parser.add_argument("--imgsz", type=int, default=640, help="Ultralytics predict imgsz")
    parser.add_argument("--confidence", type=float, default=0.01, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="IoU threshold for merge")
    parser.add_argument(
        "--window-size",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=(640, 640),
        help="Sliding window size in pixels",
    )
    parser.add_argument(
        "--step-percent",
        nargs=2,
        type=float,
        metavar=("WIDTH_PERCENT", "HEIGHT_PERCENT"),
        default=(90.0, 90.0),
        help="Sliding step as a percentage of window size",
    )
    parser.add_argument(
        "--merge-method",
        choices=("hash_nms", "nms", "nmw"),
        default="hash_nms",
        help="Prediction merge method",
    )
    parser.add_argument("--hash-cell-size", type=int, default=64, help="Spatial hash cell size in pixels")
    parser.add_argument(
        "--hash-neighbor-radius",
        type=int,
        default=1,
        help="Neighbor cell radius used by local NMS",
    )
    return parser.parse_args()


def collect_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Source not found: {source}")

    images = [path for path in sorted(source.iterdir()) if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    if not images:
        raise RuntimeError(f"No images found in directory: {source}")
    return images


def sliding_window(image, window_size, step_size):
    h, w, _ = image.shape
    windows = []
    for y in range(0, h, step_size[1]):
        for x in range(0, w, step_size[0]):
            y_end = min(y + window_size[1], h)
            x_end = min(x + window_size[0], w)
            window = image[y:y_end, x:x_end]
            window_height = y_end - y
            window_width = x_end - x
            windows.append((window, (x, y), (window_width, window_height)))
    return windows


def percent_to_step_size(window_size, step_percent):
    if len(window_size) != 2 or len(step_percent) != 2:
        raise ValueError("window_size and step_percent must both be length-2 tuples.")

    def _to_pixels(win_dim, percent):
        ratio = float(percent)
        if ratio <= 0:
            raise ValueError(f"step_percent must be > 0, got {percent}")
        if ratio <= 1:
            ratio *= 100.0
        pixels = int(round(win_dim * ratio / 100.0))
        return max(1, pixels)

    return (
        _to_pixels(window_size[0], step_percent[0]),
        _to_pixels(window_size[1], step_percent[1]),
    )


def _nmw_single_class(boxes, scores, iou_threshold=0.5):
    keep_boxes = []
    keep_scores = []
    order = torch.argsort(scores, descending=True)
    boxes = boxes[order]
    scores = scores[order]

    while boxes.shape[0] > 0:
        ref_box = boxes[0:1]
        ref_score = scores[0:1]
        if boxes.shape[0] == 1:
            keep_boxes.append(ref_box[0])
            keep_scores.append(ref_score[0])
            break

        ious = torchvision.ops.box_iou(ref_box, boxes).squeeze(0)
        merge_mask = ious >= iou_threshold
        merge_boxes = boxes[merge_mask]
        merge_scores = scores[merge_mask]

        weights = merge_scores.unsqueeze(1).clamp_min(1e-12)
        fused_box = (merge_boxes * weights).sum(dim=0) / weights.sum(dim=0)
        fused_score = merge_scores.max()
        keep_boxes.append(fused_box)
        keep_scores.append(fused_score)

        remain_mask = ~merge_mask
        boxes = boxes[remain_mask]
        scores = scores[remain_mask]

    return torch.stack(keep_boxes), torch.stack(keep_scores)


def global_merge_predictions(all_boxes_tensor, iou_threshold=0.5, merge_method="nms"):
    if all_boxes_tensor is None or all_boxes_tensor.shape[0] == 0:
        return torch.empty((0, 6), dtype=torch.float32)

    merged_chunks = []
    class_ids = all_boxes_tensor[:, 5].long().unique(sorted=True)
    for class_id in class_ids:
        cls_mask = all_boxes_tensor[:, 5].long() == class_id
        cls_boxes = all_boxes_tensor[cls_mask, :4]
        cls_scores = all_boxes_tensor[cls_mask, 4]

        if merge_method.lower() == "nmw":
            fused_boxes, fused_scores = _nmw_single_class(cls_boxes, cls_scores, iou_threshold=iou_threshold)
            fused_cls = torch.full((fused_boxes.shape[0], 1), float(class_id.item()), device=fused_boxes.device)
            merged = torch.cat([fused_boxes, fused_scores.unsqueeze(1), fused_cls], dim=1)
        else:
            keep = torchvision.ops.nms(cls_boxes, cls_scores, iou_threshold=iou_threshold)
            merged = all_boxes_tensor[cls_mask][keep]

        merged_chunks.append(merged)

    if not merged_chunks:
        return torch.empty((0, 6), dtype=torch.float32)
    merged = torch.cat(merged_chunks, dim=0)
    order = torch.argsort(merged[:, 4], descending=True)
    return merged[order]


class HashBucketLocalNMS:
    def __init__(self, iou_threshold=0.5, cell_size=64, neighbor_radius=1):
        if cell_size <= 0:
            raise ValueError("hash cell size must be positive.")
        if neighbor_radius < 0:
            raise ValueError("hash neighbor radius must be non-negative.")
        self.iou_threshold = float(iou_threshold)
        self.cell_size = int(cell_size)
        self.neighbor_radius = int(neighbor_radius)
        self.buckets: dict[tuple[int, int, int], list[torch.Tensor]] = defaultdict(list)
        self.num_raw_boxes = 0

    def _cell_of(self, box: torch.Tensor) -> tuple[int, int]:
        cx = float((box[0] + box[2]) * 0.5)
        cy = float((box[1] + box[3]) * 0.5)
        return int(cx // self.cell_size), int(cy // self.cell_size)

    def insert(self, boxes: torch.Tensor) -> None:
        if boxes is None or boxes.numel() == 0:
            return

        boxes = boxes.detach().cpu().float()
        for box in boxes:
            class_id = int(box[5].item())
            cell_x, cell_y = self._cell_of(box)
            self.buckets[(class_id, cell_x, cell_y)].append(box.clone())
            self.num_raw_boxes += 1

    def merge(self) -> torch.Tensor:
        if not self.buckets:
            return torch.empty((0, 6), dtype=torch.float32)

        merged_per_class = []
        class_ids = sorted({key[0] for key in self.buckets})
        for class_id in class_ids:
            cell_items = []
            coords = []
            for key, bucket in self.buckets.items():
                bucket_class, cell_x, cell_y = key
                if bucket_class != class_id or not bucket:
                    continue
                bucket_tensor = torch.stack(bucket)
                cell_items.append(bucket_tensor)
                coords.extend([(cell_x, cell_y)] * bucket_tensor.shape[0])

            if not cell_items:
                continue

            boxes = torch.cat(cell_items, dim=0)
            scores = boxes[:, 4]
            order = torch.argsort(scores, descending=True)
            boxes = boxes[order]
            ordered_coords = [coords[idx] for idx in order.tolist()]

            bucket_members: dict[tuple[int, int], list[int]] = defaultdict(list)
            for idx, coord in enumerate(ordered_coords):
                bucket_members[coord].append(idx)

            suppressed = torch.zeros(boxes.shape[0], dtype=torch.bool)
            kept = []
            for idx in range(boxes.shape[0]):
                if suppressed[idx]:
                    continue

                kept.append(boxes[idx])
                cell_x, cell_y = ordered_coords[idx]
                neighbor_indices = []
                for offset_x in range(-self.neighbor_radius, self.neighbor_radius + 1):
                    for offset_y in range(-self.neighbor_radius, self.neighbor_radius + 1):
                        neighbor_indices.extend(bucket_members.get((cell_x + offset_x, cell_y + offset_y), []))

                candidate_indices = [pos for pos in neighbor_indices if pos > idx and not suppressed[pos]]
                if not candidate_indices:
                    continue

                candidate_tensor = torch.tensor(candidate_indices, dtype=torch.long)
                ious = torchvision.ops.box_iou(boxes[idx : idx + 1, :4], boxes[candidate_tensor, :4]).squeeze(0)
                suppressed[candidate_tensor[ious >= self.iou_threshold]] = True

            if kept:
                merged_per_class.append(torch.stack(kept, dim=0))

        if not merged_per_class:
            return torch.empty((0, 6), dtype=torch.float32)

        merged = torch.cat(merged_per_class, dim=0)
        order = torch.argsort(merged[:, 4], descending=True)
        return merged[order]


def save_visual(image, predictions: list[dict], output_path: Path) -> None:
    draw_image = image.copy()
    for pred in predictions:
        x_min, y_min, x_max, y_max = pred["bbox_xyxy"]
        cv2.rectangle(draw_image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
    cv2.imwrite(str(output_path), draw_image)


def process_image(
    model,
    image_path: Path,
    output_dir: Path,
    win_size,
    step_percent,
    conf_thres=0.25,
    iou_thres=0.5,
    merge_method="hash_nms",
    device="cpu",
    imgsz=640,
    hash_cell_size=64,
    hash_neighbor_radius=1,
):
    start_total = time.perf_counter()
    timers = {"load": 0.0, "slice": 0.0, "prediction": 0.0, "postprocess": 0.0, "total": 0.0, "other": 0.0}
    image_output_dir = output_dir / image_path.stem
    image_output_dir.mkdir(parents=True, exist_ok=True)

    load_start = time.perf_counter()
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Error loading image {image_path}")
    timers["load"] = time.perf_counter() - load_start

    split_start = time.perf_counter()
    step_size = percent_to_step_size(win_size, step_percent)
    windows = sliding_window(image, win_size, step_size)
    timers["slice"] = time.perf_counter() - split_start

    all_slice_boxes = []
    hash_merger = None
    if merge_method == "hash_nms":
        hash_merger = HashBucketLocalNMS(
            iou_threshold=iou_thres,
            cell_size=hash_cell_size,
            neighbor_radius=hash_neighbor_radius,
        )

    raw_box_count = 0
    inf_start = time.perf_counter()
    for win, pos, _ in windows:
        results = model.predict(win, imgsz=imgsz, device=device, conf=conf_thres, verbose=False)
        batch_boxes = []
        for result in results:
            if len(result.boxes) == 0:
                continue
            boxes = result.boxes.xyxy.cpu()
            confs = result.boxes.conf.cpu().unsqueeze(1)
            clss = result.boxes.cls.cpu().unsqueeze(1)
            batch_boxes.append(torch.cat([boxes, confs, clss], dim=1))

        if not batch_boxes:
            continue

        global_boxes = torch.cat(batch_boxes, dim=0)
        global_boxes[:, 0] += pos[0]
        global_boxes[:, 1] += pos[1]
        global_boxes[:, 2] += pos[0]
        global_boxes[:, 3] += pos[1]
        raw_box_count += int(global_boxes.shape[0])

        if hash_merger is not None:
            hash_merger.insert(global_boxes)
        else:
            all_slice_boxes.append(global_boxes)
    timers["prediction"] = time.perf_counter() - inf_start

    postprocess_start = time.perf_counter()
    if hash_merger is not None:
        final_boxes = hash_merger.merge()
        hash_bucket_count = len(hash_merger.buckets)
    else:
        if all_slice_boxes:
            all_boxes_tensor = torch.cat(all_slice_boxes, dim=0)
        else:
            all_boxes_tensor = torch.empty((0, 6), dtype=torch.float32)
        final_boxes = global_merge_predictions(all_boxes_tensor, iou_threshold=iou_thres, merge_method=merge_method)
        hash_bucket_count = 0
    timers["postprocess"] = time.perf_counter() - postprocess_start

    predictions = []
    for box in final_boxes.cpu().numpy():
        x_min, y_min, x_max, y_max, conf, cls = box
        predictions.append(
            {
                "category_id": int(cls),
                "category_name": f"class_{int(cls)}",
                "score": float(conf),
                "bbox_xyxy": [float(x_min), float(y_min), float(x_max), float(y_max)],
            }
        )

    json_path = image_output_dir / f"{image_path.stem}_predictions.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    save_visual(image, predictions, image_output_dir / f"{image_path.stem}_result.jpg")

    timers["total"] = time.perf_counter() - start_total
    slice_merge = timers["slice"] + timers["postprocess"]
    timers["other"] = max(timers["total"] - slice_merge - timers["prediction"], 0.0)

    print(
        f"{image_path.name}: raw={raw_box_count}, final={len(predictions)}, "
        f"merge={merge_method}, step={step_size}, "
        f"slice+merge={slice_merge:.4f}s, infer={timers['prediction']:.4f}s, "
        f"other={timers['other']:.4f}s, total={timers['total']:.4f}s"
    )

    return {
        "image": str(image_path),
        "num_predictions": len(predictions),
        "num_raw_boxes": raw_box_count,
        "hash_bucket_count": hash_bucket_count,
        "merge_method": merge_method,
        "window_size": list(win_size),
        "step_percent": list(step_percent),
        "step_size": list(step_size),
        "prediction_json": str(json_path),
        "slice_merge_seconds": round(slice_merge, 6),
        "inference_seconds": round(timers["prediction"], 6),
        "other_seconds": round(timers["other"], 6),
        "elapsed_seconds": round(timers["total"], 6),
    }


def main() -> None:
    args = parse_args()
    source = Path(args.source).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(source)
    model = YOLO(args.model_path)

    summary = []
    total_start = time.perf_counter()
    for image_path in images:
        result = process_image(
            model=model,
            image_path=image_path,
            output_dir=output_dir,
            win_size=tuple(args.window_size),
            step_percent=tuple(args.step_percent),
            conf_thres=args.confidence,
            iou_thres=args.iou_thres,
            merge_method=args.merge_method,
            device=args.device,
            imgsz=args.imgsz,
            hash_cell_size=args.hash_cell_size,
            hash_neighbor_radius=args.hash_neighbor_radius,
        )
        summary.append(result)

    total_elapsed = time.perf_counter() - total_start
    avg_source = summary[1:] if len(summary) > 1 else summary
    avg_count = len(avg_source)
    payload = {
        "num_images": len(summary),
        "merge_method": args.merge_method,
        "hash_cell_size": args.hash_cell_size,
        "hash_neighbor_radius": args.hash_neighbor_radius,
        "total_elapsed_seconds": round(total_elapsed, 6),
        "avg_elapsed_seconds": round(sum(item["elapsed_seconds"] for item in avg_source) / avg_count, 6) if avg_count else 0.0,
        "avg_slice_merge_seconds": round(sum(item["slice_merge_seconds"] for item in avg_source) / avg_count, 6) if avg_count else 0.0,
        "avg_inference_seconds": round(sum(item["inference_seconds"] for item in avg_source) / avg_count, 6) if avg_count else 0.0,
        "avg_other_seconds": round(sum(item["other_seconds"] for item in avg_source) / avg_count, 6) if avg_count else 0.0,
        "avg_excludes_first_image": len(summary) > 1,
        "results": summary,
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Results saved to: {output_dir}")
    print(
        f"Average(exclude first image if possible): "
        f"slice+merge={payload['avg_slice_merge_seconds']:.4f}s, "
        f"infer={payload['avg_inference_seconds']:.4f}s, "
        f"other={payload['avg_other_seconds']:.4f}s, "
        f"total={payload['avg_elapsed_seconds']:.4f}s"
    )


if __name__ == "__main__":
    main()
