import cv2
import os
import json
import time
from ultralytics import YOLO
import torch
import torchvision

def _nmw_single_class(boxes, scores, iou_threshold=0.5):
    """Greedy Non-Maximum Weighted fusion for one class."""
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
    """Global merge for all slice predictions, class-wise."""
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
    return torch.cat(merged_chunks, dim=0)

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
    """Convert stride percentage to pixel stride from window size."""
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

# ####################### 淇敼鍚庣殑澶勭悊娴佺▼ (杈撳嚭 SAHI JSON 鏍煎紡) #######################
def process_image(model, image_path, output_folder, win_size, step_percent, conf_thres=0.25, iou_thres=0.5, merge_method="nms"):
    start_total = time.perf_counter()
    timers = {"load": 0.0, "slice": 0.0, "prediction": 0.0, "postprocess": 0.0, "total": 0.0, "other": 0.0}
    all_slice_boxes = []
    
    # 鍥惧儚鍔犺浇
    load_start = time.perf_counter()
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image {image_path}")
        return []
    timers["load"] = time.perf_counter() - load_start
    
    file_name = os.path.basename(image_path)
    base_name = os.path.splitext(file_name)[0]
    
    # 鍒涘缓鍥剧墖涓撳睘鐨勫瓙鏂囦欢澶?(SAHI 鏍煎紡瑕佹眰)
    image_output_dir = os.path.join(output_folder, base_name)
    os.makedirs(image_output_dir, exist_ok=True)
    
    # 鍥惧儚鍒囧垎
    split_start = time.perf_counter()
    step_size = percent_to_step_size(win_size, step_percent)
    windows = sliding_window(image, win_size, step_size)
    timers["slice"] = time.perf_counter() - split_start
    
    # 鎺ㄧ悊
    inf_start = time.perf_counter()
    for i, (win, pos, size) in enumerate(windows):
        results = model.predict(win, imgsz=640, device=0, conf=conf_thres, verbose=False)
        batch_boxes = []
        for result in results:
            if len(result.boxes) == 0: continue
            boxes = result.boxes.xyxy.cpu()
            confs = result.boxes.conf.cpu().unsqueeze(1)
            clss = result.boxes.cls.cpu().unsqueeze(1)
            batch_boxes.append(torch.cat([boxes, confs, clss], dim=1))
        
        if len(batch_boxes) > 0:
            global_boxes = torch.cat(batch_boxes, dim=0)
            global_boxes[:,0] += pos[0]  # x_min
            global_boxes[:,1] += pos[1]  # y_min
            global_boxes[:,2] += pos[0]  # x_max
            global_boxes[:,3] += pos[1]  # y_max
            
            all_slice_boxes.append(global_boxes)
    timers["prediction"] = time.perf_counter() - inf_start
    
    # 鍏ㄥ眬鍚堝苟锛堜笉鍐嶅垎妗讹級
    postprocess_start = time.perf_counter()
    if all_slice_boxes:
        all_boxes_tensor = torch.cat(all_slice_boxes, dim=0)
    else:
        all_boxes_tensor = torch.empty((0, 6), dtype=torch.float32)
    final_boxes = global_merge_predictions(all_boxes_tensor, iou_threshold=iou_thres, merge_method=merge_method)
    timers["postprocess"] = time.perf_counter() - postprocess_start
    
    # 杞崲涓?SAHI JSON 鏍煎紡
    sahi_predictions = []
    draw_image = image.copy()
    
    for box in final_boxes.cpu().numpy():
        x_min, y_min, x_max, y_max, conf, cls = box
        
        # 灏佽涓?SAHI 瑕佹眰鐨勫瓧鍏哥粨鏋?
        sahi_predictions.append({
            "category_id": int(cls),
            "category_name": f"class_{int(cls)}", # 璇勪及浠ｇ爜瀹归敊瀛楁
            "score": float(conf),
            "bbox_xyxy": [float(x_min), float(y_min), float(x_max), float(y_max)] # 娉ㄦ剰杩欓噷鏀瑰洖浜嗙粷瀵瑰潗鏍?xyxy
        })
        
        # 缁樺埗缁撴灉鐢ㄤ簬鍙鍖?
        cv2.rectangle(draw_image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0,255,0), 2)
        
    # 淇濆瓨璇ュ浘鐗囩殑 SAHI 鏍煎紡 JSON (渚嬪锛歰utput/DJI_0053/DJI_0053_predictions.json)
    json_path = os.path.join(image_output_dir, f"{base_name}_predictions.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(sahi_predictions, f, ensure_ascii=False, indent=2)
        
    # 淇濆瓨鍙鍖栧浘鐗?
    cv2.imwrite(os.path.join(image_output_dir, f"{base_name}_result.jpg"), draw_image)
    
    timers["total"] = time.perf_counter() - start_total
    slice_merge = timers["slice"] + timers["postprocess"]
    timers["other"] = max(timers["total"] - slice_merge - timers["prediction"], 0.0)
    print(
        f"Processed {file_name}: "
        f"step={step_size} ({step_percent[0]}%, {step_percent[1]}%), "
        f"slice+merge={slice_merge:.4f}s, "
        f"infer={timers['prediction']:.4f}s, "
        f"other={timers['other']:.4f}s, "
        f"total={timers['total']:.4f}s | Saved to {json_path}"
    )

    return {
        "predictions": sahi_predictions,
        "timers": timers,
        "slice_merge": slice_merge,
        "inference": timers["prediction"],
        "other": timers["other"],
        "total": timers["total"],
    }

def main():
    total_start = time.time()
    model_load_start = time.time()
    
    # 鍔犺浇妯″瀷
    #model = YOLO(r"D:\project\slicing_windows\new_slice_windows\ultralytics\ultralytics\cfg\models\12\yolo12.yaml")
    #model.load(r"D:\project\slicing_windows\new_slice_windows\ultralytics\runs\yolo12n\best.pt")

    print("Loading model...")
    model = YOLO(r"D:\project\slicing_windows\new_slice_windows\ultralytics\runs\yolo26n\best.pt")
    print(f"Model load time: {time.time()-model_load_start:.2f}s")
    
    # 璺緞閰嶇疆 (涓嶅啀闇€瑕?test.json)
    image_folder = r"D:\project\slicing_windows\new_slice_windows\test-data\images" # 浣犵殑娴嬭瘯鍘熷浘璺緞
    output_folder = r"D:\project\slicing_windows\new_slice_windows\output\sahi\yolo26n_sw" # 鐢熸垚鐨勪富鐩綍
    os.makedirs(output_folder, exist_ok=True)
    
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    print(f"Found {len(image_files)} images to process.")
    
    per_image_stats = []
    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(image_folder, img_file)
        print(f"[{idx+1}/{len(image_files)}] Processing {img_file}...")
        
        result = process_image(
            model=model,
            image_path=img_path,
            output_folder=output_folder,
            win_size=(640, 640),
            step_percent=(90, 90),
            conf_thres=0.01,  
            iou_thres=0.5,
            merge_method="nms"
        )
        per_image_stats.append(result)
    
    avg_source = per_image_stats[1:] if len(per_image_stats) > 1 else per_image_stats
    avg_count = len(avg_source)
    avg_slice_merge = sum(item["slice_merge"] for item in avg_source) / avg_count if avg_count else 0.0
    avg_inference = sum(item["inference"] for item in avg_source) / avg_count if avg_count else 0.0
    avg_other = sum(item["other"] for item in avg_source) / avg_count if avg_count else 0.0
    avg_total = sum(item["total"] for item in avg_source) / avg_count if avg_count else 0.0
    
    print(f"\nAll Done! Total processing time: {time.time()-total_start:.2f}s")
    print(
        "Average(exclude first image): "
        f"slice+merge={avg_slice_merge:.4f}s, "
        f"infer={avg_inference:.4f}s, "
        f"other={avg_other:.4f}s, "
        f"total={avg_total:.4f}s"
    )
    print(f"SAHI format predictions saved in: {output_folder}")

if __name__ == "__main__":
    main()
