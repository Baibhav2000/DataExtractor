import os
import json
import shutil
import random
from tqdm import tqdm 
from PIL import Image

def extract_doc_basename(webp_filename):
    """
    Extracts the root {basename} from your specific file structure:
    {basename}_page_{pageNumber}.webp
    """
    if "_page_" in webp_filename:
        return webp_filename.split("_page_")[0]
    return webp_filename

def process_label_studio_export(export_path, output_dir, source_image_dir="./data/intermediate/extracted_images", split_ratio=0.8, seed=42, compress=False):
    """
    Converts a Label Studio export JSON file into independent, model-specific dataset directories
    structured into train/ and val/ folders, with images in WebP format.
    """
    # Fix seed for reproducible splits
    random.seed(seed)
    
    with open(export_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    tasks = data if isinstance(data, list) else [data]
    
    # Define independent root directories for each model
    layoutlm_dir = os.path.join(output_dir, "layoutlmv3")
    paddle_det_dir = os.path.join(output_dir, "paddle_det")
    paddle_rec_dir = os.path.join(output_dir, "paddle_rec")
    
    splits = ["train", "val"]
    
    # Create train/val directories for each model
    for base_dir in [layoutlm_dir, paddle_det_dir, paddle_rec_dir]:
        for split in splits:
            os.path.join(base_dir, split, "images")
            if "paddle_rec" in base_dir:
                os.makedirs(os.path.join(base_dir, split, "word_crops"), exist_ok=True)
            else:
                os.makedirs(os.path.join(base_dir, split, "images"), exist_ok=True)
    
    # Dictionaries to group lines/objects by root document basename
    layoutlm_groups = {}
    det_groups = {}
    rec_groups = {}
    
    # Keep track of loaded images/crops temporarily during processing
    temp_tasks_data = []

    # Wrap tasks iterator with tqdm for progress tracking
    for task in tqdm(tasks, desc="Processing Tasks", unit="task"):
        task_id = task.get("task_id")
        data_field = task.get("data", {})
        raw_image_url = data_field.get("image", "") 
        
        annotations = task.get("annotations", [])
        if not annotations:
            continue
        
        result_items = annotations[0].get("result", [])
        
        items_map = {}
        for item in result_items:
            item_id = item.get("id")
            if item_id not in items_map:
                items_map[item_id] = {}
            
            val = item.get("value", {})
            if item.get("type") == "rectanglelabels":
                items_map[item_id]["box"] = val
                items_map[item_id]["label"] = val.get("rectanglelabels", ["O"])[0]
            elif item.get("type") == "textarea":
                items_map[item_id]["box"] = val
                text_list = val.get("text", [""])
                items_map[item_id]["text"] = text_list[0] if text_list else ""

        words = []
        bboxes = []
        ner_tags = []
        det_annots = []

        image_filename = os.path.basename(raw_image_url) if raw_image_url else f"task_{task_id}.png"
        file_stem, _ = os.path.splitext(image_filename)
        webp_filename = f"{file_stem}.webp"
        doc_basename = extract_doc_basename(webp_filename)

        local_source_image = os.path.join(source_image_dir, image_filename)
        img_loaded = None
        orig_w, orig_h = 0, 0
        if os.path.exists(local_source_image):
            img_loaded = Image.open(local_source_image).convert("RGB")
            orig_w, orig_h = img_loaded.size

        crop_data_list = []
        crop_idx = 0
        for item_id, content in items_map.items():
            if "box" in content and "text" in content:
                val = content["box"]
                x, y, w, h = val.get("x", 0), val.get("y", 0), val.get("width", 0), val.get("height", 0)
                text_val = content.get("text", "")
                label_val = content.get("label", "O")

                if orig_w > 0 and orig_h > 0:
                    x0_abs = (x / 100.0) * orig_w
                    y0_abs = (y / 100.0) * orig_h
                    w_abs = (w / 100.0) * orig_w
                    h_abs = (h / 100.0) * orig_h
                    x1_abs = x0_abs + w_abs
                    y1_abs = y0_abs + h_abs

                    points = [
                        [x0_abs, y0_abs],
                        [x1_abs, y0_abs],
                        [x1_abs, y1_abs],
                        [x0_abs, y1_abs]
                    ]
                    det_annots.append({"points": points, "transcription": text_val})

                    if text_val.strip() and img_loaded:
                        crop_filename = f"{file_stem}_word_{crop_idx}.webp"
                        crop_img = img_loaded.crop((x0_abs, y0_abs, x1_abs, y1_abs))
                        crop_data_list.append((crop_filename, crop_img))
                        rec_line = f"word_crops/{crop_filename}\t{text_val}\n"
                        if doc_basename not in rec_groups:
                            rec_groups[doc_basename] = []
                        rec_groups[doc_basename].append(rec_line)
                        crop_idx += 1

                x0 = int(round((x / 100.0) * 1000))
                y0 = int(round((y / 100.0) * 1000))
                x1 = int(round(((x + w) / 100.0) * 1000))
                y1 = int(round(((y + h) / 100.0) * 1000))
                x0, y0, x1, y1 = max(0, min(1000, x0)), max(0, min(1000, y0)), max(0, min(1000, x1)), max(0, min(1000, y1))
                
                words.append(text_val)
                bboxes.append([x0, y0, x1, y1])
                ner_tags.append(label_val)

        temp_tasks_data.append({
            "doc_basename": doc_basename,
            "webp_filename": webp_filename,
            "img_loaded": img_loaded,
            "det_annots": det_annots,
            "crop_data_list": crop_data_list,
            "layoutlm_task": {
                "id": str(task_id),
                "image_path": f"images/{webp_filename}",
                "tokens": words,
                "bboxes": bboxes,
                "ner_tags": ner_tags
            }
        })

    # --- SYNCHRONIZED SPLITTING LOGIC ---
    print("\nSplitting datasets deterministically based on root documents...")
    all_unique_docs = list(set(
        [t["doc_basename"] for t in temp_tasks_data]
    ))
    random.shuffle(all_unique_docs)
    
    split_idx = int(len(all_unique_docs) * split_ratio)
    train_docs = set(all_unique_docs[:split_idx])
    val_docs = set(all_unique_docs[split_idx:])

    print(f" -> Total unique documents: {len(all_unique_docs)}")
    print(f" -> Training bucket: {len(train_docs)} documents")
    print(f" -> Validation bucket: {len(val_docs)} documents")

    # Helper mapping for splits
    split_map_docs = {"train": train_docs, "val": val_docs}

    # Process and save items into respective train/val folders
    for item in tqdm(temp_tasks_data, desc="Saving Split Datasets", unit="file"):
        doc = item["doc_basename"]
        target_split = "train" if doc in train_docs else "val"
        
        img_loaded = item["img_loaded"]
        webp_filename = item["webp_filename"]
        
        if img_loaded:
            # Save to LayoutLMv3 split folder
            layoutlm_img_out = os.path.join(layoutlm_dir, target_split, "images", webp_filename)
            img_loaded.save(layoutlm_img_out, "WEBP", quality=90)
            
            # Save to Paddle Det split folder
            det_img_out = os.path.join(paddle_det_dir, target_split, "images", webp_filename)
            img_loaded.save(det_img_out, "WEBP", quality=90)

        # Save Paddle Rec crops into respective split folder
        for crop_filename, crop_img in item["crop_data_list"]:
            crop_out = os.path.join(paddle_rec_dir, target_split, "word_crops", crop_filename)
            crop_img.save(crop_out, "WEBP", quality=90)

        # Group annotations
        if item["det_annots"]:
            det_line = f"images/{webp_filename}\t{json.dumps(item['det_annots'], ensure_ascii=False)}\n"
            if doc not in det_groups:
                det_groups[doc] = {"train": [], "val": []}
            det_groups[doc][target_split].append(det_line)

        if doc not in layoutlm_groups:
            layoutlm_groups[doc] = {"train": [], "val": []}
        layoutlm_groups[doc][target_split].append(item["layoutlm_task"])

    # --- Write Annotation Files per Split ---
    for split in splits:
        # 1. LayoutLMv3
        split_layoutlm_data = [t for doc in split_map_docs[split] if doc in layoutlm_groups for t in layoutlm_groups[doc][split]]
        with open(os.path.join(layoutlm_dir, split, "annotations.json"), 'w', encoding='utf-8') as f:
            json.dump(split_layoutlm_data, f, indent=4, ensure_ascii=False)

        # 2. Paddle Det
        with open(os.path.join(paddle_det_dir, split, "det_gt.txt"), 'w', encoding='utf-8') as f:
            for doc in split_map_docs[split]:
                if doc in det_groups:
                    f.writelines(det_groups[doc][split])

        # 3. Paddle Rec
        with open(os.path.join(paddle_rec_dir, split, "rec_gt.txt"), 'w', encoding='utf-8') as f:
            for doc in split_map_docs[split]:
                if doc in rec_groups:
                    # Filter lines belonging to this split's documents
                    for line in rec_groups[doc]:
                        f.write(line)

    print(f"\nSuccessfully generated structured datasets in '{output_dir}':")
    for model_name, path in [("LayoutLMv3", layoutlm_dir), ("Paddle Det", paddle_det_dir), ("Paddle Rec", paddle_rec_dir)]:
        print(f" - [{model_name}] -> {path}/ ([train/val] / [images or word_crops] + annotation files)")

    if compress:
        print("\nPackaging datasets into ZIP archives...")
        for model_dir in [layoutlm_dir, paddle_det_dir, paddle_rec_dir]:
            zip_path = shutil.make_archive(model_dir, 'zip', model_dir)
            print(f" -> Archive created: {zip_path}")