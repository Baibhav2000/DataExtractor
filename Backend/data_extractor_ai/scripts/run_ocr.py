import os
import json
import uuid
from pathlib import Path
import pymupdf  # PyMuPDF for PDFs
from PIL import Image
import easyocr
import torch
from tqdm import tqdm

def process_documents_to_label_studio(input_folder, output_dir, languages=['en']):
    input_path = Path(input_folder)
    out_path = Path(output_dir)
    images_dir = out_path / "extracted_images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Automatically detect GPU availability
    use_gpu = torch.cuda.is_available()
    device_name = "GPU (CUDA)" if use_gpu else "CPU"
    print(f"[INFO] Initializing EasyOCR reader on {device_name}...")
    
    reader = easyocr.Reader(languages, gpu=use_gpu)
    
    pdf_exts = {'.pdf'}
    tiff_exts = {'.tif', '.tiff'}
    img_exts = {'.jpg', '.jpeg', '.png', '.webp'}
    supported_exts = pdf_exts | tiff_exts | img_exts
    
    if not input_path.exists():
        print(f"[ERROR] Input directory '{input_path}' does not exist.")
        return
        
    files = sorted([f for f in input_path.iterdir() if f.suffix.lower() in supported_exts])
    print(f"[INFO] Found {len(files)} document files to process.")
    
    tasks = []
    
    # Wrap file loop with tqdm
    for file_path in tqdm(files, desc="Processing Documents", unit="doc"):
        ext = file_path.suffix.lower()
        extracted_pages = []
        
        try:
            # 2. Handle PDFs (Extract all pages & force RGB to prevent OpenCV depth crashes)
            if ext in pdf_exts:
                doc = pymupdf.open(file_path)
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    pix = page.get_pixmap(dpi=150)
                    img_name = f"{file_path.stem}_page_{page_num + 1}.png"
                    img_save_path = images_dir / img_name
                    
                    # Convert PyMuPDF pixmap to PIL and normalize to RGB
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    img.save(str(img_save_path), "PNG")
                    extracted_pages.append(img_save_path)
                doc.close()
                
            # 3. Handle TIFFs (Extract all frames/pages & force RGB)
            elif ext in tiff_exts:
                with Image.open(file_path) as img:
                    frame_num = 0
                    while True:
                        img_name = f"{file_path.stem}_frame_{frame_num + 1}.png"
                        img_save_path = images_dir / img_name
                        
                        # Force conversion to RGB (handles 1-bit monochrome, RGBA, P, etc.)
                        rgb_img = img.convert("RGB")
                        rgb_img.save(str(img_save_path), "PNG")
                        extracted_pages.append(img_save_path)
                        
                        frame_num += 1
                        try:
                            img.seek(frame_num)
                        except EOFError:
                            break
                            
            # 4. Handle standard images (Single page document & force RGB)
            elif ext in img_exts:
                with Image.open(file_path) as img:
                    img_name = f"{file_path.name}"
                    img_save_path = images_dir / img_name
                    
                    rgb_img = img.convert("RGB")
                    rgb_img.save(str(img_save_path), "PNG")
                    extracted_pages.append(img_save_path)
                    
        except Exception as e:
            tqdm.write(f"[ERROR] Failed to process {file_path.name}: {e}")
            continue
            
        # Run OCR across all pages and link bounding boxes + transcribed text via region IDs
        result_list = []
        for page_idx, img_path in enumerate(extracted_pages):
            with Image.open(img_path) as pil_img:
                img_width, img_height = pil_img.size
                
            ocr_results = reader.readtext(str(img_path))
            
            for bbox, text, conf in ocr_results:
                x_min = min(bbox[0][0], bbox[3][0])
                y_min = min(bbox[0][1], bbox[1][1])
                x_max = max(bbox[1][0], bbox[2][0])
                y_max = max(bbox[2][1], bbox[3][1])
                
                box_w = x_max - x_min
                box_h = y_max - y_min
                
                x_pct = (x_min / img_width) * 100
                y_pct = (y_min / img_height) * 100
                w_pct = (box_w / img_width) * 100
                h_pct = (box_h / img_height) * 100
                
                # Generate a unique shared ID to bind the box and its text together
                region_id = str(uuid.uuid4())[:8]
                
                # 1. Bounding box prediction item
                box_result = {
                    "id": region_id,
                    "item_index": page_idx,  # Zero-indexed page reference
                    "from_name": "bbox",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "value": {
                        "x": round(x_pct, 2),
                        "y": round(y_pct, 2),
                        "width": round(w_pct, 2),
                        "height": round(h_pct, 2),
                        "rotation": 0,
                        "rectanglelabels": ["Text"]
                    }
                }
                
                # 2. Text transcription prediction item
                text_result = {
                    "id": region_id,
                    "item_index": page_idx,
                    "from_name": "transcription",
                    "to_name": "image",
                    "type": "textarea",
                    "value": {
                        "text": [text]
                    }
                }
                
                result_list.append(box_result)
                result_list.append(text_result)
                
        # Group all page URLs into a single document task
        page_urls = [f"/data/local-files/?d=intermediate/extracted_images/{p.name}" for p in extracted_pages]
        
        task = {
            "data": {
                "pages": page_urls
            },
            "predictions": [
                {
                    "model_version": "easyocr-v1",
                    "score": 0.85,
                    "result": result_list
                }
            ]
        }
        tasks.append(task)
        
    output_json_path = out_path / "tasks.json"
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)
        
    print(f"\n[SUCCESS] Generated {len(tasks)} grouped document tasks!")
    print(f"[SUCCESS] Saved tasks.json to: {output_json_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            description="Run OCR on raw dataset"
        )
        
    parser.add_argument(
            "--input", "-i", default="./data/raw",
            help="Path to the folder containing raw documents."
        )
    parser.add_argument(
        "--output", "-o", default="./data/intermediate",
        help="Path to output folder (or output JSON file when using 'fetch')."
    )
    parser.add_argument(
        "--langs", "-l", nargs="+", default=["en"],
        help="List of languages for EasyOCR (default: ['en'])."
    )
    args = parser.parse_args()
    process_documents_to_label_studio(args.input, args.output, languages=args.langs)

    