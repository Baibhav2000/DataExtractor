import argparse
from scripts import (
    launch_label_studio, 
    process_documents_to_label_studio, 
    fetch_annotations,
    process_label_studio_export
)

def main():
    parser = argparse.ArgumentParser(
        description="OCR Dataset Preparation and Annotation CLI Pipeline for PaddleOCR & LayoutLMv3"
    )
    
    subparsers = parser.add_subparsers(
        dest="action", 
        required=True, 
        help="Specific pipeline action to execute."
    )

    # --- Subparser for 'run-server' ---
    subparsers.add_parser("run-server", help="Launch the Label Studio server.")

    # --- Subparser for 'run-ocr' ---
    parser_ocr = subparsers.add_parser("run-ocr", help="Run OCR on raw dataset and generate tasks.json.")
    parser_ocr.add_argument("--input", "-i", default="./data/raw", help="Path to raw documents.")
    parser_ocr.add_argument("--output", "-o", default="./data/intermediate", help="Path to intermediate output folder.")
    parser_ocr.add_argument("--langs", "-l", nargs="+", default=["en"], help="List of languages for EasyOCR.")

    # --- Subparser for 'all' ---
    parser_all = subparsers.add_parser("all", help="Run OCR pipeline followed by launching Label Studio.")
    parser_all.add_argument("--input", "-i", default="./data/raw")
    parser_all.add_argument("--output", "-o", default="./data/intermediate")
    parser_all.add_argument("--langs", "-l", nargs="+", default=["en"])

    # --- Subparser for 'fetch' ---
    parser_fetch = subparsers.add_parser("fetch", help="Fetch annotations from Label Studio.")
    parser_fetch.add_argument("--project-id", type=int, required=True, help="Target Label Studio Project ID.")
    parser_fetch.add_argument("--task-ids", type=int, nargs="+", default=None, help="Optional list of Task IDs.")
    parser_fetch.add_argument("--output", "-o", default="label_studio_export.json", help="Output JSON filename.")

    # --- Subparser for 'convert' ---
    parser_convert = subparsers.add_parser("convert", help="Convert Label Studio export to multi-model formats (PaddleOCR & LayoutLMv3).")
    parser_convert.add_argument("--input-file", "-i", default="label_studio_export.json", help="Path to Label Studio JSON export.")
    parser_convert.add_argument("--output-dir", "-o", default="./data/processed", help="Path to processed target folder.")
    parser_convert.add_argument("--images-dir", default="./data/intermediate/extracted_images", help="Path to raw source images.")
    parser_convert.add_argument("--compress", action="store_true", help="Package each processed model directory into a .zip archive.")

    args = parser.parse_args()

    # Routing actions
    if args.action == "run-server":
        launch_label_studio()
    elif args.action == "run-ocr":
        process_documents_to_label_studio(args.input, args.output, languages=args.langs)
    elif args.action == "all":
        print("\n--- Step 1: Running OCR and generating tasks.json ---")
        process_documents_to_label_studio(args.input, args.output, languages=args.langs)
        print("\n--- Step 2: Preprocessing complete. Launching interface ---")
        launch_label_studio()
    elif args.action == "fetch":
        print(f"\n--- Fetching Annotations for Project {args.project_id} ---")
        fetch_annotations(project_id=args.project_id, task_ids=args.task_ids, output=args.output)
    elif args.action == "convert":
        print(f"\n--- Converting annotations for PaddleOCR & LayoutLMv3 ---")
        process_label_studio_export(
            export_path=args.input_file, 
            output_dir=args.output_dir, 
            source_image_dir=args.images_dir,
            compress=args.compress
        )

if __name__ == "__main__":
    main()