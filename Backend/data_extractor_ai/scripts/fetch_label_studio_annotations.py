import os
import json
from dotenv import load_dotenv
from label_studio_sdk import LabelStudio

load_dotenv()

LABEL_STUDIO_URL = "http://localhost:8080"
API_KEY = os.getenv("LABEL_STUDIO_API_KEY")

def fetch_annotations(project_id: int, task_ids: list = None, output: str = "label_studio_export.json"):
    """Fetches annotations for specified or all tasks in a project and saves them to JSON."""
    ls = LabelStudio(
        base_url=LABEL_STUDIO_URL,
        api_key=API_KEY
    )

    results = []

    if task_ids:
        print(f"Fetching annotations for specified tasks in Project {project_id}...")
        for task_id in task_ids:
            try:
                task = ls.tasks.get(id=task_id)
                annotations = getattr(task, 'annotations', []) or []
                data = getattr(task, 'data', {}) or {}

                results.append({
                    "task_id": task_id,
                    "project_id": project_id,
                    "annotation_count": len(annotations),
                    "annotations": annotations,
                    "data": data
                })
                print(f"✓ Task {task_id}: {len(annotations)} annotation(s) retrieved.")
            except Exception as e:
                print(f"✗ Task {task_id}: Failed to fetch ({e})")
    else:
        print(f"No task IDs specified. Fetching ALL tasks for Project {project_id}...")
        try:
            tasks = list(ls.tasks.list(project=project_id, fields="all"))
            print(f"Found {len(tasks)} task(s) in the project.")

            skipped_tasks_count = 0
            for task in tasks:
                task_id = getattr(task, 'id', None)
                annotations = getattr(task, 'annotations', []) or {}
                data = getattr(task, 'data', {}) or {}

                if not annotations:
                    print(f"⚠️ Task {task_id}: No annotations found, skipping.")
                    skipped_tasks_count += 1
                    continue

                results.append({
                    "task_id": task_id,
                    "project_id": project_id,
                    "annotation_count": len(annotations),
                    "annotations": annotations,
                    "data": data
                })
            print(f"✓ Successfully retrieved {len(results)} task(s) with annotations. {skipped_tasks_count} task(s) were skipped.")
        except Exception as e:
            print(f"✗ Failed to fetch project tasks ({e})")

    # Save to JSON file
    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False, default=str)

    print(f"\nSaved annotations to '{output}'.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch annotations for Label Studio projects.")
    parser.add_argument("--project-id", type=int, required=True, help="Target Project ID")
    parser.add_argument("--task-ids", type=int, nargs="+", default=None, help="Optional Task IDs")
    parser.add_argument("--output", type=str, default="label_studio_export.json", help="Output JSON filename")
    args = parser.parse_args()
    
    fetch_annotations(project_id=args.project_id, task_ids=args.task_ids, output=args.output)

