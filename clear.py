import os
import shutil
from pathlib import Path

def clean_project(base_dir='.'):
    # WARNING: Set DRY_RUN to False ONLY when you are ready to actually delete the files.
    # When True, it will only print what it *would* delete.
    DRY_RUN = True 
    
    extensions_to_delete = {'.pkl', '.onnx', '.png', '.json'}
    deleted_files = 0
    deleted_folders = 0
    skipped_json_files = 0
    
    print(f"Starting cleanup in: {Path(base_dir).resolve()}")
    if DRY_RUN:
        print("--- DRY RUN MODE: No files will actually be deleted ---\n")

    # Walk through the directory from the bottom up
    for root, dirs, files in os.walk(base_dir, topdown=False):
        root_path = Path(root)
        
        # 1. Delete specified files
        for file_name in files:
            file_ext = Path(file_name).suffix.lower()
            
            if file_ext in extensions_to_delete:
                file_path = root_path / file_name
                
                # NEW RULE: Skip .json files if they are inside a .venv folder
                if file_ext == '.json' and '.venv' in root_path.parts:
                    if DRY_RUN:
                        print(f"Protected (Skipped): {file_path}")
                    skipped_json_files += 1
                    continue # Move to the next file without deleting

                # If it's not a protected .json, proceed with deletion
                if not DRY_RUN:
                    try:
                        file_path.unlink()
                        print(f"Deleted file: {file_path}")
                        deleted_files += 1
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")
                else:
                    print(f"Would delete file: {file_path}")
                    deleted_files += 1

        # 2. Delete __pycache__ folders
        if root_path.name == '__pycache__':
            if not DRY_RUN:
                try:
                    shutil.rmtree(root_path)
                    print(f"Deleted folder: {root_path}")
                    deleted_folders += 1
                except Exception as e:
                    print(f"Error deleting {root_path}: {e}")
            else:
                print(f"Would delete folder: {root_path}")
                deleted_folders += 1

    print("\n--- Summary ---")
    if DRY_RUN:
        print(f"Found {deleted_files} files and {deleted_folders} __pycache__ folders to delete.")
        print(f"Safely skipped {skipped_json_files} .json files inside .venv.")
        print("Change 'DRY_RUN = False' in the script to actually delete them.")
    else:
        print(f"Successfully deleted {deleted_files} files and {deleted_folders} __pycache__ folders.")
        print(f"Safely skipped {skipped_json_files} .json files inside .venv.")

if __name__ == '__main__':
    clean_project()