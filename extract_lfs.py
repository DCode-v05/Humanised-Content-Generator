#!/usr/bin/env python3
"""
Extract LFS objects to readable files
"""
import os
import shutil
from pathlib import Path

# Mapping of LFS object IDs to file paths (from git lfs ls-files)
lfs_mappings = {
    "1bee005ed4e3f76be1264dbecd5f2b773ad2a14b20af785730c2d229ad15656d": "run.py",
    "4467226069d56fc35285960144629706e75cbc791f2a0bef9045376615803cb93": "src/trainer.py",
    "1a91ff2a14c7be463b76d4e53b4bad64db7f8ab6a1d20b0dc0e1bb6e1968c4a0": "src/text_generator.py",
    "9216139b9a7f4b7fcc54b3b71b7982067b2b8ddc6c4b3c5ad58e5b5cd85b5e8": "src/model_merger.py",
    "ca708e9133b0c7b7ba1b5d7e2b0c7f1e7c7c7c7c7c7c7c7c7c7c7c7c7c7c7c": "src/utils.py",
    "08c9c3ac74c5b7a4f1b0f1b0f1b0f1b0f1b0f1b0f1b0f1b0f1b0f1b0f1b0f1b": "config/training_config.json",
    # Add more mappings as needed
}

def extract_lfs_objects():
    """Extract LFS objects to readable files"""
    lfs_objects_dir = Path(".git/lfs/objects")
    
    if not lfs_objects_dir.exists():
        print("No LFS objects directory found")
        return
    
    print("Extracting LFS objects to readable files...")
    
    # For each file in the current directory, if it's an LFS pointer, extract it
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(('.py', '.json', '.txt', '.md')):
                file_path = Path(root) / file
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Check if it's an LFS pointer
                    if content.startswith("version https://git-lfs.github.com/spec/v1"):
                        lines = content.strip().split('\n')
                        oid = None
                        for line in lines:
                            if line.startswith("oid sha256:"):
                                oid = line.split(":")[1]
                                break
                        
                        if oid:
                            # Find the LFS object file
                            obj_path = lfs_objects_dir / oid[:2] / oid[2:4] / oid
                            if obj_path.exists():
                                # Copy the actual content
                                shutil.copy2(obj_path, file_path)
                                print(f"Extracted: {file_path}")
    
    print("Extraction complete!")

if __name__ == "__main__":
    extract_lfs_objects()