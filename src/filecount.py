#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility to count how many files share the same name (ignoring extension).
Useful for verifying duplicated annotations/images.

Author: asebyrm
"""

import os
from collections import Counter

def count_files_with_same_name(directory):
    # Extract base filenames (without extensions)
    file_names = [os.path.splitext(f)[0] for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    # Count occurrences
    file_count = Counter(file_names)

    # Filter duplicates
    duplicates = {name: count for name, count in file_count.items() if count > 1}
    return duplicates, len(file_names)

# Example usage
directory_path = "./data/raw/train"
same_name_files, total_files = count_files_with_same_name(directory_path)

print(f"[INFO] Total files (including duplicates): {total_files}")
if same_name_files:
    print("[DUPLICATES FOUND]")
    for name, count in same_name_files.items():
        print(f"  {name}: {count} files")
else:
    print("[OK] No duplicate file names found.")
