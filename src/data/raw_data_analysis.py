import json
import numpy as np
import pandas as pd
from pathlib import Path
import os

BASE_DIR = "src/data/raw/synthetic_cfdna_output"

label_map = {
    "label_0_Healthy": 0,
    "label_1_GBM": 1,
    "label_2_LGG": 2,
    "label_3_DMG_H3K27M": 3
}

# Tracking arrays for global distribution metrics
all_region_token_counts = []
all_region_fragment_counts = []
patient_region_counts = []

patient_counter = 0

print("Starting deep dataset analysis...")

for folder in Path(BASE_DIR).iterdir():
    if not folder.is_dir():
        continue

    if folder.name not in label_map:
        continue
        
    label = label_map[folder.name]
    print(f"Analyzing class: {folder.name}...")

    for file in folder.glob("*.json"):
        with open(file) as f:
            data = json.load(f)
            
        patient_counter += 1
        
        # 'regions' is a dictionary {"region_0": {...}, "region_1": {...}}
        regions_dict = data.get("regions", {})
        patient_region_counts.append(len(regions_dict))
        
        # Iterate over the values of the dictionary to get the actual region dictionaries
        for region_id, region_data in regions_dict.items():
            fragments = region_data.get("fragments", [])
            all_region_fragment_counts.append(len(fragments))
            
            # Count the tokens across all fragments inside this single region
            total_tokens_in_region = 0
            for frag_dict in fragments:
                # frag_dict is a dictionary containing the key "tokens"
                tokens = frag_dict.get("tokens", [])
                total_tokens_in_region += len(tokens)
                
            all_region_token_counts.append(total_tokens_in_region)

# Convert to numpy array for fast conditional counting
token_counts_np = np.array(all_region_token_counts)
total_regions = len(token_counts_np)

print("\n" + "="*50)
print("             COMPREHENSIVE ANALYSIS RESULTS           ")
print("="*50)
print(f"Total Patients Tracked: {patient_counter}")
print(f"Total Individual Regions Quantified: {total_regions}")
print("-"*50)

# 1. Fragments per Region Metrics
print("FRAGMENTS PER REGION DISTRIBUTION:")
print(f"  - Average (Mean)   : {np.mean(all_region_fragment_counts):.2f} fragments")
print(f"  - 95th Percentile  : {int(np.percentile(all_region_fragment_counts, 95))} fragments")
print(f"  - 99th Percentile  : {int(np.percentile(all_region_fragment_counts, 99))} fragments")
print(f"  - Absolute Max     : {np.max(all_region_fragment_counts)} fragments")
print("-"*50)

# 2. Token Lengths per Region
print("FLATTENED TOKENS PER REGION (Includes all fragments combined):")
print(f"  - 50th Percentile (Median) : {int(np.percentile(all_region_token_counts, 50))} tokens")
print(f"  - 95th Percentile          : {int(np.percentile(all_region_token_counts, 95))} tokens")
print(f"  - 99th Percentile          : {int(np.percentile(all_region_token_counts, 99))} tokens")
print(f"  - Absolute Max             : {np.max(all_region_token_counts)} tokens")
print("-"*50)

# 3. Outlier Breakdown (Answers your specific questions)
print("OUTLIER REGIONS ANALYSIS:")
for limit in [1280, 1600, 2000, 2500 ,3225]:
    count = np.sum(token_counts_np > limit)
    percentage = (count / total_regions) * 100
    print(f"  - Regions exceeding {limit:4d} tokens: {count:5d} ({percentage:.4f}%)")
print("-"*50)

# 4. Regions per Patient Metrics
print("REGIONS PER PATIENT DISTRIBUTION:")
print(f"  - Average (Mean)   : {np.mean(patient_region_counts):.2f} regions")
print(f"  - 95th Percentile  : {int(np.percentile(patient_region_counts, 95))} regions")
print(f"  - Absolute Max     : {np.max(patient_region_counts)} regions")
print("="*50)

print("1. Look at '95th Percentile' for Flattened Tokens to pick your token pad window.")
print("2. Look at '95th Percentile' for Regions Per Patient to set your model configuration sequence caps.")