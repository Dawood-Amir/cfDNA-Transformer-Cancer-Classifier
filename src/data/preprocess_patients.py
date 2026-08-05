import os
import json
import torch
import pandas as pd

from pathlib import Path
from tqdm import tqdm
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.tokenizer import CFDNATokenizer


RAW_DIR = "src/data/raw/synthetic_cfdna_output"
SAVE_DIR = "src/data/processed/patient_tensors"

os.makedirs(SAVE_DIR, exist_ok=True)

LABEL_MAP = {

    "label_0_Healthy":0,
    "label_1_GBM":1,
    "label_2_LGG":2,
    "label_3_DMG_H3K27M":3

}

tokenizer = CFDNATokenizer()

manifest = []


###############################################################
# Iterate over classes
###############################################################

for class_folder in Path(RAW_DIR).iterdir():

    if not class_folder.is_dir():
        continue

    label = LABEL_MAP[class_folder.name]

    print(f"\nProcessing {class_folder.name}")

    ###########################################################

    for patient_file in tqdm(class_folder.glob("*.json")):

        with open(patient_file) as f:

            patient = json.load(f)

        patient_regions = []

        #######################################################
        # Preserve EVERY REGION
        #######################################################

        for region_name in sorted(patient["regions"].keys()):

            region = patient["regions"][region_name]

            fragments = []

            ###################################################
            # Preserve EVERY FRAGMENT
            ###################################################

            for fragment in region["fragments"]:

                encoded = tokenizer.encode(
                    fragment["tokens"]
                )

                fragments.append({

                    "tokens": encoded,

                    "fragment_length": fragment["fragment_length"],

                    "cpg_count": fragment["cpg_count"],

                    "strand": fragment["strand"]

                })

            ###################################################
            # Save region
            ###################################################

            patient_regions.append({

                "chromosome": region["chromosome"],

                "genomic_start": region["genomic_start"],

                "fragment_count": len(fragments),

                "fragments": fragments

            })

        #######################################################
        # Save patient
        #######################################################

        save_name = patient_file.stem + ".pt"

        torch.save(

            {

                "patient_id": patient["patient_id"],

                "label": label,

                "subtype": patient["subtype"],

                "raw_ctdna_fraction":
                    patient["raw_ctdna_fraction"],

                "enriched_ctdna_fraction":
                    patient["enriched_ctdna_fraction"],

                "n_fragments":
                    patient["n_fragments"],

                "regions": patient_regions

            },

            os.path.join(
                SAVE_DIR,
                save_name
            )

        )

        manifest.append({

            "filename": save_name,

            "label": label

        })


###############################################################
# Manifest
###############################################################

pd.DataFrame(manifest).to_csv(

    os.path.join(
        SAVE_DIR,
        "manifest.csv"
    ),

    index=False

)

print("\nFinished preprocessing.")