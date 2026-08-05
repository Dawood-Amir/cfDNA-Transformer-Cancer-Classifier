import kagglehub
import os
import shutil

dataset_id = "kwarkp/model-5-cfdna-ctdna-dataset"

# Get project root (cfDNA_cancer_classifier/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dst = os.path.join(BASE_DIR, "data", "raw")
marker = os.path.join(dst, ".ready")

if os.path.exists(marker):
    print("Dataset already exists. Skipping download.")
else:
    src = kagglehub.dataset_download(dataset_id)

    os.makedirs(dst, exist_ok=True)

    shutil.copytree(src, dst, dirs_exist_ok=True)

    with open(marker, "w") as f:
        f.write("done")

    print("Dataset downloaded to:", dst)