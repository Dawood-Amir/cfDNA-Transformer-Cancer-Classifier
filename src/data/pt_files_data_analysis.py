#old structured data change it 
import torch
sample = torch.load("src/data/processed/patient_tensors/DMG_synthetic_0175.pt")

print(sample.keys())
print(len(sample["regions"]))
print(sample["regions"][0]["tokens"].shape)
print(sample["total_tokens"])