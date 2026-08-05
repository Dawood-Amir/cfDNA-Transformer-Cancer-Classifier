import torch
from torch.utils.data import DataLoader


from data.patient_dataset import PatientDataset
from data.patient_collate import PatientCollate

from models.tokenizer import CFDNATokenizer



DATA_DIR = "src/data/processed/patient_tensors"



tokenizer = CFDNATokenizer()


dataset = PatientDataset(
    DATA_DIR
)


collate = PatientCollate(
    pad_token_id=tokenizer.vocab["<pad>"]
)



loader = DataLoader(

    dataset,

    batch_size=2,

    shuffle=True,

    collate_fn=collate

)



batch = next(iter(loader))


print("="*60)

print("INPUT IDS")
print(batch["input_ids"].shape)


print()

print("TOKEN MASK")
print(batch["token_padding_mask"].shape)


print()

print("FRAGMENT MASK")
print(batch["fragment_padding_mask"].shape)


print()

print("REGION MASK")
print(batch["region_padding_mask"].shape)


print()

print("LABELS")
print(batch["labels"])


print("="*60)