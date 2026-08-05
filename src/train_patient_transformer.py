import os
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

from data.patient_dataset import PatientDataset
from data.patient_collate import PatientCollate

from models.tokenizer import CFDNATokenizer
from models.cfdna_transformer import CFDNATransformer



# ============================================================
# CONFIG
# ============================================================


DATA_DIR = "src/data/processed/patient_tensors"


SAVE_DIR = "outputs/checkpoints"

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)


BATCH_SIZE = 2

GRAD_ACCUMULATION = 4


EPOCHS = 1


LR = 1e-4


WEIGHT_DECAY = 0.01


PATIENCE = 7


NUM_CLASSES = 4



DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)



# ============================================================
# DATA
# ============================================================


print("="*70)
print("cfDNA TRANSFORMER TRAINING")
print("="*70)


tokenizer = CFDNATokenizer()


dataset = PatientDataset(
    DATA_DIR
)



train_size = int(
    0.8 * len(dataset)
)


val_size = len(dataset) - train_size



train_dataset, val_dataset = random_split(

    dataset,

    [
        train_size,
        val_size
    ],

    generator=torch.Generator().manual_seed(42)

)



collate = PatientCollate(

    tokenizer.vocab["<pad>"]

)



train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    collate_fn=collate,

    num_workers=0

)



val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    collate_fn=collate,

    num_workers=0

)



print()

print("Total patients:",len(dataset))

print("Train:",len(train_dataset))

print("Validation:",len(val_dataset))

print("Device:",DEVICE)




# ============================================================
# MODEL
# ============================================================


model = CFDNATransformer(

    vocab_size=len(tokenizer.vocab),

    embed_dim=128,

    num_heads=4,

    fragment_layers=2,

    patient_layers=2,

    num_classes=NUM_CLASSES

)


model.to(DEVICE)
#model = torch.compile(model)


print()

print("Model created")



# ============================================================
# OPTIMIZER
# ============================================================


optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LR,

    weight_decay=WEIGHT_DECAY

)



scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

    optimizer,

    T_max=EPOCHS

)



criterion = nn.CrossEntropyLoss()



# ============================================================
# TRAINING LOOP
# ============================================================


best_val_loss = float("inf")

patience_counter = 0



for epoch in range(EPOCHS):


    print("\n")
    print("="*70)
    print(
        f"EPOCH {epoch+1}/{EPOCHS}"
    )
    print("="*70)



    model.train()


    train_loss = 0


    optimizer.zero_grad()



    progress = tqdm(

        enumerate(train_loader),

        total=len(train_loader)

    )



    for step,batch in progress:


        input_ids = batch["input_ids"].to(DEVICE)

        token_mask = batch["token_padding_mask"].to(DEVICE)

        fragment_mask = batch["fragment_padding_mask"].to(DEVICE)

        region_mask = batch["region_padding_mask"].to(DEVICE)

        labels = batch["labels"].to(DEVICE)



        logits = model(

            input_ids,

            token_mask,

            fragment_mask,

            region_mask

        )



        loss = criterion(

            logits,

            labels

        )



        loss = loss / GRAD_ACCUMULATION


        loss.backward()



        if (
            (step+1)
            %
            GRAD_ACCUMULATION
            ==
            0
        ):


            torch.nn.utils.clip_grad_norm_(

                model.parameters(),

                1.0

            )


            optimizer.step()

            optimizer.zero_grad()



        train_loss += loss.item() * GRAD_ACCUMULATION



        progress.set_description(

            f"loss={loss.item()*GRAD_ACCUMULATION:.4f}"

        )



    train_loss /= len(train_loader)



    # ========================================================
    # VALIDATION
    # ========================================================


    model.eval()


    val_loss = 0


    preds=[]

    probs=[]

    targets=[]



    with torch.no_grad():


        for batch in tqdm(
            val_loader,
            desc="Validation"
        ):


            input_ids=batch["input_ids"].to(DEVICE)

            token_mask=batch["token_padding_mask"].to(DEVICE)

            fragment_mask=batch["fragment_padding_mask"].to(DEVICE)

            region_mask=batch["region_padding_mask"].to(DEVICE)

            labels=batch["labels"].to(DEVICE)



            logits=model(

                input_ids,

                token_mask,

                fragment_mask,

                region_mask

            )



            loss=criterion(

                logits,

                labels

            )


            val_loss += loss.item()



            probability=torch.softmax(

                logits,

                dim=1

            )



            preds.extend(

                torch.argmax(

                    probability,

                    dim=1

                ).cpu().numpy()

            )


            probs.extend(

                probability.cpu().numpy()

            )


            targets.extend(

                labels.cpu().numpy()

            )



    val_loss /= len(val_loader)



    accuracy = accuracy_score(

        targets,

        preds

    )



    try:

        auc = roc_auc_score(

            targets,

            probs,

            multi_class="ovr"

        )

    except:

        auc = 0



    print()

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Val Loss: {val_loss:.4f}"
    )

    print(
        f"Val Accuracy: {accuracy:.4f}"
    )

    print(
        f"Val AUC: {auc:.4f}"
    )



    scheduler.step()



    # ========================================================
    # SAVE BEST
    # ========================================================


    if val_loss < best_val_loss:


        best_val_loss = val_loss

        patience_counter = 0



        torch.save(

            {

            "model":
                model.state_dict(),

            "optimizer":
                optimizer.state_dict(),

            "epoch":
                epoch,

            },

            f"{SAVE_DIR}/best_model.pth"

        )


        print("Saved best model ✅")



    else:


        patience_counter += 1


        print(
            "Early stopping counter:",
            patience_counter
        )


        if patience_counter >= PATIENCE:

            print(
                "Early stopping triggered"
            )

            break



# ============================================================
# FINAL REPORT
# ============================================================


print("\n")
print("="*70)
print("FINAL VALIDATION RESULTS")
print("="*70)



print(
    confusion_matrix(
        targets,
        preds
    )
)


print(

    classification_report(

        targets,

        preds

    )

)


print("="*70)
print("TRAINING COMPLETE")
print("="*70)