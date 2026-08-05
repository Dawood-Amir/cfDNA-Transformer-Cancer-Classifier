import os
import json
import torch
import numpy as np

from pathlib import Path
from tqdm import tqdm

from torch.utils.data import DataLoader, random_split
import torch.nn as nn

from sklearn.metrics import classification_report


from data.patient_dataset import PatientDataset
from data.patient_collate import PatientCollate

from models.tokenizer import CFDNATokenizer
from models.cfdna_transformer import CFDNATransformer


from utils.metrics import calculate_metrics
from utils.early_stopping import EarlyStopping
from utils.plotting import plot_training



# ============================================================
# CONFIG
# ============================================================


DATA_DIR = "src/data/processed/patient_tensors"


OUTPUT_DIR = "outputs"


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)



BATCH_SIZE = 2


EPOCHS = 50


LR = 1e-4


WEIGHT_DECAY = 0.01


CHUNK_SIZE = 64


EMBED_DIM = 128


NUM_CLASSES = 4



DEVICE = torch.device(

    "cuda"
    if torch.cuda.is_available()
    else "cpu"

)



# ============================================================
# DATA
# ============================================================


tokenizer = CFDNATokenizer()



dataset = PatientDataset(
    DATA_DIR
)



train_size = int(
    0.7 * len(dataset)
)


val_size = int(
    0.15 * len(dataset)
)


test_size = (
    len(dataset)
    -
    train_size
    -
    val_size
)



train_set, val_set, test_set = random_split(

    dataset,

    [
        train_size,
        val_size,
        test_size
    ],

    generator=torch.Generator().manual_seed(42)

)



collate_fn = PatientCollate(

    pad_token_id=
    tokenizer.vocab["<pad>"]

)



train_loader = DataLoader(

    train_set,

    batch_size=BATCH_SIZE,

    shuffle=True,

    collate_fn=collate_fn

)



val_loader = DataLoader(

    val_set,

    batch_size=BATCH_SIZE,

    shuffle=False,

    collate_fn=collate_fn

)



test_loader = DataLoader(

    test_set,

    batch_size=BATCH_SIZE,

    shuffle=False,

    collate_fn=collate_fn

)



print("="*70)

print("DATA")

print("="*70)

print(
    "Total:",
    len(dataset)
)

print(
    "Train:",
    len(train_set)
)

print(
    "Validation:",
    len(val_set)
)

print(
    "Test:",
    len(test_set)
)



# ============================================================
# MODEL
# ============================================================



model = CFDNATransformer(

    vocab_size=len(tokenizer.vocab),

    embed_dim=EMBED_DIM,

    num_heads=4,

    fragment_layers=2,

    patient_layers=2,

    num_classes=NUM_CLASSES

)



model.to(DEVICE)



criterion = nn.CrossEntropyLoss()



optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LR,

    weight_decay=WEIGHT_DECAY

)



scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="min",

    patience=3,

    factor=0.5

)



early_stopping = EarlyStopping(

    patience=10

)



# ============================================================
# HELPER FUNCTION
# ============================================================


def run_epoch(
        loader,
        training=True
):


    if training:

        model.train()

    else:

        model.eval()



    total_loss = 0


    all_labels=[]

    all_preds=[]

    all_probs=[]



    for batch in tqdm(loader):


        input_ids=batch["input_ids"]

        token_mask=batch["token_padding_mask"]

        fragment_mask=batch["fragment_padding_mask"]

        labels=batch["labels"]



        B,R,F,T=input_ids.shape



        flat_ids=input_ids.view(
            -1,T
        )


        flat_mask=token_mask.view(
            -1,T
        )


        flat_fragment_mask=fragment_mask.view(
            -1
        )



        real_ids=flat_ids[
            ~flat_fragment_mask
        ]


        real_masks=flat_mask[
            ~flat_fragment_mask
        ]



        # mappings

        patient_ids=[]

        region_ids=[]



        for b in range(B):

            for r in range(R):

                n=(~fragment_mask[b,r]).sum().item()


                if n>0:

                    patient_ids.extend(
                        [b]*n
                    )

                    region_ids.extend(
                        [r]*n
                    )



        patient_ids=torch.tensor(
            patient_ids,
            device=DEVICE
        )


        region_ids=torch.tensor(
            region_ids,
            device=DEVICE
        )



        region_mask=torch.ones(

            B,

            R,

            dtype=torch.bool,

            device=DEVICE

        )



        for b in range(B):

            for r in range(R):

                if (~fragment_mask[b,r]).sum()>0:

                    region_mask[b,r]=False



        labels=labels.to(DEVICE)



        real_ids=real_ids.to(DEVICE)

        real_masks=real_masks.to(DEVICE)



        fragment_embeddings=[]



        with torch.set_grad_enabled(training):


            for i in range(
                0,
                real_ids.size(0),
                CHUNK_SIZE
            ):


                emb=model.fragment_encoder(

                    real_ids[i:i+CHUNK_SIZE],

                    real_masks[i:i+CHUNK_SIZE]

                )


                fragment_embeddings.append(
                    emb
                )



            fragment_embeddings=torch.cat(
                fragment_embeddings
            )



            region_embeddings,mapping = model.region_pool(

                fragment_embeddings,

                patient_ids,

                region_ids

            )



            region_batch=torch.zeros(

                B,

                R,

                EMBED_DIM,

                device=DEVICE

            )



            for i in range(
                region_embeddings.size(0)
            ):


                region_batch[
                    mapping[i,0],
                    mapping[i,1]
                ] = region_embeddings[i]



            patient_embedding=model.patient_encoder(

                region_batch,

                region_mask

            )


            logits=model.classifier(
                patient_embedding
            )



            loss=criterion(

                logits,

                labels

            )



            if training:


                optimizer.zero_grad()


                loss.backward()


                torch.nn.utils.clip_grad_norm_(

                    model.parameters(),

                    1.0

                )


                optimizer.step()



        total_loss += loss.item()



        probs=torch.softmax(
            logits,
            dim=1
        )

        preds=torch.argmax(
            probs,
            dim=1
        )


        all_labels.extend(
            labels.cpu().numpy()
        )

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_probs.extend(
            probs.detach().cpu().numpy()
        )



    return (

        total_loss/len(loader),

        all_labels,

        all_preds,

        np.array(all_probs)

    )



# ============================================================
# TRAINING
# ============================================================


history={

    "train_loss":[],

    "val_loss":[]

}



best_loss=float("inf")



for epoch in range(EPOCHS):


    print()

    print(
        f"Epoch {epoch+1}/{EPOCHS}"
    )



    train_loss,_,_,_=run_epoch(

        train_loader,

        True

    )


    val_loss,_,_,_=run_epoch(

        val_loader,

        False

    )



    scheduler.step(
        val_loss
    )



    print(
        "Train loss:",
        train_loss
    )


    print(
        "Val loss:",
        val_loss
    )



    history["train_loss"].append(
        train_loss
    )

    history["val_loss"].append(
        val_loss
    )



    if val_loss < best_loss:


        best_loss=val_loss


        torch.save(

            model,

            f"{OUTPUT_DIR}/best_model.pt"

        )


        torch.save(

            model.state_dict(),

            f"{OUTPUT_DIR}/best_state_dict.pt"

        )


        torch.save(

            optimizer.state_dict(),

            f"{OUTPUT_DIR}/optimizer.pt"

        )


        torch.save(

            scheduler.state_dict(),

            f"{OUTPUT_DIR}/scheduler.pt"

        )


        print("Saved best model")



    if early_stopping(val_loss):

        break



# save history

with open(
    f"{OUTPUT_DIR}/training_history.json",
    "w"
) as f:

    json.dump(
        history,
        f
    )



plot_training(

    history["train_loss"],

    history["val_loss"],

    f"{OUTPUT_DIR}/training_curve.png"

)



# ============================================================
# TEST
# ============================================================


print("\nTESTING BEST MODEL")



model=torch.load(

    f"{OUTPUT_DIR}/best_model.pt",

    map_location=DEVICE

)



test_loss,labels,preds,probs=run_epoch(

    test_loader,

    False

)



metrics,cm=calculate_metrics(

    labels,

    preds,

    probs

)



with open(

    f"{OUTPUT_DIR}/final_metrics.json",

    "w"

) as f:

    json.dump(

        metrics,

        f,

        indent=4

    )


print("\nDONE")