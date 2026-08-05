import torch
import torch.nn as nn

from models.fragment_transformer import FragmentTransformer
from models.region_pooling import RegionPooling
from models.region_batch_builder import RegionBatchBuilder
from models.patient_transformer import PatientTransformer
from models.classification_head import ClassificationHead



class CFDNATransformer(nn.Module):


    def __init__(
            self,
            vocab_size,
            embed_dim=128,
            num_heads=4,
            fragment_layers=2,
            patient_layers=2,
            num_classes=4
    ):

        super().__init__()


        self.fragment_encoder = FragmentTransformer(

            vocab_size=vocab_size,

            embed_dim=embed_dim,

            num_heads=num_heads,

            num_layers=fragment_layers

        )


        self.region_pooling = RegionPooling()


        self.region_builder = RegionBatchBuilder(

            embed_dim=embed_dim

        )


        self.patient_encoder = PatientTransformer(

            embed_dim=embed_dim,

            num_heads=num_heads,

            num_layers=patient_layers

        )


        self.classifier = ClassificationHead(

            embed_dim=embed_dim,

            num_classes=num_classes

        )



    ####################################################
    # Create fragment ownership mapping
    ####################################################

    def build_fragment_mapping(
            self,
            fragment_padding_mask
    ):


        B,R,F = fragment_padding_mask.shape


        patient_ids = []

        region_ids = []



        for b in range(B):

            for r in range(R):


                real_fragments = (
                    ~fragment_padding_mask[b,r]
                ).sum().item()



                if real_fragments > 0:


                    patient_ids.extend(

                        [b] * real_fragments

                    )


                    region_ids.extend(

                        [r] * real_fragments

                    )



        device = fragment_padding_mask.device


        patient_ids = torch.tensor(

            patient_ids,

            dtype=torch.long,

            device=device

        )


        region_ids = torch.tensor(

            region_ids,

            dtype=torch.long,

            device=device

        )


        return patient_ids, region_ids




    ####################################################
    # Forward
    ####################################################


    def forward(

            self,

            input_ids,

            token_padding_mask,

            fragment_padding_mask,

            region_padding_mask,

            show_progress=False

    ):


        B = input_ids.size(0)



        ################################################
        # Fragment Transformer
        ################################################


        fragment_embeddings = self.fragment_encoder(

            input_ids,

            token_padding_mask,

            fragment_padding_mask,

            chunk_size=256,

            show_progress=show_progress

        )



        ################################################
        # Create mapping
        ################################################


        fragment_patient_ids, fragment_region_ids = self.build_fragment_mapping(

            fragment_padding_mask

        )



        ################################################
        # Region pooling
        ################################################


        region_embeddings, region_mapping = self.region_pooling(

            fragment_embeddings,

            fragment_patient_ids,

            fragment_region_ids

        )



        ################################################
        # Region -> Patient batch
        ################################################


        region_batch, region_mask = self.region_builder(

            region_embeddings,

            region_mapping[:,0],

            region_mapping[:,1],

            B

        )



        ################################################
        # Patient Transformer
        ################################################


        patient_embeddings = self.patient_encoder(

            region_batch,

            region_mask

        )



        ################################################
        # Classification
        ################################################


        logits = self.classifier(

            patient_embeddings

        )


        return logits