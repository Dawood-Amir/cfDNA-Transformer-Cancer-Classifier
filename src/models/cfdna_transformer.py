import torch
import torch.nn as nn

from models.fragment_transformer import FragmentTransformer
from models.region_pooling import RegionPooling
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



        ########################################
        # Fragment level encoder
        ########################################

        self.fragment_encoder = FragmentTransformer(

            vocab_size=vocab_size,

            embed_dim=embed_dim,

            num_heads=num_heads,

            num_layers=fragment_layers

        )



        ########################################
        # Fragment -> Region
        ########################################

        self.region_pool = RegionPooling()



        ########################################
        # Region -> Patient
        ########################################

        self.patient_encoder = PatientTransformer(

            embed_dim=embed_dim,

            num_heads=num_heads,

            num_layers=patient_layers

        )



        ########################################
        # Classification
        ########################################

        self.classifier = ClassificationHead(

            embed_dim=embed_dim,

            num_classes=num_classes

        )



    def forward(
            self,
            fragment_embeddings,
            region_patient_ids,
            region_ids,
            batch_size,
            region_padding_mask
    ):


        """
        fragment_embeddings:

            [N_fragments,128]

        region_patient_ids:

            [N_fragments]

            Which patient owns fragment


        region_ids:

            [N_fragments]

            Which region owns fragment


        region_padding_mask:

            [batch,max_regions]


        """



        ########################################
        # Fragment -> Region
        ########################################


        region_embeddings, region_mapping = self.region_pool(

            fragment_embeddings,

            region_patient_ids,

            region_ids

        )



        ########################################
        # Create patient region tensor
        ########################################


        max_regions = region_padding_mask.shape[1]


        region_batch = torch.zeros(

            batch_size,

            max_regions,

            fragment_embeddings.size(1),

            device=fragment_embeddings.device

        )


        patient_region_counter = torch.zeros(

            batch_size,

            dtype=torch.long,

            device=fragment_embeddings.device

        )



        for i in range(
            region_embeddings.size(0)
        ):


            patient = region_mapping[i,0]


            position = patient_region_counter[patient]


            region_batch[

                patient,

                position

            ] = region_embeddings[i]


            patient_region_counter[patient] += 1



        ########################################
        # Patient Transformer
        ########################################


        patient_embedding = self.patient_encoder(

            region_batch,

            region_padding_mask

        )



        ########################################
        # Classifier
        ########################################


        logits = self.classifier(

            patient_embedding

        )


        return logits