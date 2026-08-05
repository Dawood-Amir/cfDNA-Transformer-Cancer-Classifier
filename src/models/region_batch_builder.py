import torch


class RegionBatchBuilder:


    def __init__(
        self,
        embed_dim=128
    ):

        self.embed_dim = embed_dim



    def __call__(
        self,
        region_embeddings,
        region_patient_ids,
        region_ids,
        batch_size
    ):


        """
        Convert flattened region embeddings into
        padded patient batches.

        Input:

        region_embeddings:
            [N_regions, embed_dim]


        region_patient_ids:
            [N_regions]

            Example:
            [0,0,0,1,1,1]


        region_ids:
            [N_regions]

            Example:
            [0,1,2,0,1,2]


        Output:

        patient_regions:

            [batch_size, max_regions, embed_dim]


        region_mask:

            [batch_size,max_regions]

            False = real region
            True  = padding

        """


        ############################################
        # Validation
        ############################################


        assert region_embeddings.dim() == 2, (
            "region_embeddings must be [N_regions,embed_dim]"
        )


        assert region_embeddings.size(1) == self.embed_dim, (
            "Embedding dimension mismatch"
        )


        assert (
            region_patient_ids.size(0)
            ==
            region_embeddings.size(0)
        ), "Patient mapping length mismatch"


        assert (
            region_ids.size(0)
            ==
            region_embeddings.size(0)
        ), "Region mapping length mismatch"



        device = region_embeddings.device



        ############################################
        # Find maximum regions in this batch
        ############################################


        regions_per_patient = torch.bincount(
            region_patient_ids,
            minlength=batch_size
        )


        max_regions = int(
            regions_per_patient.max().item()
        )



        ############################################
        # Allocate output
        ############################################


        patient_region_embeddings = torch.zeros(

            batch_size,

            max_regions,

            self.embed_dim,

            dtype=region_embeddings.dtype,

            device=device

        )



        ############################################
        # Padding mask
        #
        # True = padding
        ############################################


        region_padding_mask = torch.ones(

            batch_size,

            max_regions,

            dtype=torch.bool,

            device=device

        )



        ############################################
        # Scatter regions back to patients
        ############################################


        for idx in range(
            region_embeddings.size(0)
        ):


            patient = int(
                region_patient_ids[idx]
            )


            region = int(
                region_ids[idx]
            )


            patient_region_embeddings[
                patient,
                region
            ] = region_embeddings[idx]


            region_padding_mask[
                patient,
                region
            ] = False



        return (

            patient_region_embeddings,

            region_padding_mask

        )