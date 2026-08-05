import torch


class PatientCollate:


    def __init__(
            self,
            pad_token_id
    ):

        self.pad_token_id = pad_token_id



    def __call__(self, batch):


        batch_size = len(batch)


        ############################################
        # Find maximum sizes
        ############################################

        max_regions = max(
            len(patient["regions"])
            for patient in batch
        )


        max_fragments = max(

            len(region["fragments"])

            for patient in batch

            for region in patient["regions"]

        )


        max_tokens = max(

            len(fragment["tokens"])

            for patient in batch

            for region in patient["regions"]

            for fragment in region["fragments"]

        )


        ############################################
        # Allocate tensors
        ############################################

        input_ids = torch.full(

            (
                batch_size,
                max_regions,
                max_fragments,
                max_tokens
            ),

            self.pad_token_id,

            dtype=torch.long

        )


        token_padding_mask = torch.ones(

            (
                batch_size,
                max_regions,
                max_fragments,
                max_tokens
            ),

            dtype=torch.bool

        )


        fragment_padding_mask = torch.ones(

            (
                batch_size,
                max_regions,
                max_fragments
            ),

            dtype=torch.bool

        )


        region_padding_mask = torch.ones(

            (
                batch_size,
                max_regions
            ),

            dtype=torch.bool

        )


        labels = torch.zeros(

            batch_size,

            dtype=torch.long

        )


        ############################################
        # New mappings
        ############################################

        fragment_patient_ids = []

        fragment_region_ids = []



        ############################################
        # Copy data
        ############################################


        for b, patient in enumerate(batch):


            labels[b] = patient["label"]



            for r, region in enumerate(patient["regions"]):


                region_padding_mask[b,r] = False



                for f, fragment in enumerate(
                    region["fragments"]
                ):


                    fragment_padding_mask[b,r,f] = False


                    tokens = fragment["tokens"]


                    length = len(tokens)


                    input_ids[
                        b,
                        r,
                        f,
                        :length
                    ] = tokens



                    token_padding_mask[
                        b,
                        r,
                        f,
                        :length
                    ] = False



                    ####################################
                    # Store ownership information
                    ####################################

                    fragment_patient_ids.append(b)

                    fragment_region_ids.append(r)



        ############################################
        # Convert mappings to tensors
        ############################################


        fragment_patient_ids = torch.tensor(

            fragment_patient_ids,

            dtype=torch.long

        )


        fragment_region_ids = torch.tensor(

            fragment_region_ids,

            dtype=torch.long

        )



        return {


            "input_ids": input_ids,


            "token_padding_mask":
                token_padding_mask,


            "fragment_padding_mask":
                fragment_padding_mask,


            "region_padding_mask":
                region_padding_mask,


            "fragment_patient_ids":
                fragment_patient_ids,


            "fragment_region_ids":
                fragment_region_ids,


            "labels": labels

        }