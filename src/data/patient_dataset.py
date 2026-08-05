import os
import torch 
import pandas as pd

from torch.utils.data import Dataset


import os
import torch
import pandas as pd

from torch.utils.data import Dataset
    

class PatientDataset(Dataset):

    def __init__(self, data_dir):

        self.data_dir = data_dir

        manifest_path = os.path.join(
            data_dir,
            "manifest.csv"
        )

        self.manifest = pd.read_csv(
            manifest_path
        )


    def __len__(self):

        return len(self.manifest)



    def __getitem__(self, idx):

        row = self.manifest.iloc[idx]


        patient = torch.load(
            os.path.join(
                self.data_dir,
                row["filename"]
            ),
            weights_only=False
        )


        return patient