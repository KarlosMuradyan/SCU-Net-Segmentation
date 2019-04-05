from torch.utils.data import Dataset
import librosa
import h5py
import random
from transforms import Normalize_Mask
import skimage
from icecream import ic
import numpy as np
import torch

class WaveDataset(Dataset):
    def __init__(self, dataframe, transforms=None):
        self.dataframe = dataframe
        self.transforms = transforms
        # self.norm = Normalize_Mask(minv=0, maxv=6)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        paths = self.dataframe.iloc[item, :].values
        mel_specs = []
        for i, p in enumerate(paths):
            # Read saved numpy arrays that correspond to the initial music
            with h5py.File(p, 'r') as hf:
                data = hf['dataset'][:]
            mlc, phase = librosa.magphase(data)
            # mlc = skimage.transform.resize(mlc, (512,512), anti_aliasing=True)
            # mlc = librosa.amplitude_to_db(mlc)
            #TODO find better way to handle even shape
            if mlc.shape[1]%2 == 0:
                mlc = mlc[:, :-1]
            mel_specs.append(mlc)

        # TODO add pipeline
        if self.transforms:
            for tr in self.transforms:
                mel_specs = tr(mel_specs)

        # Normalization!!
        # assert(len(mel_specs)>1)
        # true_mask = mel_specs[-1] / (mel_specs[-2] + 0.001)
        # true_mask[true_mask>6] = 6 - random.random()/2
        # true_mask = self.norm(true_mask)
        # mel_specs[-1] = true_mask

        assert(len(mel_specs)>1)
        true_mask = mel_specs[-1] / (mel_specs[-2] + 0.001)
        true_mask[true_mask>=0.5] = 1
        true_mask[true_mask<0.5] = 0
        true_mask = torch.Tensor(true_mask)
        mel_specs[-1] = true_mask
        

        return mel_specs
