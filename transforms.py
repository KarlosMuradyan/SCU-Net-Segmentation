import torch
import torch.nn as nn

class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, tracks):
        with torch.no_grad():
            vector = tracks[0]
            vector = torch.Tensor(vector)
            min_v = torch.min(vector)
            range_v = torch.max(vector) - min_v
            if range_v > 0:
                normalised = (vector - min_v) / range_v
            else:
                normalised = torch.zeros(vector.size())
            tracks.insert(0, normalised)
        return tracks

class Normalize_Mask(nn.Module):
    def __init__(self, minv, maxv):
        super(Normalize_Mask, self).__init__()
        self.min = minv
        self.max = maxv
        self.range = maxv-minv

    def forward(self, vector):
        with torch.no_grad():
            if self.range > 0:
                normalised = (vector - self.min) / self.range
            else:
                normalised = torch.zeros(vector.size())
        return normalised

    def back(self, mask):
        with torch.no_grad():
            if self.range > 0:
                unnormalised = mask*self.range + self.min
            else:
                unnormalised = torch.zeros(mask.size())
        return unnormalised

class HorizontalCrop(nn.Module):
    def __init__(self, crop_size):
        super(HorizontalCrop, self).__init__()
        self.crop_size = crop_size

    def forward(self, vector):
        processed_tracks = []
        with torch.no_grad():
            for track in vector:
                cropped_track = track[:, :self.crop_size]
                processed_tracks.append(cropped_track)
        return processed_tracks
