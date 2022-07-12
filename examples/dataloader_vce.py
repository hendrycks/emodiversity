import argparse
import json
from pathlib import Path

import decord
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def loadvideo_decord(fname, width, height, frame_sample_rate=1):
    """
    Load video content using Decord
    Output: numpy array of shape (T H W C)
    """
    try:
        if width and height:
            vr = decord.VideoReader(fname, width=width, height=height, num_threads=1, ctx=decord.cpu(0))
        else:
            vr = decord.VideoReader(fname, num_threads=1, ctx=decord.cpu(0))
    except FileNotFoundError:
        print("video cannot be loaded by decord: ", fname)
        return np.zeros((1, height, width, 3))

    all_index = [x for x in range(0, len(vr), frame_sample_rate)]
    buffer = vr.get_batch(all_index).asnumpy()
    return buffer

class VCEDataset(Dataset):

    def __init__(self, vce_dataset_path, split="train", video_width=320, video_height=256):
        assert split == "train" or split == "test"
        self.video_width = video_width
        self.video_height = video_height
        self.data_path = Path(vce_dataset_path)
        assert self.data_path.is_dir()

        with open(self.data_path / "metadata.json") as f:
            metadata_dict = json.load(f)
        
        self.video_paths = []
        self.labels = []
        self.metadata = []
        with open(self.data_path / f"{split}_labels.json") as f:
            labels_dict = json.load(f)
            for key, obj in labels_dict.items():
                path, label = self.get_path_and_label(obj)
                self.video_paths.append(path)
                self.labels.append(label)
                self.metadata.append(metadata_dict[key])

        assert len(self.video_paths) == len(self.metadata)
        assert len(self.video_paths) == len(self.labels)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video = loadvideo_decord(self.video_paths[idx], self.video_width, self.video_height)
        label = self.labels[idx]
        md = self.metadata[idx]
        return video, label, md

    def get_path_and_label(self, label_obj):
        # Convert from 27-vector of emotion scores to a single classification label corresponding to the max-scoring emotion
        emotions_and_scores = sorted(list(label_obj["emotions"].items())) # Make sure they are sorted alphabetically
        scores = [score for emotion, score in emotions_and_scores]
        label = int(np.argmax(scores))
        path = str(self.data_path / label_obj['file'])
        return path, label


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Dataset and DataLoader for VCE dataset')
    parser.add_argument("-d", "--data-dir", required=True, help="Dataset directory.")
    options = parser.parse_args()


    full_dataset = VCEDataset(options.data_dir)
    print("Length of dataset", len(full_dataset))
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])


    def collate_batch(batch):
        videos, labels, metadata = [], [], []
        for (video, label, md) in batch:
            videos.append(video)
            labels.append(label)
            metadata.append(md)
        return videos, labels, metadata

    batch_size = 128
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch)

    for batch_idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        print("Batch", batch_idx)
        videos, labels, metadata = batch
        print("Labels:", labels)
        for vid, md in zip(videos, metadata):
            print(vid.shape)
            print(md)