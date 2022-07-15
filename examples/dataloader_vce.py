import argparse
import json
from pathlib import Path

import decord
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def get_n_evenly_spaced(arr, n):
    """
    get_n_evenly_spaced(lst=[0,1,2,3,4,5,6,7,8,9], n=3) -> [0,4,9]
    get_n_evenly_spaced(lst=[0,1,2], n=9) -> [0,0,0,1,1,1,2,2,2]
    """
    idx = np.round(np.linspace(0, len(arr) - 1, n)).astype(int)
    return list(np.array(arr)[idx])

def loadvideo_decord(fname, width, height, num_frames, frame_sample_rate=1):
    """
    Load video content using Decord
    Output: numpy array of shape (T H W C)
    """
    try:
        vr = decord.VideoReader(fname, width=width, height=height, num_threads=1, ctx=decord.cpu(0))
    except FileNotFoundError:
        print("video cannot be loaded by decord: ", fname)
        return np.zeros((1, height, width, 3))

    all_idxs = [x for x in range(0, len(vr), frame_sample_rate)]
    chosen_idxs = get_n_evenly_spaced(all_idxs, num_frames)
    buffer = vr.get_batch(chosen_idxs).asnumpy()
    return buffer

class VCEDataset(Dataset):

    def __init__(self, dataset_path, split, video_width, video_height, num_frames):
        assert split == "train" or split == "test"
        self.video_width = video_width
        self.video_height = video_height
        self.num_frames =  num_frames
        self.data_path = Path(dataset_path)
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
        video = loadvideo_decord(self.video_paths[idx], self.video_width, self.video_height, self.num_frames)
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


    dataset = VCEDataset(options.data_dir, split="train", video_width=320, video_height=256, num_frames=10)
    print("Length of dataset", len(dataset))

    def collate_batch(batch):
        videos, labels, metadata = [], [], []
        for (video, label, md) in batch:
            videos.append(video)
            labels.append(label)
            metadata.append(md)
        return videos, labels, metadata
    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0, collate_fn=collate_batch)

    for batch_idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        print("Batch", batch_idx)
        videos, labels, metadata = batch
        for vid, md, label in zip(videos, metadata, labels):
            print(vid.shape, md, " >> Label:", label)