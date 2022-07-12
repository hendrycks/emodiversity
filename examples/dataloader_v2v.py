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
        if width and height:
            vr = decord.VideoReader(fname, width=width, height=height, num_threads=1, ctx=decord.cpu(0))
        else:
            vr = decord.VideoReader(fname, num_threads=1, ctx=decord.cpu(0))
    except FileNotFoundError:
        print("video cannot be loaded by decord: ", fname)
        return np.zeros((1, height, width, 3))

    all_idxs = [x for x in range(0, len(vr), frame_sample_rate)]
    chosen_idxs = get_n_evenly_spaced(all_idxs, num_frames)
    buffer = vr.get_batch(chosen_idxs).asnumpy()
    return buffer

class V2VPairwiseDataset(Dataset):

    def __init__(self, dataset_path, split, video_width, video_height, num_frames):
        assert split == "train" or split == "test"
        self.video_width = video_width
        self.video_height = video_height
        self.num_frames =  num_frames
        self.data_path = Path(dataset_path)
        assert self.data_path.is_dir()

        with open(self.data_path / "metadata.json") as f:
            metadata_dict = json.load(f)
        
        self.video_paths_1 = []
        self.video_paths_2 = []
        self.metadata_1 = []
        self.metadata_2 = []
        with open(self.data_path / f"{split}_labels.json") as f:
            labels_dict = json.load(f)
            for comparison_list in labels_dict["comparisons"]:
                video_id_1, video_id_2 = comparison_list
                path_1 = str(self.data_path / f"videos/{video_id_1}.mp4")
                path_2 = str(self.data_path / f"videos/{video_id_2}.mp4")

                self.video_paths_1.append(path_1)
                self.video_paths_2.append(path_2)
                self.metadata_1.append(metadata_dict[video_id_1])
                self.metadata_2.append(metadata_dict[video_id_2])

        assert len(self.video_paths_1) == len(self.metadata_1)
        assert len(self.video_paths_2) == len(self.metadata_2)
        assert len(self.video_paths_1) == len(self.video_paths_2)

    def __len__(self):
        return len(self.video_paths_1)

    def __getitem__(self, idx):
        video_1 = loadvideo_decord(self.video_paths_1[idx], self.video_width, self.video_height, self.num_frames)
        video_2 = loadvideo_decord(self.video_paths_2[idx], self.video_width, self.video_height, self.num_frames)
        md_1 = self.metadata_1[idx]
        md_2 = self.metadata_2[idx]
        return video_1, video_2, md_1, md_2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Dataset and DataLoader for V2V dataset')
    parser.add_argument("-d", "--data-dir", required=True, help="Dataset directory.")
    options = parser.parse_args()

    
    full_dataset = V2VPairwiseDataset(options.data_dir, split="train", video_width=320, video_height=256, num_frames=10)
    print("Length of dataset", len(full_dataset))
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])


    def collate_batch(batch):
        videos_1, videos_2, metadata_1, metadata_2 = [], [], [], []
        for (video_1, video_2, md_1, md_2) in batch:
            videos_1.append(video_1)
            videos_2.append(video_2)
            metadata_1.append(md_1)
            metadata_2.append(md_2)
        return videos_1, videos_2, metadata_1, metadata_2

    batch_size = 64
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch)

    for batch_idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        print("Batch", batch_idx)
        videos_1, videos_2, metadata_1, metadata_2 = batch
        for vid_1, vid_2, md_1, md_2 in zip(videos_1, videos_2, metadata_1, metadata_2):
            print("vid 1:", vid_1.shape, md_1)
            print("vid 2:", vid_2.shape, md_2)