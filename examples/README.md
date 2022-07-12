This directory contains example dataloaders for training/evaluating on the VCE/V2V datasets.

Assuming you have downloaded the VCE and V2V datasets to `/data/v2v_dataset/` and `/data/vce_dataset/`, you can simply run the examples:
```
python dataset_v2v.py -d /data/v2v_dataset/
```
and
```
python dataset_v2v.py -d /data/v2v_dataset/
```

Pay attention to the Dataset class arguments `video_width`, `video_height`, `num_frames`, which will force each video into the exact width, height, and number of frames specified. (For a video with originally N frames, we will take `num_frames` evenly-spaced frames from the video)