# Emodiversity

This repository will contain the data and code that was used in the creation of Video Cognitive Empathy (VCE) and Video to Valence (V2V) datasets. 
The work is currently in submission and will be made available soon.

## Video Cognitive Empathy (VCE) dataset
The VCE dataset contains 61,046 videos, each labelled for the intensity of 27 emotions. The dataset is split into
1. `train`: 50,000 videos
2. `test`: 11,046 videos

The dataset is structured as follows:

```bash
vce_dataset/
├── metadata.json
├── train_labels.json
├── test_labels.json
├── videos/00000.mp4
├── videos/00001.mp4
...
```
where
- `videos/` contains MP4 video files from both the train and test splits 
- `train_labels.json` and `test_labels.json` contain emotion labels corresponding to the videos. For example:
```
{
    "00000": {
        "emotions": { # This is the average rated "intensity" score (rated 0-1) for each emotion
            "Admiration": 0.4416666666666667,
            "Adoration": 0.23333333333333334,
            "Aesthetic Appreciation": 0.0,
            "Amusement": 0.06666666666666667,
            "Anger": 0.0,
            "Anxiety": 0.06666666666666667,
            "Awe (or Wonder)": 0.225,
            "Awkwardness": 0.0,
            "Boredom": 0.0,
            "Calmness": 0.0,
            "Confusion": 0.0,
            "Craving": 0.0,
            "Disgust": 0.0,
            "Empathic Pain": 0.0,
            "Entrancement": 0.0,
            "Excitement": 0.13333333333333333,
            "Fear": 0.0,
            "Horror": 0.0,
            "Interest": 0.125,
            "Joy": 0.08333333333333334,
            "Nostalgia": 0.0,
            "Relief": 0.0,
            "Romance": 0.0,
            "Sadness": 0.0,
            "Satisfaction": 0.14166666666666666,
            "Sexual Desire": 0.0,
            "Surprise": 0.20833333333333334
        },
        "file": "videos/00000.mp4",
        "topK": [ # The 3 highest-intensity emotions, from most intense to least intense
            "Admiration",
            "Adoration",
            "Awe (or Wonder)"
        ]
    },
    "00001": {
    ...
}
```
- `metadata.json` contains helpful metadata for all videos, e.g.
```
{
    "00000": {
        "codec_name": "h264",
        "duration": "12.000000",
        "file": "videos/00000.mp4",
        "frame_rate": "30/1",
        "height": 320,
        "number_of_frames": "360",
        "width": 256
    },
    "00001": {
    ...
}
```


## Video to Valence (V2V) Dataset
The V2V dataset consists preference annotations over a total of 26,670 videos, split into:
- `train`: 11,038 pairwise comparisons on 16,125 unique videos
- `test`: 3,189 pairwise comparisons on 6,223 unique videos
- `listwise`: 1,758 listwise ranked comparisons on 4,322 unique videos (up to 4 videos per list)

Note that `train`, `test`, `listwise` videos are mutually exclusive.

The dataset has the form:
```bash
v2v_dataset/
├── metadata.json
├── train_labels.json
├── test_labels.json
├── listwise_labels.json
├── videos/00000.mp4
├── videos/00001.mp4
...
```
where
- `videos/` contains 26,670 MP4 video files (this is a subset of the VCE dataset videos that have V2V labels)
- `metadata.json` contains helpful metadata for all videos (this is the same as the metadata file in the VCE dataset)
- `train_labels.json` and `test_labels.json` each contain a list of pairwise video comparisons, where the second video is more preferred:
```
{
    "comparisons": [
        ["52694", "15036"],
        ["49134", "56215"],
        ["34304", "31620"],
        ...
    ]
}
```
- `listwise_labels.json` contains a list of listwise video comparisons (ordered from least-preferred to most-preferred):
```
{
    "comparisons": [
        ["50986", "05956", "42507"],
        ["52542", "52733", "53157", "50334"],
        ["38647", "11277", "53157", "50334"],
        ["56616", "33536", "39234"],
        ...
    ]
}
```
