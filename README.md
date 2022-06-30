# Emodiversity

This repository will contain the data and code that was used in the creation of Video Cognitive Empathy (VCE) and Video to Valence (V2V) datasets. 
The work is currently in submission and will be made available soon.

## Video Cognitive Empathy (VCE) dataset
The VCE dataset is structured as follows:

```bash
vce_dataset/
├── metadata.json
├── train_labels.json
├── test_labels.json
├── train/00000.mp4
├── train/00001.mp4
...
├── train/49999.mp4
├── test/50000.mp4
├── test/50001.mp4
...
```
where
- `train/` and `test/` directories contain MP4 video files for the train and test splits (50,000 and 11,046 videos respectively)
- `train_labels.json` and `test_labels.json` contain labels for their respective videos like 
```json
{
    "00001": {
        "emotions": {
            "Admiration": 1.4166666666666667, # This is the average "intensity" score (rated 1-10) given by annotators who selected this emotion
            "Adoration": 0.0,
            "Aesthetic Appreciation": 5.583333333333333,
            "Amusement": 1.4166666666666667,
            "Anger": 0.0,
            "Anxiety": 0.0,
            "Awe (or Wonder)": 1.1666666666666667,
            "Awkwardness": 0.0,
            "Boredom": 0.0,
            "Calmness": 0.0,
            "Confusion": 0.0,
            "Craving": 0.8333333333333334,
            "Disgust": 0.0,
            "Empathic Pain": 0.0,
            "Entrancement": 0.0,
            "Excitement": 0.0,
            "Fear": 0.0,
            "Horror": 0.0,
            "Interest": 0.8333333333333334,
            "Joy": 0.0,
            "Nostalgia": 0.0,
            "Relief": 0.0,
            "Romance": 0.0,
            "Sadness": 0.0,
            "Satisfaction": 1.5,
            "Sexual Desire": 0.0,
            "Surprise": 0.0
        },
        "topK": [
            "Aesthetic Appreciation",
            "Satisfaction",
            "Admiration"
        ]
    },
    "00002": {
    ...
}
```
- `metadata.json` contains helpful metadata for all videos, e.g.
```json
{
    "00000": {
        "codec_name": "h264",
        "duration": "12.000000",
        "file": "train/00000.mp4",
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

The V2V dataset contains:
```bash
v2v_dataset/
├── metadata.json
├── train_labels.json
├── test_labels.json
├── train/00000.mp4
├── train/00001.mp4
...
├── train/49999.mp4
├── test/50000.mp4
├── test/50001.mp4
...
```
where everything has the same form as the VCE dataset, except that 
- `train_labels.json` and `test_labels.json` contain a list of preference-ordered comparisons (most-preferred to least-preferred):
```json
{
    "comparisons": [
        ["08711", "00842", "22249"],
        ["25894", "58217", "22029", "22249"],
        ["53147", "02989", "11888"],
        ["32206", "06875", "61492"],
        ["26382", "31415", "25377", "07105"],
        ...
    ]
}
```
- And `train/` and `test/` only contain a subset of the videos from VCE that have V2V labels.

## Citation
