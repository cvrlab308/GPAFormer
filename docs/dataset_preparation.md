# Dataset Preparation

Prepare the dataset locally and provide a MONAI Decathlon-style datalist JSON to the
training script.

## Expected Layout

One common layout is:

```text
/path/to/BTCV/
  imagesTr/
    img0001.nii.gz
    img0002.nii.gz
  labelsTr/
    label0001.nii.gz
    label0002.nii.gz
  dataset_fold0.json
```

The datalist file should contain paths relative to `data.root` in the config, or absolute paths:

```json
{
  "training": [
    {
      "image": "imagesTr/img0001.nii.gz",
      "label": "labelsTr/label0001.nii.gz"
    }
  ],
  "validation": [
    {
      "image": "imagesTr/img0002.nii.gz",
      "label": "labelsTr/label0002.nii.gz"
    }
  ]
}
```

Only the list selected by `data.list_key` is used by the included training script.


## Preprocessing

The default BTCV config uses:

- orientation: RAS
- spacing: `[1.5, 1.5, 2.0]`
- intensity window: `[-175, 250]`, scaled to `[0, 1]`
- foreground crop based on the image
- positive/negative random crops with ROI size `[96, 96, 96]`
- random flips, random 90-degree rotations, and random intensity shifts

Adjust `configs/btcv.yaml` for other datasets or label sets.
