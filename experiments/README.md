# Pipeline comparison

This project is organized by pipeline layers:

```text
src/
  preprocessing/
    plainPP.py
    resnetPP.py
  data/
    splits.py
    plain_dataset.py
    resnet_dataset.py
  models/
    clothes_model.py
    resnet.py
  training/
    train_plain.py
    train_resnet.py
  inference/
    predict_resnet.py
```

The local dataset stays untouched in:

- `styles.csv`
- `data/images/`

Run the plain TensorFlow pipeline:

```powershell
python src\training\train_plain.py
```

The plain/VGG training code lives in `src/training/train_vgg.py`.
`models/clothes_model.py` is only a thin compatibility wrapper pointing to the VGG code in `src/models/VGG.py`.

Run the ResNet18 PyTorch pipeline on the same local dataset:

```powershell
python src\training\train_resnet.py
```

If ImageNet weights cannot be downloaded, run:

```powershell
python src\training\train_resnet.py --weights none
```

Compare completed runs:

```powershell
python experiments\compare_pipelines.py
```

Outputs are written under `experiments/results/`.
The ResNet training script follows the GitHub training file closely; only dataset/output paths are synchronized for this project.
