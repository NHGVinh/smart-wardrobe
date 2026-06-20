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

The plain pipeline intentionally imports the original model from `models/clothes_model.py`.

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
