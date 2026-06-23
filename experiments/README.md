# Pipeline comparison

This project keeps the two AI pipelines separate so they can be compared by hand:

```text
VGG:
  preprocessing: src/preprocessing/vggPP.py
  model:         src/models/VGG.py
  training:      src/training/train_vgg.py
  prediction:    predict/predict_vgg.py
  utilities:     utils/vgg/

ResNet:
  preprocessing: src/preprocessing/resnetPP.py
  model:         src/models/resnet.py
  training:      src/training/train_resnet.py
  prediction:    predict/predict_resnet.py
  utilities:     utils/resnet/
```

Trained weights are written under:

```text
weights/vgg/vgg_model.h5
weights/vgg/label_map.json
weights/resnet/resnet_model.pth
weights/resnet/labels_map.pth
```

Run training:

```powershell
python src\training\train_vgg.py
python src\training\train_resnet.py
```

Run prediction:

```powershell
python predict\predict_vgg.py test_images\test_pants_1.png
python predict\predict_resnet.py test_images\test_pants_1.png
```

Run ResNet prediction with KNN outfit suggestion:

```powershell
python predict\predict_resnet.py test_images\test_pants_1.png --wardrobe-csv wardrobe.csv
```

`wardrobe.csv` must contain:

```text
image_path,category,style
```

Compare completed runs:

```powershell
python experiments\compare_pipelines.py
```

The comparison script expects `metrics.json` files from completed experiments. If the team is comparing by hand, this script can be left unused.
