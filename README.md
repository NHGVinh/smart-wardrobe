# Smart Wardrobe Demo

This project contains separate VGG and ResNet pipelines for clothes classification.
The ResNet18 demo UI also includes a local SQLite-backed user wardrobe.

## Local Wardrobe Database

The app creates `wardrobe.db` in the project root. It stores metadata for saved
wardrobe items:

- relative image path
- model name
- predicted category
- predicted style
- optional color
- confidence
- creation time

Uploaded images are copied to `my_closet/` with UUID filenames. Images are not
stored as binary data in SQLite.

## Run The ResNet18 Demo UI

Make sure the ResNet files exist:

```powershell
weights\resnet\resnet_model.pth
weights\resnet\labels_map.pth
```

Then run:

```powershell
python app_resnet_demo.py
```

Open:

```text
http://127.0.0.1:7860
```

## Use My Wardrobe

1. Open `Upload & Predict`.
2. Choose a sample image or upload an image.
3. Click `Run ResNet18`.
4. Review category, style, and confidence.
5. Click `Save to Wardrobe`.
6. Open `My Wardrobe` to view saved items.
7. Use `Delete` to remove an item from the database.

If an image file is missing, the UI shows a warning instead of crashing.

## Outfit Recommendation

Open `Outfit Recommendation`, choose a saved wardrobe item, and click
`Recommend`. The ResNet recommendation reads from `wardrobe.db` and only uses:

- Topwear
- Bottomwear
- Shoes
- Dress

The current ResNet workflow does not include Accessories, so recommendation does
not add accessory slots.
