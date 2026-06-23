import numpy as np
import tensorflow as tf

from src.preprocessing.vggPP import preprocess_vgg_image


class VggClothesDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, image_dir, label_map, batch_size=32, target_size=(128, 128), shuffle=True):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.label_map = label_map
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        x_batch = []
        y_batch = []

        for i in batch_indexes:
            row = self.df.iloc[i]
            image_path = f"{self.image_dir}/{row['id']}.jpg"
            x_batch.append(preprocess_vgg_image(image_path, target_size=self.target_size))
            y_batch.append(self.label_map[row["label_name"]])

        return np.array(x_batch, dtype=np.float32), tf.keras.utils.to_categorical(
            y_batch,
            num_classes=len(self.label_map),
        )

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


ClothesDataGenerator = VggClothesDataGenerator
