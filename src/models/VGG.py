import tensorflow as tf
import tensorflow.keras.layers as tfl

def build_clothes_model(input_shape=(128,128,3), num_classes=4):
    """
    flow:
    preprocessed photo
    vgg16: 13 lớp conv + 3 lớp fully connected
    flatten
    dense
    softmax
    """
    model = tf.keras.Sequential([
        # Khối 1: Trích xuất đặc trưng bậc 64 filter: ảnh 128x128 --> 64x64
        tfl.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        tfl.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tfl.MaxPooling2D((2, 2), strides=(2, 2)),

        # Khối 2: Trích xuất đặc trưng bậc 128 filter: ảnh 64x64 --> 32x32
        tfl.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tfl.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tfl.MaxPooling2D((2, 2), strides=(2, 2)),
    
        # Khối 3: Trích xuất đặc trưng bậc 256 filter: ảnh 32x32 --> 16x16
        tfl.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tfl.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tfl.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tfl.MaxPooling2D((2, 2), strides=(2, 2)),
    
        # Khối 4: Trích xuất đặc trưng bậc 512 filter: ảnh 16x16 --> 8x8
        tfl.Conv2D(512, (3, 3), activation='relu', padding='same'),
        tfl.Conv2D(512, (3, 3), activation='relu', padding='same'),
        tfl.Conv2D(512, (3, 3), activation='relu', padding='same'),
        tfl.MaxPooling2D((2, 2), strides=(2, 2)),          
    
        # Khối 5: Trích xuất đặc trưng bậc 512 filter: ảnh 8x8 --> 4x4
        tfl.Conv2D(512, (3, 3), activation='relu', padding='same'),
        tfl.Conv2D(512, (3, 3), activation='relu', padding='same'),
        tfl.Conv2D(512, (3, 3), activation='relu', padding='same'),
        tfl.MaxPooling2D((2, 2), strides=(2, 2)),       
    
        # flatten + dense + softmax
        tfl.Flatten(),
        tfl.Dense(512, activation='relu'), 
        tfl.Dropout(0.5), 
        tfl.Dense(512, activation='relu'),
        tfl.Dropout(0.5),
        tfl.Dense(num_classes, activation='softmax')
    ])
    
    return model