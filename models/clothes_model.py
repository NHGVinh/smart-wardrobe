import tensorflow as tf
import tensorflow.keras.layers as tfl

def build_clothes_model(input_shape=(128,128,3), num_classes = 4):

    """
    flow:
    preprocessed photo
    2 layers conv
    flatten
    dense
    softmax
    """

    model = tf.keras.Sequential([

        #conv1 học feature đơn giản, dùng 32 filter, kernel 3x3, activation relu
        tfl.Conv2D(32, (3, 3), padding  = 'same', input_shape = input_shape),
        tfl.ReLU(),
        tfl.MaxPooling2D((2, 2)),

        #conv2 học feature phức tạp hơn, dùng 64 filter, kernel 3x3, activation relu
        tfl.Conv2D(64, (3, 3), padding  = 'same', input_shape = input_shape),
        tfl.ReLU(),
        tfl.MaxPooling2D((2, 2)),

        tfl.Flatten(),
        
        tfl.Dense(128, activation = 'relu'),

        tfl.Dropout(0.3),
        
        tfl.Dense(num_classes, activation = 'softmax')
    ])

    return model