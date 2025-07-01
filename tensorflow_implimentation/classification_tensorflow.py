import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Detect GPU
device_name = tf.config.list_physical_devices('GPU')
print(f"Using device: {device_name if device_name else 'CPU'}")

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize to [-1, 1] like in your PyTorch code
x_train = (x_train / 255.0 - 0.5) / 0.5
x_test = (x_test / 255.0 - 0.5) / 0.5

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomCrop(32, 32)
])

# CNN Model
def build_model():
    model = models.Sequential([
        layers.Input(shape=(32, 32, 3)),

        data_augmentation,

        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    return model

model = build_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Training the model
history = model.fit(x_train, y_train,
                    epochs=30,
                    batch_size=128,
                    validation_data=(x_test, y_test),
                    shuffle=True)

# Save the model
model.save("deeper_cnn_cifar10_tf.h5")
print("Model saved as deeper_cnn_cifar10_tf.h5")
