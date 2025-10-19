import os
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
from dataload import load_image_datasets

import dvc.api

params = dvc.api.params_show()

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 123
EPOCHS = 1
if params['cnn']["epochs"] is not None:
    EPOCHS = params['cnn']["epochs"]
if params["seed"] is not None:
    SEED = params["seed"]
train_ds, val_ds, eval_ds, class_names = load_image_datasets(
    img_size=IMG_SIZE, batch_size=BATCH_SIZE, seed=SEED
)

normalization = layers.Rescaling(scale=1.0 / 255)

inputs = keras.Input(shape=IMG_SIZE + (3,))
x = normalization(inputs)
x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D()(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

loss, accuracy = model.evaluate(val_ds)
json.dump({"loss": loss, "accuracy": accuracy}, open("cnn.json", "w"))
model.save("./ignored/models/cnn.h5")
