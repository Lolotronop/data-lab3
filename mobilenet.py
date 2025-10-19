import os
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
from dataload import load_image_datasets

import dvc.api

params = dvc.api.params_show()['mobilenet']

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 123
EPOCHS = 1
if params["epochs"] is not None:
    EPOCHS = params["epochs"]
if params["seed"] is not None:
    SEED = params["seed"]
train_ds, val_ds, eval_ds, class_names = load_image_datasets(
    img_size=IMG_SIZE, batch_size=BATCH_SIZE, seed=SEED
)

preprocess_input = keras.applications.mobilenet_v2.preprocess_input
base_model = keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
)
base_model.trainable = False

inputs = keras.Input(shape=IMG_SIZE + (3,))
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

loss, accuracy = model.evaluate(val_ds)
json.dump({"loss": loss, "accuracy": accuracy}, open("mobilenet.json", "w"))
model.save("./ignored/models/mobilenet.h5")
