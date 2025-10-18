import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
from dataload import load_image_datasets

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 123
EPOCHS = 1
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

print("Evaluation:", model.evaluate(eval_ds))
model.save("./ignored/models/mobilenetv2.h5")
with open("class_names.txt", "w") as f:
    f.write("\n".join(train_ds.class_names))
print("Saved model and class names:", train_ds.class_names)
