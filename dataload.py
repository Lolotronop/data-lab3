import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

base_dir = "./ignored/data"


def load_image_datasets(
    img_size=(224, 224),
    batch_size=32,
    seed=123,
    label_mode="binary",
    shuffle_train=True,
):
    train_dir = os.path.join(base_dir, "training")
    val_dir = os.path.join(base_dir, "validation")
    eval_dir = os.path.join(base_dir, "evaluation")

    train_ds = image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode=label_mode,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=shuffle_train,
        seed=seed,
    )
    val_ds = image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode=label_mode,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )
    eval_ds = image_dataset_from_directory(
        eval_dir,
        labels="inferred",
        label_mode=label_mode,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )

    return train_ds, val_ds, eval_ds, train_ds.class_names
