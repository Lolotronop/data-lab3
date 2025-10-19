import os
import pathlib
import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers


SRC_TRAIN_DIR = "./ignored/data/training"
OUT_TRAIN_DIR = "./ignored/data/training"
IMG_SIZE = (224, 224)
AUG_PER_IMAGE = 1
SEED = 123

augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal", seed=SEED),
        layers.RandomRotation(0.08, seed=SEED + 1),
        layers.RandomZoom(0.08, seed=SEED + 2),
    ],
    name="offline_augmentation",
)


def load_and_prepare(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = arr / 255.0
    return arr


def save_image_from_array(arr, out_path):
    arr = np.clip(arr * 255.0, 0, 255).astype("uint8")
    Image.fromarray(arr).save(out_path, format="JPEG", quality=90)


def make_augmented_copies(src_dir, out_dir, aug_per_image=1):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)

    for class_name in sorted(os.listdir(src_dir)):
        class_src = os.path.join(src_dir, class_name)
        if not os.path.isdir(class_src):
            continue
        class_out = os.path.join(out_dir, class_name)
        os.makedirs(class_out, exist_ok=True)

        for fname in sorted(os.listdir(class_src)):
            src_path = os.path.join(class_src, fname)
            if not os.path.isfile(src_path):
                continue
            try:
                img_arr = load_and_prepare(src_path)
            except Exception as e:
                print("skip", src_path, ":", e)
                continue

            base_name = pathlib.Path(fname).stem
            ext = pathlib.Path(fname).suffix or ".jpg"

            # save_image_from_array(img_arr, os.path.join(class_out, fname))

            for i in range(aug_per_image):
                batch = np.expand_dims(img_arr, axis=0)
                aug_batch = augmentation(batch, training=True)
                aug_img = aug_batch[0].numpy()
                out_fname = f"aug_{i:02d}_{base_name}{ext}"
                out_path = os.path.join(class_out, out_fname)

                if os.path.exists(out_path):
                    unique = rng.integers(1_000_000)
                    out_fname = f"aug_{i:02d}_{base_name}_{unique}{ext}"
                    out_path = os.path.join(class_out, out_fname)

                save_image_from_array(aug_img, out_path)

        print("Processed class:", class_name)


if __name__ == "__main__":
    make_augmented_copies(SRC_TRAIN_DIR, OUT_TRAIN_DIR, AUG_PER_IMAGE)
    print("Augmentation complete.")
