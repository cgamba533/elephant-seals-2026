import os
import random
import shutil

from PIL import Image, ImageEnhance
from sklearn.model_selection import train_test_split

# -----------------------------
# CONFIG
# -----------------------------

RANDOM_SEED = 42

SOURCE_DIR = "data/raw/All_Seals"
OUTPUT_DIR = "data/splits"

TRAIN_RATIO = 0.70
VALID_RATIO = 0.10
TEST_RATIO = 0.20

# augmentation intensity
TARGET_COUNTS = {
    "adult_male": 220,
    "adult_female": 250,
    "young_seal": 300
}

# -----------------------------
# SET SEED
# -----------------------------

random.seed(RANDOM_SEED)

# -----------------------------
# CREATE OUTPUT FOLDERS
# -----------------------------

splits = ["train", "valid", "test"]
classes = ["adult_male", "adult_female", "young_seal"]

for split in splits:
    for cls in classes:
        os.makedirs(
            os.path.join(OUTPUT_DIR, split, cls),
            exist_ok=True
        )

# -----------------------------
# AUGMENTATION FUNCTIONS
# -----------------------------

def horizontal_flip(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def vertical_flip(img):
    return img.transpose(Image.FLIP_TOP_BOTTOM)

def rotate_90(img):
    return img.rotate(90)

def rotate_minus_90(img):
    return img.rotate(-90)

def rotate_180(img):
    return img.rotate(180)

def brighten(img, factor=1.5):
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)

def darken(img, factor=0.5):
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)

def increase_contrast(img, factor=1.5):
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)

def decrease_contrast(img, factor=0.5):
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)

def crop_top(img):
    width, height = img.size
    return img.crop((0, height * 0.05, width, height))

def crop_bottom(img):
    width, height = img.size
    return img.crop((0, 0, width, height * 0.95))

AUGMENTATIONS = [
    ("hflip", horizontal_flip),
    ("vflip", vertical_flip),
    ("rot90", rotate_90),
    ("rotm90", rotate_minus_90),
    ("rot180", rotate_180),
    ("bright", brighten),
    ("dark", darken),
    ("contrast_up", increase_contrast),
    ("contrast_down", decrease_contrast),
    ("crop_top", crop_top),
    ("crop_bottom", crop_bottom),
]

# -----------------------------
# SPLIT ORIGINALS
# -----------------------------

for cls in classes:

    class_dir = os.path.join(SOURCE_DIR, cls)

    images = [
        f for f in os.listdir(class_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    train_files, temp_files = train_test_split(
        images,
        test_size=(1 - TRAIN_RATIO),
        random_state=RANDOM_SEED
    )

    valid_fraction_of_temp = TEST_RATIO / (VALID_RATIO + TEST_RATIO)
    
    valid_files, test_files = train_test_split(
        temp_files,
        test_size=valid_fraction_of_temp,
        random_state=RANDOM_SEED)

    split_map = {
        "train": train_files,
        "valid": valid_files,
        "test": test_files
    }

    # copy originals
    for split, files in split_map.items():

        for file_name in files:

            src = os.path.join(class_dir, file_name)

            dst = os.path.join(
                OUTPUT_DIR,
                split,
                cls,
                file_name
            )

            shutil.copy2(src, dst)

# -----------------------------
# AUGMENT TRAINING SET
# -----------------------------

print("\nGenerating augmentations...")

for cls in classes:

    train_class_dir = os.path.join(
        OUTPUT_DIR,
        "train",
        cls
    )

    current_images = [
        f for f in os.listdir(train_class_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    target_count = TARGET_COUNTS[cls]

    augment_index = 0

    while len(current_images) < target_count:

        original_file = random.choice(current_images)

        original_path = os.path.join(
            train_class_dir,
            original_file
        )

        try:
            img = Image.open(original_path).convert("RGB")

            aug_name, aug_func = random.choice(AUGMENTATIONS)

            augmented = aug_func(img)

            new_name = (
                f"aug_{augment_index}_{aug_name}_{original_file}"
            )

            save_path = os.path.join(
                train_class_dir,
                new_name
            )

            augmented.save(save_path)

            current_images.append(new_name)

            augment_index += 1

        except Exception as e:
            print(f"Error augmenting {original_file}: {e}")

# -----------------------------
# FINAL COUNTS
# -----------------------------

print("\nFinal dataset counts:")

for split in splits:

    print(f"\n{split.upper()}")

    for cls in classes:

        folder = os.path.join(
            OUTPUT_DIR,
            split,
            cls
        )

        count = len(os.listdir(folder))

        print(f"{cls}: {count}")