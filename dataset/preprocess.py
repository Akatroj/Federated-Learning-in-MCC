import os
from pathlib import Path

import numpy as np
from PIL import Image

MAIN_DIR = Path(__file__).parent
ARCHIVE_DIR = MAIN_DIR.joinpath('archive', 'data')
RES_DIR = MAIN_DIR.joinpath('preprocessed')

TOTAL_IMAGES = 1200

assert not RES_DIR.exists(), f'{RES_DIR} exists, remove it first'

cycle_counter = 0
imgs_to_preprocess = []

while len(imgs_to_preprocess) < TOTAL_IMAGES:
    for author_images in os.listdir(ARCHIVE_DIR):
        if len(imgs_to_preprocess) == TOTAL_IMAGES:
            break
        path = ARCHIVE_DIR.joinpath(author_images)
        if path.is_dir():
            imgs = os.listdir(path)
            if len(imgs) > cycle_counter:
                imgs_to_preprocess.append(path.joinpath(imgs[cycle_counter]))
    cycle_counter += 1

os.mkdir(RES_DIR)

for i, img_path in enumerate(imgs_to_preprocess):
    if (i + 1) % 100 == 0:
        print(f'{i + 1}/{len(imgs_to_preprocess)}...')
    img = Image.open(img_path)
    shrink_ratio = np.clip(np.random.normal(2., 0.5, 1)[0], 1.25, 2.75)
    resized = img.resize((int(img.width / shrink_ratio), int(img.height / shrink_ratio))) 
    res_path = RES_DIR.joinpath(f'img_{i}.jpg')
    resized.save(res_path)

print(f'saved {len(imgs_to_preprocess)} images in {RES_DIR}')
