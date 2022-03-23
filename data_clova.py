import h5py
import numpy as np
import torchvision.transforms
from PIL import ImageFont, ImageDraw, Image
import cv2
import os
import torch


def generate_random_character():
    return chr(0xac00 + np.random.choice(0xd7a3 - 0xac00 + 1))


def generate_text_img(text: str, font: ImageFont.FreeTypeFont):
    img_pil = Image.new('L', (font.size * 4 // 3 * len(text), font.size * 4 // 3))
    draw_pil = ImageDraw.Draw(img_pil)
    draw_pil.text((0, 0), text, (255), font)
    img = np.array(img_pil, dtype=np.uint8)

    rows = np.any(img>0, axis=1)
    cols = np.any(img>0, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    img = img[rmin:rmax, cmin:cmax]
    img = cv2.resize(img, (font.size * 4 // 3 * len(text), font.size * 4 // 3))

    return img


def generate_data():
    font_paths = []
    for root, dir, files in os.walk('fonts/clova-all'):
        for file in files:
            if file[-4:] == '.ttf':
                font_paths.append((root, file))


    with h5py.File('data/char_imgs.h5', 'w') as h5_file:
        for j, (font_root, font_file) in enumerate(font_paths):
            font_path = os.path.join(font_root, font_file)
            if font_path in {'fonts/clova-all\동화또박\나눔손글씨 동화또박.ttf', 'fonts/clova-all\하나손글씨\나눔손글씨 하나손글씨.ttf'}:
                continue
            font_name = font_file.replace('.ttf', '')
            print(font_name)
            font = ImageFont.truetype(font_path, 24)
            h5_file.create_dataset(font_name, (0xd7a3 - 0xac00 + 1, font.size * 4 // 3, font.size * 4 // 3), 'u8', chunks=True)
            for i in range(0xd7a3 - 0xac00 + 1):
                c = chr(i + 0xac00)
                img = generate_text_img(c, font)
                h5_file[font_name][i, :img.shape[0], :img.shape[1]] = img


class CharDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path='data/char_imgs.h5'):
        super(CharDataset, self).__init__()
        self.h5_path = h5_path
        self.h5_file = None
        with h5py.File(self.h5_path, 'r') as h5_file:
            self.fonts = list(h5_file.keys())
        self.font_indices = {f: i for i, f in enumerate(self.fonts)}
        self.to_tensor = torchvision.transforms.ToTensor()

    def __getitem__(self, index):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
        font_i = index // (0xd7a3 - 0xac00 + 1)
        char_i = index % (0xd7a3 - 0xac00 + 1)
        init = char_i // 588
        medial = (char_i % 588) // 28
        final = char_i % 28
        return self.to_tensor(np.array(self.h5_file[self.fonts[font_i]][char_i], dtype=np.uint8)), init, medial, final, font_i

    def __len__(self):
        return len(self.fonts) * (0xd7a3 - 0xac00 + 1)
