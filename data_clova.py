import queue
import time

import h5py
import numpy as np
import torchvision.transforms
from PIL import ImageFont, ImageDraw, Image
import cv2
import os
import torch


def generate_text_img(text: str, font: ImageFont.FreeTypeFont, img_size):
    img_pil = Image.new('L', (img_size, img_size))
    draw_pil = ImageDraw.Draw(img_pil)
    draw_pil.text((0, 0), text, 255, font)
    img = np.array(img_pil, dtype=np.uint8)

    # rows = np.any(img > 0, axis=1)
    # cols = np.any(img > 0, axis=0)
    # rmin, rmax = np.where(rows)[0][[0, -1]]
    # cmin, cmax = np.where(cols)[0][[0, -1]]
    # img = img[rmin:rmax+1, cmin:cmax+1]
    bbox = font.getbbox(text)
    bbox = np.clip(bbox, 0, img_size)
    bbox_size = max(bbox[2], bbox[3])

    img = img[bbox[1]:bbox_size, bbox[0]:bbox_size]
    img = cv2.resize(img, (img_size, img_size))

    # print(bbox)
    # cv2.imshow('img', img)
    # cv2.waitKey()

    return img


def generate_data(font_paths, img_size=32):
    with h5py.File('data/char_imgs.h5', 'w') as h5_file:
        start = time.time()
        for j, (font_root, font_file) in enumerate(font_paths):
            font_path = os.path.join(font_root, font_file)
            font_name = font_file.replace('.ttf', '')
            print(font_name)
            font = ImageFont.truetype(font_path, img_size * 3 // 4)
            imgs = np.zeros((0xd7a3 - 0xac00 + 1, img_size, img_size), dtype=np.uint8)
            for i in range(0xd7a3 - 0xac00 + 1):
                c = chr(i + 0xac00)
                img = generate_text_img(c, font, img_size)
                imgs[i] = img
            #h5_file[font_name] = imgs
            h5_file.create_dataset(font_name, (0xd7a3 - 0xac00 + 1, img_size, img_size), 'u8', imgs, compression='gzip')
            print(time.time() - start)
            start = time.time()


class CharDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path='data/clova.h5'):
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


class CharColorDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path='data/clova.h5'):
        super(CharColorDataset, self).__init__()
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
        img = np.array(self.h5_file[self.fonts[font_i]][char_i], dtype=np.int16)
        color_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.int16)
        back_color = np.random.randint(0, 256, 3, np.int16)
        color_img[:, :] = back_color
        color_img[:, :, 0] = (color_img[:, :, 0] + img * np.random.randint(-back_color[0], 256 - back_color[0], 1, np.int16) // 255).astype(np.int16)
        color_img[:, :, 1] = (color_img[:, :, 0] + img * np.random.randint(-back_color[1], 256 - back_color[1], 1, np.int16) // 255).astype(np.int16)
        color_img[:, :, 2] = (color_img[:, :, 0] + img * np.random.randint(-back_color[2], 256 - back_color[2], 1, np.int16) // 255).astype(np.int16)
        color_img = self.to_tensor(color_img.astype(np.uint8))
        return color_img, init, medial, final, font_i

    def __len__(self):
        return len(self.fonts) * (0xd7a3 - 0xac00 + 1)


if __name__ == '__main__':
    font_paths = []
    for root, dir, files in os.walk('fonts/clova-all'):
        for file in files:
            if file[-4:] == '.ttf' and file not in {'나눔손글씨 동화또박.ttf', '나눔손글씨 하나손글씨.ttf'}:
                font_paths.append((root, file))

    generate_data(font_paths, 64)
    # dataset = CharColorDataset()
    # print(dataset.__getitem__(0))