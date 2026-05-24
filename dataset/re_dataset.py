import json
import os
import torch
from torch.utils.data import Dataset
from jieba import analyse
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption




class re_train_dataset(Dataset):

    def __init__(self, ann_file, transform, original_transform, image_root, max_words=30):

        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.original_transform = original_transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}

        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __len__(self):
        return len(self.ann)


    def __getitem__(self, index):

        ann = self.ann[index]
        image_path = os.path.join(self.image_root, ann['image'])
        image_rgb = Image.open(image_path).convert('RGB')
        image = self.transform(image_rgb)
        image_original = self.original_transform(image_rgb)
        caption = pre_caption(ann['caption'], self.max_words)
        label = torch.tensor(ann['label'])

        return image, image_original, caption, self.img_ids[ann['image_id']], label


class re_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        self.text = []
        # self.mask_text = []
        self.image = []
        # self.image_data = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption, self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1


    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):

        image_path = os.path.join(self.image_root, self.ann[index]['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, index
