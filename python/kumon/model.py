import argparse
import hashlib
import json
import os
import random

from PIL import Image, ImageDraw
from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mnist import SerializableModule, pad_square

def iou(rect1, rect2):
    ix1 = torch.max([rect1[0], rect2[0]])
    iy1 = torch.max([rect1[1], rect2[1]])
    ix2 = torch.min([rect1[0] + rect1[2], rect2[0] + rect2[2]])
    iy2 = torch.min([rect1[1] + rect1[3], rect2[1] + rect2[3]])
    if iy1 >= iy2 or ix1 >= ix2:
        i = 0
    else:
        i = (ix2 - ix1) * (iy2 - iy1)
    u = rect1[2] * rect1[3] + rect2[2] * rect2[3] - i + 1E-6
    return i / u

class CNNModel(SerializableModule):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(1, 16, 4)
        self.bn0 = nn.BatchNorm2d(16, affine=False)
        self.conv1 = nn.Conv2d(16, 16, 4)
        self.bn1 = nn.BatchNorm2d(16, affine=False)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16 * 25, 4)

    def loss(self, scores, labels):
        criterion = nn.CrossEntropyLoss()
        labels = Variable(labels, requires_grad=False).cuda()
        return criterion(scores, labels)

    def label(self, image, draw_input=False):
        image = Image.fromarray(pad_square(image))
        x = np.array(image.resize((16, 16), Image.BILINEAR), dtype=np.float32)
        if draw_input:
            Image.fromarray(x).show()
        x = (x - np.mean(x)) / np.sqrt(np.var(x) + 1E-6)
        x = Variable(torch.from_numpy(x), volatile=True)
        if self.use_cuda:
            x = x.cuda()
        x = x.unsqueeze(0)
        ret = F.softmax(self.forward(x)).cpu().data[0].numpy()
        return np.argmax(ret), np.max(ret)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.bn0(F.relu(self.conv0(x)))
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc1(x)

class KumonDataset(data.Dataset):
    def __init__(self, config, dataset, use_clean=False):
        super().__init__()
        self.config = config
        self.examples = dataset
        self.use_clean = use_clean

    def __len__(self):
        return len(self.examples)

    @staticmethod
    def _erase_region(image, rect):
        x, y, w, h = rect
        image[y:y + h, x:x + w] = 0
        return image

    def __getitem__(self, index):
        is_none = index >= len(self.examples)
        if is_none:
            image, data = random.choice(self.examples)
        else:
            image, data = self.examples[index]
        image = image.copy()
        random.shuffle(data)
        if is_none:
            for d in data:
                rect = d["rect"]
                self._erase_region(image, rect)
            a = random.randint(0, image.shape[0] - 16)
            b = random.randint(0, image.shape[1] - 16)
            image = image[a:a + 16, b:b + 16]
            t = 0
        else:
            rand_choice = random.choice(data)
            t = rand_choice["type"]
            rx, ry, rw, rh = rand_choice["rect"]
            image = image[ry:ry + rh, rx:rx + rw]

            h, w = image.shape[:2]
            min_len = min(h, w)
            pad11 = random.randint(0, w - min_len)
            pad12 = max(0, w - min_len - pad11)
            pad21 = random.randint(0, h - min_len)
            pad22 = max(0, h - min_len - pad21)
            image = np.pad(image, ((pad11, pad12), (pad21, pad22)), mode="constant")
            image = Image.fromarray(image).resize((16, 16), Image.BILINEAR)
            if random.random() < 0.5:
                image = image.rotate(random.randint(-15, 15))
            image = np.array(image)
        if random.random() < 0.5 and not self.use_clean:
            image = image + np.abs(15 * np.random.normal(size=(16, 16)))
        if random.random() < 0.5 and not self.use_clean:
            image = image + 100 * np.random.binomial(1, 0.005, size=(16, 16))
        if not self.use_clean:
            image = np.roll(image, random.randint(-4, 4), 0)
            image = np.roll(image, random.randint(-4, 4), 1)
        image = (image - np.mean(image)) / np.sqrt(np.var(image) + 1E-6)
        return torch.from_numpy(image).float(), t

    @staticmethod
    def find_all_examples(directory, types):
        examples = []
        for name in os.listdir(directory):
            full_name = os.path.join(directory, name)
            if os.path.isfile(full_name) and full_name.endswith(".dat"):
                img_arr = np.array(Image.open(full_name[:-4]).convert("L"))
                img_arr2 = cv2.GaussianBlur(img_arr, (9, 9), 1)
                img_arr = cv2.adaptiveThreshold(img_arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 20)
                img_arr2 = cv2.adaptiveThreshold(img_arr2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 10)
                img_arr = 0.5 * img_arr + 0.5 * img_arr2
                img = Image.fromarray(img_arr)
                img.thumbnail((800, 800))
                factor = img_arr.shape[1] / img.size[0]

                with open(full_name) as f:
                    data = json.loads(f.read())
                for box in data:
                    x, y, rw, rh = box["rect"]
                    x = int(x / factor)
                    y = int(y / factor)
                    rw = int(rw / factor)
                    rh = int(rh / factor)
                    box["rect"] = [x, y, rw, rh]
                    box["type"] = types.index(box["type"])
                examples.append((np.array(img), data, full_name))
            elif os.path.isdir(full_name):
                examples.extend(KumonDataset.find_all_examples(full_name, types))
        return examples

    @classmethod
    def splits(cls, config):
        example_pairs = cls.find_all_examples(config.directory, config.types)
        sets = [[], []]
        for image, data, full_name in example_pairs:
            h = int(hashlib.md5(full_name.encode()).hexdigest(), 16)
            sets[int(h % 100 > 80)].append((image, data))
        return cls(config, sets[0]), cls(config, sets[1], True)

def draw_predictions(image, prediction):
    img = 5 * (image + np.min(image)).astype(np.uint8)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    y, x, h, w = prediction * 384
    draw.rectangle([x, y, x + w, y + h], fill=255)
    return img

def train(args):
    if args.in_file:
        model.load(args.in_file)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

    train_set, test_set = KumonDataset.splits(args)
    train_loader = data.DataLoader(train_set, batch_size=4, shuffle=True, drop_last=True)
    test_loader = data.DataLoader(test_set, batch_size=min(32, len(test_set)), shuffle=True)
    avg_loss = None
    min_loss = np.inf

    for _ in range(args.n_epochs):
        for i, (model_in, labels) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            model_in = Variable(model_in.cuda(), requires_grad=False)
            model_out = model(model_in)
            loss = model.loss(model_out, labels)
            loss.backward()
            optimizer.step()
            accuracy = (torch.max(model_out, 1)[1].data == labels.cuda()).sum() / labels.size(0)
            loss = loss.cpu().data[0]
            print("train loss: {}, accuracy: {}".format(loss, accuracy))
            avg_loss = loss if avg_loss is None else 0.9 * avg_loss + 0.1 * loss
            if avg_loss < min_loss:
                min_loss = avg_loss
                model.save(args.out_file)
                print("saving best model: {}".format(avg_loss))
    model.eval()
    for i, (model_in, labels) in enumerate(test_loader):
            model_in = Variable(model_in.cuda(), requires_grad=False)
            model_out = model(model_in)
            loss = model.loss(model_out, labels)
            accuracy = (torch.max(model_out, 1)[1].data == labels.cuda()).sum() / labels.size(0)
            print("test total loss: {}, accuracy: {}".format(loss.cpu().data[0], accuracy))

model = CNNModel()

def init_model(input_file=None, use_cuda=True):
    model.use_cuda = use_cuda
    if use_cuda:
        model.cuda()
    if input_file:
        model.load(input_file)
    model.eval()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str)
    parser.add_argument("--in_file", type=str, default="")
    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--out_file", type=str, default="output.pt")
    parser.add_argument("--types", nargs="+", type=str, default=["_none_", "vmult", "vadd", "vsub"])
    train(parser.parse_known_args()[0])

if __name__ == "__main__":
    main()