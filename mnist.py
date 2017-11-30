import argparse
import io
import os
import random

from PIL import Image
from torch.autograd import Variable
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

class SerializableModule(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))

def read_idx(bytes):
    reader = io.BytesIO(bytes)
    reader.read(3)
    n_dims = int.from_bytes(reader.read(1), byteorder="big")
    sizes = []
    for _ in range(n_dims):
        sizes.append(int.from_bytes(reader.read(4), byteorder="big"))
    size = int(np.prod(sizes))
    buf = reader.read(size)
    return np.frombuffer(buf, dtype=np.uint8).reshape(sizes)

class MnistDataset(data.Dataset):
    def __init__(self, images, labels, is_training):
        self.images = []
        for image in images:
            self.images.append(Image.fromarray(image))
        self.labels = labels
        self.is_training = is_training

    @classmethod
    def splits(cls, config):
        data_dir = config.dir
        img_files = [os.path.join(data_dir, "train-images-idx3-ubyte"),
            os.path.join(data_dir, "t10k-images-idx3-ubyte")]
        image_sets = []
        for image_set in img_files:
            with open(image_set, "rb") as f:
                content = f.read()
            arr = read_idx(content)
            image_sets.append(arr)

        lbl_files = [os.path.join(data_dir, "train-labels-idx1-ubyte"),
            os.path.join(data_dir, "t10k-labels-idx1-ubyte")]
        lbl_sets = []
        for lbl_set in lbl_files:
            with open(lbl_set, "rb") as f:
                content = f.read()
            lbl_sets.append(read_idx(content).astype(np.int))
        return cls(image_sets[0], lbl_sets[0], True), cls(image_sets[1], lbl_sets[1], False)

    def __getitem__(self, index):
        lbl = self.labels[index]
        img = self.images[index]
        if random.random() < 0.5 and self.is_training:
            img = img.rotate(random.randint(-15, 15))
            arr = np.array(img)
            if random.random() < 0.5:
                resized = int(28 * (random.random() * 0.7 + 0.3))
                pad_w1 = random.randint(0, 28 - resized)
                pad_w2 = 28 - resized - pad_w1
                pad_h1 = random.randint(0, 28 - resized)
                pad_h2 = 28 - resized - pad_h1
                img.thumbnail((resized, resized))
                arr = np.pad(np.array(img), ((pad_w1, pad_w2), (pad_h1, pad_h2)), mode="constant")
            if random.random() < 0.5:
                arr = cv2.dilate(arr, np.ones((2, 2)))
            if random.random() < 0.7:
                arr = cv2.erode(arr, np.ones((2, 2)))
            if random.random() < 0.5:
                arr = arr + 50 * np.random.normal(size=(28, 28))
            if random.random() < 0.5:
                arr = arr + 255 * np.random.binomial(1, 0.05, size=(28, 28))
            arr = np.roll(arr, random.randint(-4, 4), 0)
            arr = np.roll(arr, random.randint(-4, 4), 1)
            arr = (arr - np.mean(arr)) / np.sqrt(np.var(arr) + 1E-6)
            arr = arr.astype(np.float32)
            return torch.from_numpy(arr), lbl
        else:
            arr = np.array(img)
            arr = (arr - np.mean(arr)) / np.sqrt(np.var(arr) + 1E-6)
            return torch.from_numpy(arr.astype(np.float32)), lbl

    def __len__(self):
        return len(self.images)

class ConvModel(SerializableModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 48, 5)
        self.bn1 = nn.BatchNorm2d(48, affine=False)
        self.conv2 = nn.Conv2d(48, 64, 5)
        self.bn2 = nn.BatchNorm2d(64, affine=False)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def label(self, digit):
        pad_w = max(digit.shape[1] - digit.shape[0], 0)
        std_pad = 5
        pad_w1 = pad_w // 2 + std_pad
        pad_w2 = pad_w // 2 + pad_w % 2 + std_pad
        pad_h = max(digit.shape[0] - digit.shape[1], 0)
        pad_h1 = pad_h // 2 + std_pad
        pad_h2 = pad_h // 2 + pad_h % 2 + std_pad

        digit = Image.fromarray(np.pad(digit, ((pad_w1, pad_w2), (pad_h1, pad_h2)), "constant"))
        x = np.array(digit.resize((28, 28), Image.BILINEAR), dtype=np.float32)
        x = (x - np.mean(x)) / np.sqrt(np.var(x) + 1E-6)
        x = Variable(torch.from_numpy(x), volatile=True).cuda()
        x = x.unsqueeze(0)
        ret = F.softmax(self.forward(x)).cpu().data[0].numpy()
        return np.argmax(ret), np.max(ret)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(self.bn1(F.relu(self.conv1(x))))
        x = self.pool(self.bn2(F.relu(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

def train(args):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss()

    train_set, test_set = MnistDataset.splits(args)
    train_loader = data.DataLoader(train_set, batch_size=64, shuffle=True, drop_last=True)
    test_loader = data.DataLoader(test_set, batch_size=min(32, len(test_set)))

    for _ in range(args.n_epochs):
        for i, (model_in, labels) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()

            model_in = Variable(model_in.cuda(), requires_grad=False)
            labels = Variable(labels.cuda(), requires_grad=False)
            scores = model(model_in)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            if i % 64 == 0:
                accuracy = (torch.max(scores, 1)[1].view(model_in.size(0)).data == labels.data).sum() / model_in.size(0)
                print("train accuracy: {:>10}, loss: {:>25}".format(accuracy, loss.data[0]))
    
    model.eval()
    n = 0
    accuracy = 0
    for model_in, labels in test_loader:
        model_in = Variable(model_in.cuda(), volatile=True)
        labels = Variable(labels.cuda(), volatile=True)
        scores = model(model_in)
        accuracy += (torch.max(scores, 1)[1].view(model_in.size(0)).data == labels.data).sum()
        n += model_in.size(0)
    print("test accuracy: {:>10}".format(accuracy / n))
    model.save(args.out_file)

def init_model(input_file=None, use_cuda=True):
    if use_cuda:
        model.cuda()
    if input_file:
        model.load(input_file)
    model.eval()

def main():
    init_model()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    parser.add_argument("--out_file", type=str, default="output.pt")
    parser.add_argument("--n_epochs", type=int, default=30)
    args = parser.parse_args()
    train(args)

model = ConvModel()

if __name__ == "__main__":
    main()