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

def iou(rect1, rect2):
    ix1 = max([rect1[0], rect2[0]])
    iy1 = max([rect1[1], rect2[1]])
    ix2 = min([rect1[0] + rect1[2], rect2[0] + rect2[2]])
    iy2 = min([rect1[1] + rect1[3], rect2[1] + rect2[3]])
    if iy1 >= iy2 or ix1 >= ix2:
        i = 0
    else:
        i = (ix2 - ix1) * (iy2 - iy1)
    u = rect1[2] * rect1[3] + rect2[2] * rect2[3] - i + 1E-6
    return i / u

def find_xline(init_rect, rects, x_thresh=16, y_thresh=25, visited_set=set(), size_fac=0.6, max_depth=4):
    curr_rects = [init_rect]
    if max_depth == 0:
        return curr_rects
    x1, y1, w1, h1 = init_rect
    min_rect = None
    min_dist = np.inf
    for i, rect in enumerate(rects):
        if i in visited_set:
            continue
        x2, y2, w2, h2 = rect
        if x1 + w1 + x_thresh < x2 or abs(y2 - y1) > y_thresh or x2 <= x1:
            continue
        if w2 < size_fac * w1 and h2 < size_fac * h1:
            continue
        dist = (x1 + w1 - x2)**2 + 0.1 * (y2 - y1)**2
        if dist < min_dist:
            min_dist = dist
            min_rect = i, rect
    if not min_rect:
        return curr_rects
    i, rect = min_rect
    visited_set.add(i)
    curr_rects.extend(find_xline(rect, rects, x_thresh, y_thresh, visited_set, max_depth=max_depth - 1))
    return curr_rects

def find_first_xline_above(base_rect, rects, max_depth=3, x_thresh=16, y_thresh=25):
    min_dist = np.inf
    x0, y0, w0, h0 = base_rect
    x0 += w0 / 2
    for rect in rects:
        x, y, w, h = rect
        if y > y0 or x < x0 - w0 / 2:
            continue
        dist = (x - x0)**2 + (y - (y0 + h))**2
        if dist < min_dist:
            min_dist = dist
            min_rect = rect
    return find_xline(min_rect, rects, max_depth=max_depth, x_thresh=x_thresh, y_thresh=y_thresh)

def find_first_xline_below(base_rect, rects, max_depth=3, x_thresh=16, y_thresh=25):
    min_dist = np.inf
    x0, y0, w0, h0 = base_rect
    for rect in rects:
        x, y, w, h = rect
        if y < y0 + h0 or x < x0 - w0 / 2:
            continue
        dist = (x - x0)**2 + (y - (y0 + h0))**2
        if dist < min_dist:
            min_dist = dist
            min_rect = rect
    return find_xline(min_rect, rects, max_depth=max_depth, x_thresh=x_thresh, y_thresh=y_thresh)

def fetch_region(image, rect):
    x, y, w, h = rect
    return image[y:y + h, x:x + w]

def erase_region(image, rect):
    x, y, w, h = rect
    image[y:y + h, x:x + w] = 0
    return image

def nms(features, rect_key=None, score_key=None):
    del_set = set()
    for i, f1 in enumerate(features):
        rect1 = rect_key(f1)
        prob1 = score_key(f1)
        for j in range(i + 1, len(features)):
            f2 = features[j]
            rect2 = rect_key(f2)
            prob2 = score_key(f2)
            if iou(rect1, rect2) < 0.5:
                continue
            if prob2 < prob1:
                del_set.add(j)
            else:
                del_set.add(i)
    del_list = sorted(list(del_set), reverse=True)
    for i in del_list:
        del features[i]
    return features

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

def pad_square(image, std_pad=0):
    pad_w = max(image.shape[1] - image.shape[0], 0)
    pad_w1 = pad_w // 2 + std_pad
    pad_w2 = pad_w // 2 + pad_w % 2 + std_pad
    pad_h = max(image.shape[0] - image.shape[1], 0)
    pad_h1 = pad_h // 2 + std_pad
    pad_h2 = pad_h // 2 + pad_h % 2 + std_pad
    return np.pad(image, ((pad_w1, pad_w2), (pad_h1, pad_h2)), "constant")

class MnistDataset(data.Dataset):
    def __init__(self, images, labels, is_training):
        self.clean_images = []
        self.clean_labels = []
        self.unk_images = []
        self.unk_labels = []
        for image, label in zip(images, labels):
            is_clean = label <= 9 or label == 27
            image = np.transpose(image)
            if not is_clean:
                label = 11
            if label == 27:
                label = 10
            if is_clean:
                self.clean_images.append(Image.fromarray(image))
                self.clean_labels.append(int(label))
            else:
                self.unk_images.append(Image.fromarray(image))
                self.unk_labels.append(int(label))
        self.is_training = is_training

    @classmethod
    def splits(cls, config):
        data_dir = config.dir
        img_files = [os.path.join(data_dir, "emnist-balanced-train-images-idx3-ubyte"),
            os.path.join(data_dir, "emnist-balanced-test-images-idx3-ubyte")]
        image_sets = []
        for image_set in img_files:
            with open(image_set, "rb") as f:
                content = f.read()
            arr = read_idx(content)
            image_sets.append(arr)

        lbl_files = [os.path.join(data_dir, "emnist-balanced-train-labels-idx1-ubyte"),
            os.path.join(data_dir, "emnist-balanced-test-labels-idx1-ubyte")]
        lbl_sets = []
        for lbl_set in lbl_files:
            with open(lbl_set, "rb") as f:
                content = f.read()
            lbl_sets.append(read_idx(content).astype(np.int))
        return cls(image_sets[0], lbl_sets[0], True), cls(image_sets[1], lbl_sets[1], False)

    def __getitem__(self, index):
        if index < len(self.clean_labels):
            lbl = self.clean_labels[index]
            img = self.clean_images[index]
        else:
            index = random.randint(0, len(self.unk_labels) - 1)
            lbl = self.unk_labels[index]
            img = self.unk_images[index]
        if random.random() < 0.5 and self.is_training:
            img = img.rotate(random.randint(-15, 15))
            arr = np.array(img)
            if lbl == 8:
                arr[:random.randint(0, 10)] = 0
            if lbl == 10:
                if random.random() < 0.75:
                    a = -random.randint(1, 10)
                    b = max(28 - random.randint(1, 10), a)
                    c = random.randint(1, 10)
                    d = min(random.randint(1, 10), b)
                    arr[a:b, c:28 - d] = 255
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
            arr = np.roll(arr, random.randint(-2, 2), 0)
            arr = np.roll(arr, random.randint(-2, 2), 1)
            arr = (arr - np.mean(arr)) / np.sqrt(np.var(arr) + 1E-6)
            arr = arr.astype(np.float32)
            return torch.from_numpy(arr), lbl
        else:
            arr = np.array(img)
            arr = (arr - np.mean(arr)) / np.sqrt(np.var(arr) + 1E-6)
            return torch.from_numpy(arr.astype(np.float32)), lbl

    def __len__(self):
        return len(self.clean_images)

class ConvModel(SerializableModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 5)
        self.bn1 = nn.BatchNorm2d(64, affine=False)
        self.conv2 = nn.Conv2d(64, 96, 5)
        self.bn2 = nn.BatchNorm2d(96, affine=False)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.6)
        self.fc1 = nn.Linear(16 * 96, 1024)
        self.fc2 = nn.Linear(1024, 11)

    def label(self, digit, draw_input=False):
        digit = Image.fromarray(pad_square(digit, std_pad=5))
        x = np.array(digit.resize((28, 28), Image.BILINEAR), dtype=np.float32)
        if draw_input:
            Image.fromarray(x).show()
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

    for n_epoch in range(args.n_epochs):
        print("Epoch: {}".format(n_epoch + 1))
        for i, (model_in, labels) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()

            model_in = Variable(model_in.cuda(), requires_grad=False)
            labels = Variable(labels.cuda(), requires_grad=False)
            scores = model(model_in)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            if i % 16 == 0:
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
    parser.add_argument("--n_epochs", type=int, default=40)
    args = parser.parse_args()
    train(args)

model = ConvModel()

if __name__ == "__main__":
    main()