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

from geometry import fit_resize, pad, pad_square, iou, within_bounds, RectScaler, fetch_region

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

class SingleMnistDataset(data.Dataset):
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
    def splits(cls, config, **kwargs):
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
        return cls(image_sets[0], lbl_sets[0], True, **kwargs), cls(image_sets[1], lbl_sets[1], False, **kwargs)

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

class RegionProposalGenerator(SingleMnistDataset):
    def __init__(self, images, labels, is_training, **kwargs):
        super().__init__(images, labels, is_training)
        self.digit_range = (1, 5)
        self.model = kwargs["model"]

    @classmethod
    def splits(cls, config, model):
        return super().splits(config, model=model)

    def next(self, iteration=2):
        items = []
        erode_roll = random.random() < 0.7
        dilate_roll = random.random() < 0.5
        resize_roll = random.random() < 0.5
        offset_x_roll = random.random() < 0.7
        offset_y_roll = random.random() < 0.5
        for _ in range(random.randint(*self.digit_range)):
            rnd_idx = random.randint(0, len(self.clean_labels) - 1)
            lbl = self.clean_labels[rnd_idx]
            img = self.clean_images[rnd_idx].copy()
            arr = np.array(img)
            if self.is_training:
                if random.random() < 0.5:
                    img = img.rotate(random.randint(-15, 15))
                    arr = np.array(img)
                if random.random() < 0.5 and resize_roll:
                    resized = int(28 * (random.random() * 0.3 + 0.7))
                    pad_w1 = random.randint(0, 28 - resized)
                    pad_w2 = 28 - resized - pad_w1
                    pad_h1 = random.randint(0, 28 - resized)
                    pad_h2 = 28 - resized - pad_h1
                    img.thumbnail((resized, resized))
                    arr = np.pad(np.array(img), ((pad_w1, pad_w2), (pad_h1, pad_h2)), mode="constant")
                if lbl == 8:
                    arr[:random.randint(0, 10)] = 0
                if dilate_roll:
                    arr = cv2.dilate(arr, np.ones((2, 2)))
                if erode_roll:
                    arr = cv2.erode(arr, np.ones((2, 2)))
            offset_x = random.randint(-16, -8) if offset_x_roll else 0
            offset_y = random.randint(5, 16) if offset_y_roll else 0
            rect = list(cv2.boundingRect(arr))
            items.append((lbl, arr, (offset_x, offset_y), rect))

        curr_x = random.randint(0, 10)
        canvas = np.zeros((28  + max(item[2][1] for item in items),
            len(items) * 28 + curr_x + sum(item[2][0] for item in items[:-1])))
        curr_y = items[0][2][1]
        for item in items:
            lbl, img, (offset_x, offset_y), rect = item
            rect[0] += curr_x
            rect[1] += curr_y
            canvas[curr_y:curr_y + 28, curr_x:curr_x + 28] = np.maximum(canvas[curr_y:curr_y + 28, curr_x:curr_x + 28], img)
            curr_x += 28 + offset_x
            curr_y = offset_y
        canvas = np.array(fit_resize(Image.fromarray(canvas), 28))
        scale_factor = 28 / canvas.shape[0]
        gt_abs_rects = [item[3] for item in items]
        for rect in gt_abs_rects:
            rect[0] *= scale_factor
            rect[1] *= scale_factor
            rect[2] *= scale_factor
            rect[3] *= scale_factor
        gt_rel_rects = self._scale_rects(gt_abs_rects)
        if random.random() < 0.5 and self.is_training:
            canvas = canvas + 50 * np.random.normal(size=canvas.shape)
        if random.random() < 0.5 and self.is_training:
            canvas = canvas + 255 * np.random.binomial(1, 0.05, size=canvas.shape)
        img = canvas
        canvas = torch.from_numpy((canvas - np.mean(canvas)) / np.sqrt(np.var(canvas) + 1E-6))
        
        neg_examples = []
        pos_examples = []
        features = self.model.encode(Variable(canvas.unsqueeze(0), volatile=True).cuda().float()).squeeze(0)
        h, w = canvas.size()
        canvas_rect = (0, 0, w, h)
        for i in range(features.size(1)):
            for j in range(features.size(2)):
                feats = features[:, i, j]
                a_box1, a_box2 = self._feat_to_anchor_box(j, i)
                if not within_bounds(a_box1, canvas_rect):
                    a_box1 = None
                if not within_bounds(a_box2, canvas_rect):
                    a_box2 = None
                for abs_rect, rel_rect in zip(gt_abs_rects, gt_rel_rects[0]):
                    if a_box1 is None:
                        continue
                    if iou(abs_rect, a_box1) > 0.6:
                        pos_examples.append((a_box1, feats, rel_rect))
                    elif iou(abs_rect, a_box1) < 0.3:
                        neg_examples.append((a_box1, feats, rel_rect))
                for abs_rect, rel_rect in zip(gt_abs_rects, gt_rel_rects[1]):
                    if a_box2 is None:
                        continue
                    if iou(abs_rect, a_box2) > 0.6:
                        pos_examples.append((a_box2, feats, rel_rect))
                    elif iou(abs_rect, a_box2) < 0.3:
                        neg_examples.append((a_box2, feats, rel_rect))
        random.shuffle(pos_examples)
        random.shuffle(neg_examples)
        neg_examples = neg_examples[:len(pos_examples)]
        print(len(pos_examples), len(items))
        if len(pos_examples) == 0:
            #Image.fromarray(img).show()
            return self.next(iteration=iteration)
        #Image.fromarray(img).show()
        if iteration > 0:
            p_examples, n_examples = self.next(iteration=iteration - 1)
            pos_examples.extend(p_examples)
            neg_examples.extend(n_examples)
        Image.fromarray(fetch_region(img, pos_examples[0][0])).show()
        return pos_examples, neg_examples

    def _feat_to_anchor_box(self, x, y):
        x += 0.5
        y += 0.5
        x *= 7
        y *= 7
        w1 = h1 = 20
        h2 = 20
        w2 = 12
        return ([int(x - w1 // 2), int(y - h1 // 2), w1, h1], [int(x - w2 // 2), int(y - h2 // 2), w2, h2])

    def _scale_rects(self, rects):
        scaled_rects = ([], [])
        for rect in rects:
            x, y, w, h = rect
            scaler = RectScaler.from_absolute(rect)
            x_a = int(7 * (x // 7 + 0.5))
            y_a = int(7 * (y // 7 + 0.5))
            h_a = w_a = 20
            rect1 = scaler.to_relative([x_a, y_a, w_a, h_a])
            h_a = 20
            w_a = 12
            rect2 = scaler.to_relative([x_a, y_a, w_a, h_a])
            scaled_rects[0].append(rect1)
            scaled_rects[1].append(rect2)
        return scaled_rects

    def __len__(self):
        return len(self.clean_images)

class ProposalNetwork(SerializableModule):
    def __init__(self):
        super().__init__()
        self.model = ConvModel()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(96 * 9, 256)
        self.fc_bbox = nn.Linear(256, 8)
        self.fc_cls = nn.Linear(256, 4)

    def loss(self, bboxes, classes, truth_bboxes, truth_classes):
        criterion_bbox = nn.SmoothL1Loss(reduce=False)
        criterion_cls = nn.CrossEntropyLoss()
        def local_repeat(x, n):
            x = x.repeat(n, 1)
            x = x.permute(1, 0)
            x = x.contiguous()
            return x.view(-1)
        loss_bbox = criterion_bbox(bboxes, truth_bboxes)
        loss_bbox = local_repeat(truth_classes, 2) * loss_bbox
        loss_bbox = loss_bbox.sum() / truth_classes.sum()
        truth_classes = torch.chunk(truth_classes, 2, 1)
        classes = torch.split(classes, 2, 1)
        classes = [torch.stack(classes[::2]), torch.stack(classes[1::2])]
        loss_cls = criterion_cls(classes[0], truth_classes[0]) + criterion_cls(classes[1], truth_classes[1])
        return loss_cls / 2 + loss_bbox

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x.view(x.size(0), -1))))
        bboxes.append(self.fc_bbox(x))
        classes.append(self.fc_cls(x))
        return torch.stack(bboxes), torch.stack(classes)

class ConvModel(SerializableModule):
    def __init__(self):
        super().__init__()
        self.use_cuda = True
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
        x = Variable(torch.from_numpy(x), volatile=True)
        if self.use_cuda:
            x = x.cuda()
        x = x.unsqueeze(0)
        ret = F.softmax(self.forward(x)).cpu().data[0].numpy()
        return np.argmax(ret), np.max(ret)

    def encode(self, x):
        x = x.unsqueeze(1)
        x = self.pool(self.bn1(F.relu(self.conv1(x))))
        x = self.pool(self.bn2(F.relu(self.conv2(x))))
        return x

    def forward(self, x):
        x = self.encode(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

def train_proposal(args):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss()
    base_model = ConvModel()
    base_model.load(args.base_in_file)
    base_model.cuda()
    base_model.eval()

    train_set, test_set = RegionProposalGenerator.splits(args, model=base_model)
    train_set.next()

def train_single(args):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss()

    train_set, test_set = SingleMnistDataset.splits(args)
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

def init_model(mode, input_file=None, use_cuda=True):
    global model
    if mode == "proposal":
        model = ConvModel()
    else:
        model = ConvModel()
    model.use_cuda = use_cuda
    if use_cuda:
        model.cuda()
    if input_file:
        model.load(input_file)
    model.eval()

model = None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_in_file", type=str, default="")
    parser.add_argument("--dir", type=str)
    parser.add_argument("--in_file", type=str, default="")
    parser.add_argument("--mode", type=str, default="proposal", choices=["proposal", "single"])
    parser.add_argument("--out_file", type=str, default="output.pt")
    parser.add_argument("--n_epochs", type=int, default=40)
    args = parser.parse_args()
    global model
    init_model(args.mode, input_file=args.in_file)
    if args.mode == "proposal":
        train_proposal(args)
    else:
        train_single(args)

if __name__ == "__main__":
    main()