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

from geometry import pad_square

class SerializableModule(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage), strict=False)

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
        return cls(image_sets[0], lbl_sets[0], True), cls(image_sets[1], lbl_sets[1], True)

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

class SequentialMnistDataset(SingleMnistDataset):
    def __init__(self, images, labels, is_training):
        super().__init__(images, labels, is_training)
        self.digit_range = (1, 8)

    @classmethod
    def splits(cls, config):
        return super().splits(config)

    def __getitem__(self, index):
        items = []
        erode_roll = random.random() < 0.7
        dilate_roll = random.random() < 0.5
        resize_roll = random.random() < 0.5
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
                    resized = int(28 * (random.random() * 0.5 + 0.5))
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
            offset = random.randint(-16, 0)
            items.append((lbl, arr, offset))

        canvas = np.zeros((28, len(items) * 28))
        curr_x = 0
        for item in items:
            lbl, img, offset = item
            canvas[:, curr_x:curr_x + 28] = np.maximum(canvas[:, curr_x:curr_x + 28], img)
            curr_x += 28 + offset
        if random.random() < 0.5 and self.is_training:
            canvas = canvas + 50 * np.random.normal(size=canvas.shape)
        if random.random() < 0.5 and self.is_training:
            canvas = canvas + 255 * np.random.binomial(1, 0.05, size=canvas.shape)
        labels = [item[0] for item in items]
        labels.append(11)
        Image.fromarray(canvas).show()
        canvas = (canvas - np.mean(canvas)) / np.sqrt(np.var(canvas) + 1E-6)
        return torch.from_numpy(canvas).float(), labels

    def __len__(self):
        return len(self.clean_images)

class Decoder(SerializableModule):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        def make_weight(in_size, out_size):
            return nn.Parameter(nn.init.xavier_normal(torch.Tensor(in_size, out_size)))
        self.w0 = make_weight(output_size, hidden_size)
        self.wz = make_weight(output_size, hidden_size)
        self.wr = make_weight(output_size, hidden_size)
        self.ws = make_weight(input_size, hidden_size)
        
        self.wa = make_weight(hidden_size, hidden_size)
        self.ua = make_weight(input_size, hidden_size)
        self.va = make_weight(hidden_size, 1)
        
        self.u0 = make_weight(hidden_size, hidden_size)
        self.uz = make_weight(hidden_size, hidden_size)
        self.ur = make_weight(hidden_size, hidden_size)
        self.c0 = make_weight(input_size, hidden_size)
        self.cz = make_weight(input_size, hidden_size)
        self.cr = make_weight(input_size, hidden_size)

        self.fc = nn.Linear(hidden_size, output_size)

    def compute_context(self, last_s, input_states):
        e_logits = []
        s = torch.matmul(last_s, self.wa)
        for in_j in input_states:
            e_ij = torch.matmul(F.tanh(s + torch.matmul(in_j, self.ua)), self.va)
            e_logits.append(e_ij.squeeze(0).squeeze(1))
        norm = torch.sum(torch.exp(torch.stack(e_logits)), 0)
        c_i = 0
        for i, in_j in enumerate(input_states):
            a_ij = torch.exp(e_logits[i]) / norm
            c_i += a_ij.unsqueeze(1) * in_j.squeeze(0)
        return c_i

    def output(self, last_s, last_out, hidden_states):
        c_i = self.compute_context(last_s, hidden_states)
        r_i = F.sigmoid(torch.matmul(last_out, self.wr) + torch.matmul(last_s, self.ur) + \
            torch.matmul(c_i, self.cr))
        z_i = F.sigmoid(torch.matmul(last_out, self.wz) + torch.matmul(last_s, self.uz) + \
            torch.matmul(c_i, self.cz))
        sh_i = F.tanh(torch.matmul(last_out, self.w0) + torch.matmul(r_i * last_s, self.u0) + \
            torch.matmul(c_i, self.c0))
        s_i = (1 - z_i) * last_s + z_i * sh_i
        return s_i

    def predict(self, x):
        hidden_states = torch.chunk(x, x.size(0), 0)
        s = [F.tanh(torch.matmul(hidden_states[0], self.ws))]
        outputs = [self.fc(s[0].squeeze(0))]
        lbl = torch.max(outputs[-1], 1)[1].cpu().data[0]
        alive = lbl != 11
        labels = []
        while alive:
            lbl = "R" if lbl == 10 else str(lbl)
            labels.append(lbl)
            s_i = self.output(s[-1], outputs[-1], hidden_states)
            s.append(s_i)
            outputs.append(self.fc(s_i.squeeze(0)))
            lbl = torch.max(outputs[-1], 1)[1].cpu().data[0]
            alive = lbl != 11
        return torch.stack(outputs).permute(1, 0, 2), "".join(labels)

    def forward(self, x, max_labels):
        hidden_states = torch.chunk(x, x.size(0), 0)
        s = [F.tanh(torch.matmul(hidden_states[0], self.ws))]
        outputs = [torch.stack([self.fc(x) for x in s[0].squeeze(0)])]
        for i in range(1, max_labels):
            s_i = self.output(s[-1], outputs[-1], hidden_states)
            s.append(s_i)
            outputs.append(self.fc(s_i.squeeze(0)))
        return torch.stack(outputs).permute(1, 0, 2)

class Seq2SeqModel(SerializableModule):
    def __init__(self):
        super().__init__()
        self.use_cuda = True
        self.encode_rnn = nn.GRU(28, 200, 2, batch_first=True, bidirectional=True)
        self.decode_rnn = Decoder(400, 300, 12)

    def loss(self, scores, labels):
        criterion = nn.CrossEntropyLoss()
        loss = 0
        score_chunks = torch.chunk(scores, scores.size(0), 0)
        label_chunks = torch.chunk(labels, labels.size(0), 0)
        for s, l in zip(score_chunks, label_chunks):
            s = s.squeeze(0)
            l = l.squeeze(0)
            loss += criterion(s, l)
        return loss / scores.size(0)

    def predict(self, x):
        rnn_out, _ = self.encode_rnn(x.permute(0, 2, 1))
        return self.decode_rnn.predict(rnn_out.permute(1, 0, 2))

    def accuracy(self, scores, labels):
        accuracies = []
        score_chunks = torch.chunk(scores, scores.size(0), 0)
        label_chunks = torch.chunk(labels, labels.size(0), 0)
        for s, l in zip(score_chunks, label_chunks):
            s = s.squeeze(0)
            l = l.squeeze(0)
            accuracies.append((torch.max(s, 1)[1].view(l.size(0)).data == l.data).sum() / l.size(0))
        return np.mean(accuracies)

    def forward(self, x, max_labels):
        rnn_out, _ = self.encode_rnn(x.permute(0, 2, 1))
        return self.decode_rnn(rnn_out.permute(1, 0, 2), max_labels)

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

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(self.bn1(F.relu(self.conv1(x))))
        x = self.pool(self.bn2(F.relu(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

def train_sequential(args):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss()

    def collate_fn(batch):
        max_img_width = max(b[0].size()[1] for b in batch)
        max_lbl_len = max(len(b[1]) for b in batch)
        images = []
        labels = []
        for image, label in batch:
            label.extend([11] * (max_lbl_len - len(label)))
            labels.append(torch.LongTensor(label))
            if image.size(1) == max_img_width:
                images.append(image)
                continue
            padding = torch.zeros(28, max_img_width - image.size(1))
            images.append(torch.cat([image, padding], 1))
        return torch.stack(images), torch.stack(labels)

    train_set, test_set = SequentialMnistDataset.splits(args)
    train_loader = data.DataLoader(train_set, batch_size=64, shuffle=True, drop_last=True, collate_fn=collate_fn)
    test_loader = data.DataLoader(test_set, batch_size=1, collate_fn=collate_fn)

    for n_epoch in range(args.n_epochs):
        if n_epoch == 20:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.05, weight_decay=0.0005)
        elif n_epoch == 35:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.005, weight_decay=0.0005)
        print("Epoch: {}".format(n_epoch + 1))
        for i, (model_in, labels) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()

            model_in = Variable(model_in.cuda(), requires_grad=False)
            labels = Variable(labels.cuda(), requires_grad=False)
            scores = model(model_in, labels.size(1))
            loss = model.loss(scores, labels)
            accuracy = model.accuracy(scores, labels)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), max_norm=5)
            optimizer.step()
            print("train accuracy: {:>20}, loss: {:>25}".format(accuracy, loss.data[0]))

            a = random.randint(1, 7)
            train_set.digit_range = (a, a + 1)
        print("saving model...")
        model.save(args.out_file)

    model.eval()
    n = 0
    accuracies = []
    test_set.use_training = True
    for i, (model_in, labels) in enumerate(test_loader):
        a = random.randint(1, 7)
        test_set.digit_range = (a, a + 1)
        model_in = Variable(model_in.cuda(), requires_grad=False)
        labels = Variable(labels.cuda(), requires_grad=False)
        scores, label = model.predict(model_in)
        print(label)
        print()
        print()
        if i == 18:
            return
    print("final test accuracy: {:>10}".format(np.mean(accuracies)))

def train_single(args):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss()

    train_set, test_set = SequentialMnistDataset.splits(args)
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
    model.use_cuda = use_cuda
    if use_cuda:
        model.cuda()
    if input_file:
        model.load(input_file)
    model.eval()

model = None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    parser.add_argument("--in_file", type=str, default="")
    parser.add_argument("--mode", type=str, default="sequential", choices=["sequential", "single"])
    parser.add_argument("--out_file", type=str, default="output.pt")
    parser.add_argument("--n_epochs", type=int, default=40)
    args = parser.parse_args()
    global model
    if args.mode == "sequential":
        model = Seq2SeqModel()
        init_model(input_file=args.in_file)
        train_sequential(args)
    else:
        model = ConvModel()
        init_model(input_file=args.in_file)
        train_single(args)

if __name__ == "__main__":
    main()