import argparse
import json
import os
import pickle

from PIL import Image
import cv2
import numpy as np

def draw_box(box, image, tag):
    color = name_to_color(tag)
    box = [int(b) for b in box]
    cv2.rectangle(image, tuple(box[:2]), tuple(box[2:]), color, 1)
    return image

_colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 100, 255], [0, 255, 100], [100, 255, 0], [0, 100, 255], 
    [100, 0, 255], [255, 0, 255]]

def name_to_color(name):
    h = abs(hash(name))
    return _colors[h % len(_colors)]

def visualize(boxes, images_path):
    for image_name, tags in boxes.items():
        image_path = os.path.join(images_path, "{}.jpg".format(image_name))
        print(image_path)
        im = cv2.imread(image_path)
        for tag_name, boxes in tags.items():
            for box in boxes:
                draw_box(box, im, tag_name)
        Image.fromarray(im).show()

def read_line(line):
    splits = line.split(" ")
    return splits[0], [float(x) for x in splits[2:]], float(splits[1])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", default="local_data/data", type=str)
    args = parser.parse_args()
    data_dir = os.path.join(args.data_folder, "VOCdevkit2007", "results", "VOC2007", "Main")
    boxes = {}
    for filename in os.listdir(data_dir):
        fullpath = os.path.join(data_dir, filename)
        with open(fullpath) as f:
            for line in f.readlines():
                line = line.strip()
                name, box, score = read_line(line)
                if score < 0.7:
                    continue
                try:
                    boxes[name][filename].append(box)
                except KeyError:
                    try:
                        boxes[name][filename] = [box]
                    except KeyError:
                        boxes[name] = {filename: [box]}
    print(boxes)
    visualize(boxes, os.path.join(args.data_folder, "VOCdevkit2007", "VOC2007", "JPEGImages"))

if __name__ == "__main__":
    main()