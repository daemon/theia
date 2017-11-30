from PIL import Image
from scipy.signal import correlate
from sklearn.cluster import KMeans
import cv2
import numpy as np
import matplotlib.pyplot as plt

import mnist

class Document(object):
    def __init__(self, file):
        im = Image.open(file)
        im = im.convert("L")
        image_data = self.crop(np.array(im))
        self.items = self.segment(image_data)
        self.grade()

    def _compute_min_frame(self, data, dim=0, dim_out=1, mean_min=100, std_max=20):
        mean_color = np.mean(data, dim)
        color_std = np.sqrt(np.var(cv2.Laplacian(data, cv2.CV_64F), dim))
        mask = np.logical_and(mean_color > mean_min, color_std < std_max)
        l = data.shape[dim_out]
        for min_left in range(l):
            if mask[min_left]:
                break
        for min_right in reversed(range(l)):
            if mask[min_right]:
                break
        if min_left > min_right:
            raise ValueError
        return min_left, min_right

    def crop(self, image_data):
        min_left, min_right = self._compute_min_frame(image_data)
        image_data = image_data[:, min_left:min_right]
        min_left, min_right = self._compute_min_frame(image_data, 1, 0)
        image_data = image_data[min_left:min_right, :]
        image_data = image_data[image_data.shape[0] // 10:, :]
        return image_data

    def split_multiply_bar(self, img, slope_max=0.3):
        img = cv2.Laplacian(img, cv2.CV_64F)
        img = cv2.GaussianBlur(img, (3, 3), 0.5)
        ret, img = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
        img = img.astype(np.uint8)
        
        lines = cv2.HoughLinesP(img, 1, 3.1415926 / 180, 5, None, 4)
        canvas = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x1 - x2 + 1E-6)
            if np.abs(slope) < slope_max:
                cv2.line(canvas, (x1, y1), (x2, y2), (255,))
        
        output = correlate(canvas, np.ones((2, canvas.shape[1])), mode="valid")
        mult_height = np.argmax(output)
        top = img[:mult_height - 2, :]
        bottom = img[mult_height + 3:, :]
        return top, bottom

    def segment(self, image_data, k_factor=10, pos_threshold=20):
        proc_data = cv2.Canny(image_data, 100, 255)
        h, w = proc_data.shape

        cluster_data = np.stack(np.where(proc_data >= pos_threshold), 1)

        def squashed_kmeans(cluster_data, n_clusters, k_factor=k_factor, k_dim=0, dist_cutoff=np.inf):
            fit_data = np.copy(cluster_data)
            if k_dim is not None:
                fit_data[:, k_dim] = fit_data[:, k_dim] / k_factor

            kmeans = KMeans(n_clusters).fit(fit_data)
            data = {}
            for l, c in zip(kmeans.labels_, cluster_data):
                try:
                    data[l].append(c)
                except KeyError:
                    data[l] = [c]
            for l in data.keys():
                filtered_list = []
                std = np.sqrt(np.var(np.array(data[l])))
                for c in data[l]:
                    if np.linalg.norm(c - kmeans.cluster_centers_[l]) < dist_cutoff * std:
                        filtered_list.append(c)
                data[l] = filtered_list
            return data

        data = squashed_kmeans(cluster_data, 2, k_dim=0)
        group1 = np.array(data[0])
        group2 = np.array(data[1])
        data1 = squashed_kmeans(group1, 6, k_dim=1)
        data2 = squashed_kmeans(group2, 6, k_dim=1)
        data2 = {l + 6: val for l, val in data2.items()}
        data1.update(data2)

        def min_dist(contour1, contour2):
            min_dist = np.inf
            for p1 in contour1:
                for p2 in contour2:
                    dist = np.linalg.norm(p1[0] - p2[0])
                    min_dist = dist if dist < min_dist else min_dist
            return min_dist

        def segment_digits(image):
            im2, contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            image_features = []
            for c in contours:
                canvas = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                cv2.drawContours(canvas, c, -1, 255, 1)
                rx, ry, rw, rh = bounding_rect = cv2.boundingRect(canvas)
                image_features.append((bounding_rect, image[ry:ry + rh, rx:rx + rw]))
            return image_features

        items = []
        for k, v in data1.items():
            cluster = np.array(v)
            rx, ry, rw, rh = cv2.boundingRect(cluster)
            segment = proc_data.astype(np.uint8)[rx:rx + rw, ry:ry + rh]
            im_data = image_data.astype(np.uint8)[rx:rx + rw, ry:ry + rh]
            question, answer = self.split_multiply_bar(im_data)
            question_features, answer_features = segment_digits(question), segment_digits(answer)
            items.append((question_features, answer_features))
        return items

    def grade(self):
        accuracy = []
        for question_features, answer_features in self.items:
            areas = []
            labels = []
            digit_order = []
            for bounding_rect, digit in answer_features:
                digit_order.append(bounding_rect[0])
                areas.append(np.prod(digit.shape))
                labels.append(mnist.model.label(digit)[0])
            labels = np.array(labels)
            digit_order = np.array(digit_order)
            area_order = np.argsort(areas)[-3:]
            
            digits = labels[area_order]
            digit_order = digit_order[area_order]
            digits = digits[np.argsort(digit_order)]
            answer = np.sum(np.power(10, np.arange(3)[::-1]) * digits)

            areas = []
            labels = []
            digit_order = []
            for bounding_rect, digit in question_features:
                digit_order.append(bounding_rect[0] * bounding_rect[1])
                areas.append(np.prod(digit.shape))
                labels.append(mnist.model.label(digit)[0])
            labels = np.array(labels)
            digit_order = np.array(digit_order)
            area_order = np.argsort(areas)[-3:]

            digits = labels[area_order]
            digit_order = digit_order[area_order]
            digits = digits[np.argsort(digit_order)]
            true_answer = (10 * digits[0] + digits[1]) * digits[2]
            accuracy.append(int(true_answer == answer))
        print("Accuracy: {}".format(np.sum(accuracy) / len(accuracy)))

