from PIL import Image, ImageFont, ImageDraw
from scipy.signal import correlate
from sklearn.cluster import DBSCAN
import cv2
import numpy as np
import matplotlib.pyplot as plt

import mnist

class Document(object):
    def __init__(self, file, debug=False):
        self.debug = debug
        im = Image.open(file)
        if debug:
            im.show()
        self.image_data = self.crop(np.array(im))
        self.items = self.segment(self.image_data)
        self.grade()

    def _compute_min_frame(self, data, dim=0, dim_out=1, mean_min=100, std_max=100):
        mean_color = np.mean(np.mean(data, 2), dim)
        data = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
        color_std = np.max(cv2.Laplacian(data, cv2.CV_64F), dim)
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

    def segment(self, image_data, k_factor=10, pos_threshold=20, rh_threshold=0.035):
        image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)
        proc_data = cv2.Canny(image_data, 100, 255)
        self.paper_height, self.paper_width = h, w = proc_data.shape

        cluster_data = np.stack(np.where(proc_data >= pos_threshold), 1)
        def cluster_dbscan(cluster_data, eps, core_pts):
            dbscan = DBSCAN(eps, core_pts).fit(cluster_data)
            data = {}
            for l, c in zip(dbscan.labels_, cluster_data):
                try:
                    data[l].append(c)
                except KeyError:
                    data[l] = [c]

            n_questions = 0
            brects = {}
            for l, cluster in list(data.items()):
                ry, rx, rh, rw = rect = cv2.boundingRect(np.array(cluster))
                rh_pct = rh / w
                if rh_pct < rh_threshold:
                    n_questions += 1
                    del data[l]
                brects[l] = rect

            if n_questions >= len(data.keys()):
                return data

            distances = {}
            for l1, cluster1 in data.items():
                ry1, rx1, rh1, rw1 = brects[l1]
                p11, p12, p13, p14 = np.array([ry1, rx1]), np.array([ry1 + rh1, rx1]),\
                    np.array([ry1, rx1 + rw1]), np.array([ry1 + rh1, rx1 + rw1])
                for l2, cluster2 in data.items():
                    ry2, rx2, rh2, rw2 = brects[l2]
                    p2 = np.array([ry2, rx2])
                    if l1 == l2 or rh2 > rh1:
                        continue
                    distances[l1, l2] = min(np.linalg.norm(p2 - p11), np.linalg.norm(p2 - p12),\
                        np.linalg.norm(p2 - p13), np.linalg.norm(p2 - p14))
            distances = sorted(list(distances.items()), key=lambda x: x[1])[:len(data.keys()) - n_questions]
            for (lbl, merge_lbl), dist in distances:
                if dist > min(self.paper_width, self.paper_height) / 20:
                    continue
                data[lbl].extend(data[merge_lbl])
                del data[merge_lbl]
            return data

        data1 = cluster_dbscan(cluster_data, h / 35, 5)

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
            try:
                question, answer = self.split_multiply_bar(im_data)
            except TypeError:
                continue
            question_features, answer_features = segment_digits(question), segment_digits(answer)
            items.append((question_features, answer_features, (rx, ry, rw, rh)))
        return items

    def grade(self):
        accuracy = []
        font = ImageFont.truetype("DejaVuSans.ttf", self.paper_width // 20)
        image = Image.fromarray(self.image_data.copy())
        graphics = ImageDraw.Draw(image)
        for i, (question_features, answer_features, rect) in enumerate(self.items):
            areas = []
            labels = []
            digit_order = []
            for bounding_rect, digit in answer_features:
                digit_order.append(bounding_rect[0])
                areas.append(np.prod(digit.shape))
                labels.append(mnist.model.label(digit, draw_input=False)[0])
            labels = np.array(labels)
            digit_order = np.array(digit_order)
            area_order = np.argsort(areas)[-3:]
            
            digits = labels[area_order]
            digit_order = digit_order[area_order]
            digits = digits[np.argsort(digit_order)]
            try:
                answer = np.sum(np.power(10, np.arange(3)[::-1]) * digits)
            except ValueError:
                answer = None

            areas = []
            labels = []
            digit_order = []
            for bounding_rect, digit in question_features:
                lbl, prob = mnist.model.label(digit, draw_input=False)
                if prob < 0.85:
                    continue
                rx, ry, rw, rh = bounding_rect
                digit_order.append((rx + rw) * (ry + rh))
                areas.append(np.prod(digit.shape))
                labels.append(lbl)
            labels = np.array(labels)
            digit_order = np.array(digit_order)
            area_order = np.argsort(areas)[-3:]

            digits = labels[area_order]
            digit_order = digit_order[area_order]
            digits = digits[np.argsort(digit_order)]
            try:
                true_answer = (10 * digits[0] + digits[1]) * digits[2]
            except IndexError:
                continue

            is_correct = true_answer == answer
            checkmark = "✓" if is_correct else "✗"
            fill = (100, 220, 20) if is_correct else (255, 0, 0)
            print("{} × {} = {} {}".format(10 * digits[0] + digits[1], digits[2], answer, checkmark))

            graphics.text((rect[1] - self.paper_width / 30, rect[0]), checkmark, font=font, fill=fill)
            accuracy.append(int(is_correct))

        grade = 100 * np.sum(accuracy) / len(accuracy)
        base_grade = round(grade)
        if abs(grade - base_grade) > 1E-6:
            grade_str = str(round(grade, 1))
        else:
            grade_str = str(int(round(grade)))
        graphics.text((10, 10), "{}%".format(grade_str), font=font, fill=(255, 0, 0))
        if self.debug:
            image.show()
            print("Grade: {}%".format(grade_str))

