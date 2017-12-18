import enum
import random

from PIL import Image, ImageFont, ImageDraw
from scipy.signal import correlate
from sklearn.cluster import DBSCAN
import cv2
import numpy as np
import matplotlib.pyplot as plt

import kumon.model as mod
import mnist
import geometry as G

class QuestionEnum(enum.Enum):
    Q_VDIV = 0
    Q_VMULT = 1
    Q_VADD = 2
    Q_VSUB = 3

    @classmethod
    def find_label(cls, lbl_int):
        for q in cls:
            if q.value == lbl_int:
                return q

class Document(object):
    def __init__(self, file):
        im = Image.open(file)
        im.thumbnail((800, 800))
        self.orig_size = im.size
        self.image_data = self.crop(np.array(im))
        self.items = self.segment(self.image_data)
        self.mark_dict = self.grade()

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
        self._w_crop = min_left, min_right = self._compute_min_frame(image_data)
        image_data = image_data[:, min_left:min_right]
        self._h_crop = min_left, min_right = self._compute_min_frame(image_data, 1, 0)
        image_data = image_data[min_left:min_right, :]
        return image_data

    def _compute_number(self, digits):
        n = 0
        if not digits:
            return np.nan
        digits = list(sorted(digits, key=lambda x: -x[1]))
        for i, (digit, _) in enumerate(digits):
            n += 10**i * digit
        return n

    def _parse_vx_question(self, mult_height, rect):
        rect[1] -= (self.paper_height // 15 + mult_height)
        rect[0] += self.paper_width // 20
        rect[2] -= self.paper_width // 20
        rect[3] = self.paper_height // 15
        question = G.fetch_region(self.proc_data, rect)
        q_rect = rect.copy()
        q_rect[1] += mult_height + 3
        digits = self.segment_digits(question, use_image=True)
        digits1 = []
        digits2 = []
        for d in digits:
            d_rect = d[0]
            if (d_rect[2] < 6 and d_rect[3] < 6) or d_rect[3] < 8:
                continue
            lbl, prob = mnist.model.label(d[1], draw_input=False)
            if d_rect[1] > rect[3] // 3:
                digits2.append((lbl, d_rect[0]))
            else:
                digits1.append((lbl, d_rect[0]))
        n1 = self._compute_number(digits1)
        n2 = self._compute_number(digits2)

        rect[1] += self.paper_height // 30
        rect[0] -= self.paper_width // 20
        rect[2] = self.paper_width // 20
        rect[3] = self.paper_height // 30
        img = G.fetch_region(self.proc_data, rect)
        lbl, prob = mod.model.label(img)
        if prob < 0.8:
            return np.nan, lbl, q_rect
        lbl = QuestionEnum.find_label(lbl)
        if lbl == QuestionEnum.Q_VADD:
            ans = n1 + n2
        elif lbl == QuestionEnum.Q_VSUB:
            ans = n1 - n2
        elif lbl == QuestionEnum.Q_VMULT:
            ans = n1 * n2
        else:
            ans = np.nan
        return ans, lbl, q_rect

    def _parse_vdiv_question(self, mult_height, rect):
        if rect[1] - mult_height < 10 or rect[0] > self.paper_width / 4:
            return
        x, y, w, h = rect
        x += self.paper_width // 45
        y += mult_height + 3
        h -= mult_height + 3
        feats = self.segment_digits(self.proc_data[y:y + h, x:x + w], use_image=True)
        x -= self.paper_width // 45
        q_rect = x, y - 3, w, h + 3
        labels = []
        for feat in feats:
            lbl, prob = mnist.model.label(feat[1])
            if prob > 0.4:
                labels.append((lbl, feat[0][0]))
        if not labels:
            return None
        q_labels = sorted(labels, key=lambda x: x[1], reverse=True)
        q = 0
        for i, (lbl, _) in enumerate(q_labels):
            q += 10**i * lbl

        x -= self.paper_width // 10 + self.paper_width // 90
        w = self.paper_width // 10
        feats = self.segment_digits(self.proc_data[y:y + h, x:x + w])
        labels = []
        for feat in feats:
            d_rect = feat[0]
            if (d_rect[2] < 6 and d_rect[3] < 6) or d_rect[3] < 8:
                continue
            lbl, prob = mnist.model.label(feat[1])
            if prob > 0.4:
                labels.append((lbl, feat[0][0]))
        if not labels:
            return
        d_labels = sorted(labels, key=lambda x: x[1], reverse=True)
        d = 0
        for i, (lbl, _) in enumerate(d_labels):
            d += 10**i * lbl
        ans = (q // d, q % d)
        self.proc_data = G.erase_region(self.proc_data, q_rect)
        return ans, QuestionEnum.Q_VDIV, q_rect

    def parse_question(self, image_feat, slope_max=0.3):
        rect, ct_img = image_feat
        if ct_img.shape[1] < self.paper_width / 15:
            return
        lines = cv2.HoughLinesP(ct_img, 1, 3.1415926 / 180, 5, None, 4)
        if lines is None:
            return
        canvas = np.zeros((ct_img.shape[0], ct_img.shape[1]), dtype=np.uint8)
        has_bar = False
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if self.has_header and rect[1] + y1 < self.paper_height // 5:
                return
            slope = (y2 - y1) / (x1 - x2 + 1E-6)
            if np.abs(slope) < slope_max:
                has_bar = True
                cv2.line(canvas, (x1, y1), (x2, y2), 255)
        if not has_bar:
            return
        output = correlate(canvas, np.ones((2, canvas.shape[1])), mode="valid")
        mult_height = np.argmax(output)

        ans = self._parse_vdiv_question(mult_height, rect)
        if not ans:
            ans = self._parse_vx_question(mult_height, rect)
        return ans

    def segment_digits(self, image, use_image=False):
        im2, contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        image_features = []
        for c in contours:
            canvas = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            cv2.drawContours(canvas, c, -1, 255, 1)
            rx, ry, rw, rh = bounding_rect = cv2.boundingRect(canvas)
            if rw > 1 and rh > 0:
                img = image if use_image else canvas
                image_features.append((list(bounding_rect), img[ry:ry + rh, rx:rx + rw]))
        return image_features

    def _grade_vx(self, q_rect, rects):
        x, y, w, h = q_rect
        y += h
        x -= self.paper_width // 10
        h = min(self.paper_height // 15, self.paper_height - h)
        w = min(self.paper_width // 4, self.paper_width - w)
        digits = self.segment_digits(self.proc_data[y:y + h, x:x + w], use_image=True)
        ans_digits = []
        for d in digits:
            d_rect = d[0]
            if (d_rect[2] < 6 and d_rect[3] < 6) or d_rect[3] < 8:
                continue
            lbl, prob = mnist.model.label(d[1], draw_input=False)
            if prob < 0.4:
                continue
            ans_digits.append((lbl, d_rect[0]))
        ans = self._compute_number(ans_digits)
        return ans

    def _grade_vdiv(self, q_rect, rects):
        a_rects = list(reversed(G.find_first_xline_above(q_rect, rects)))
        ans_q = np.inf if len(a_rects) == 0 else 0
        ans_r = np.inf if len(a_rects) == 0 else 0
        is_remainder = True
        i = 0
        for rect in a_rects:
            x, y, w, h = rect
            lbl, prob = mnist.model.label(self.proc_data[y - 4:y + h, x:x + w], draw_input=False)
            if lbl == 10:
                is_remainder = False
                i = 0
                continue
            if is_remainder:
                ans_r += 10**i * lbl
            else:
                ans_q += 10**i * lbl
            i += 1
        return (ans_q, ans_r)

    def segment(self, image_data):
        image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)
        proc_data = image_data.astype(np.uint8)
        proc_data = cv2.GaussianBlur(proc_data, (9, 9), 0.7)
        proc_data = cv2.adaptiveThreshold(proc_data, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
        self.paper_height, self.paper_width = h, w = proc_data.shape
        self.proc_data = proc_data.astype(np.uint8)
        header_mean = np.mean(self.proc_data[:self.paper_height // 5] > 0)
        self.has_header = header_mean > 0.03
        
        feats = self.segment_digits(proc_data.astype(np.uint8))
        questions = []
        for i, feat in enumerate(feats):
            q_data = self.parse_question(feat)
            if q_data:
                questions.append(q_data)
        feats = self.segment_digits(self.proc_data)
        rects = [f[0] for f in feats]
        answers = []
        for ans, q_type, q_rect in questions:
            if q_type == QuestionEnum.Q_VDIV:
                answer = self._grade_vdiv(q_rect, rects)
            else:
                answer = self._grade_vx(q_rect, rects)
            answers.append(answer)
        return questions, answers

    def grade(self):
        marks = []
        font = ImageFont.truetype("DejaVuSans.ttf", self.paper_width // 20)
        image = Image.fromarray(self.image_data.copy())
        graphics = ImageDraw.Draw(image)
        questions, answers = self.items
        for i, (q, a) in enumerate(zip(questions, answers)):
            if np.isnan(q[0]) or np.isnan(a):
                continue
            rect = q[2]
            is_correct = bool(a == q[0])
            checkmark = "✓" if is_correct else "✗"
            fill = (100, 220, 20) if is_correct else (255, 0, 0)
            checkmark_pt = (rect[0] - self.paper_width / 10, rect[1] - self.paper_width / 30)
            graphics.text(checkmark_pt, checkmark, font=font, fill=fill)
            marks.append((is_correct, checkmark_pt))

        accuracy = [int(m[0]) for m in marks]
        grade = 100 * np.sum(accuracy) / len(accuracy)
        base_grade = round(grade)
        if abs(grade - base_grade) > 1E-6:
            grade_str = str(round(grade, 1))
        else:
            grade_str = str(int(round(grade)))
        graphics.text((10, 10), "{}%".format(grade_str), font=font, fill=(255, 0, 0))
        print("Grade: {}%".format(grade_str))

        w, h = self.orig_size
        dx, dy = self._w_crop[0], self._h_crop[0]
        mark_dict = [dict(x=(x + dx) / w, y=(y + dy) / h, is_correct=is_correct) for is_correct, (x, y) in marks]
        return mark_dict
