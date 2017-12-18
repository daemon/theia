import numpy as np

def pad_square(image, std_pad=0):
    pad_w = max(image.shape[1] - image.shape[0], 0)
    pad_w1 = pad_w // 2 + std_pad
    pad_w2 = pad_w // 2 + pad_w % 2 + std_pad
    pad_h = max(image.shape[0] - image.shape[1], 0)
    pad_h1 = pad_h // 2 + std_pad
    pad_h2 = pad_h // 2 + pad_h % 2 + std_pad
    return np.pad(image, ((pad_w1, pad_w2), (pad_h1, pad_h2)), "constant")

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