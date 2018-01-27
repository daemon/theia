import json
import os
import random
import shutil
import xml.etree.ElementTree as ET

from PIL import Image

class RawBoxWriter(object):
    def __init__(self):
        pass

    def write(self, filename, boxes):
        with open(filename, "w") as f:
            f.write(repr([(wkst.dict(), bbox.dict()) for wkst, bbox in boxes]))

def _txt_subelem(root, tag, text=""):
    subelem = ET.SubElement(root, tag)
    subelem.text = str(text)
    return subelem

def bbox_to_xml(bbox):
    root = ET.Element("bndbox")
    _txt_subelem(root, "xmin", text=bbox["x"])
    _txt_subelem(root, "ymin", text=bbox["y"])
    _txt_subelem(root, "xmax", text=bbox["x"] + bbox["w"])
    _txt_subelem(root, "ymax", text=bbox["y"] + bbox["h"])
    return root

def labeled_box_to_xml(labeled_box):
    root = ET.Element("object")
    _txt_subelem(root, "name", text=labeled_box["label_name"])
    root.append(bbox_to_xml(labeled_box["box_data"]))
    return root

def size_to_xml(image):
    im = Image.open(image)
    w, h = im.size
    root = ET.Element("size")
    _txt_subelem(root, "width", text=w)
    _txt_subelem(root, "height", text=h)
    _txt_subelem(root, "depth", text=1)
    return root

def labeled_worksheet_to_xml(worksheet, labeled_boxes):
    root = ET.Element("annotation")
    _txt_subelem(root, "folder", text="VOC2007")
    _txt_subelem(root, "filename", text=os.path.basename(worksheet["image_path"]))
    root.append(size_to_xml(worksheet["image_path"]))
    _txt_subelem(root, "segmented", text=0)
    for box in labeled_boxes:
        root.append(labeled_box_to_xml(box))
    return root

def _try_mkdirs(directory):
    try:
        os.makedirs(directory)
    except:
        pass

def _init_voc_folders(base_name):
    _try_mkdirs(os.path.join(base_name, "Annotations"))
    _try_mkdirs(os.path.join(base_name, "ImageSets", "Layout"))
    _try_mkdirs(os.path.join(base_name, "ImageSets", "Main"))
    _try_mkdirs(os.path.join(base_name, "ImageSets", "Segmentation"))
    _try_mkdirs(os.path.join(base_name, "JPEGImages"))

class VocWriter(object):
    def __init__(self, folder_name, boxes):
        _init_voc_folders(folder_name)
        self.folder_name = folder_name
        self.box_table = {}
        self.worksheet_table = {}
        for worksheet, lbox in boxes:
            lbox = lbox.dict()
            self.worksheet_table[worksheet.id] = worksheet
            try:
                self.box_table[worksheet.id].append(lbox)
            except KeyError:
                self.box_table[worksheet.id] = [lbox]

        self.base_name = os.path.join(folder_name, "VOC2007")
        self.annotation_base = os.path.join(self.base_name, "Annotations")
        self.is_main_base = os.path.join(self.base_name, "ImageSets", "Main")
        self.jpg_base = os.path.join(self.base_name, "JPEGImages")

    def write_images(self):
        for wkst in self.worksheet_table.values():
            basename = os.path.basename(wkst.image_path)
            shutil.copy(wkst.image_path, os.path.join(self.jpg_base, basename))

    def write_splits(self):
        def write_names(names, ntype):
            filename = "{}.txt".format(ntype)
            with open(os.path.join(self.is_main_base, filename), "w") as f:
                f.write("\n".join(names))

        worksheets = list(self.worksheet_table.values())
        random.shuffle(worksheets)
        names = [os.path.basename(w.image_path).split(".")[0] for w in worksheets]

        train_index = int(0.8 * len(names))
        dev_index = int(0.9 * len(names))
        train_names = names[:train_index]
        dev_names = names[train_index:dev_index]
        test_names = names[dev_index:]
        write_names(train_names, "train")
        write_names(dev_names, "val")
        write_names(test_names, "test")

    def write_annotations(self):
        for wkst_id, lboxes in self.box_table.items():
            worksheet = self.worksheet_table[wkst_id]
            xml_str = ET.tostring(labeled_worksheet_to_xml(worksheet.dict(), lboxes))
            xml_name = "{}.xml".format(os.path.basename(worksheet.image_path).split(".")[0])
            with open(os.path.join(self.annotation_base, xml_name), "w") as f:
                f.write(xml_str.decode())
    
    def write(self):
        print("Writing annotations...")
        self.write_annotations()
        print("Writing splits...")
        self.write_splits()
        print("Writing images...")
        self.write_images()

def make_writer(writer_name, folder_name, boxes):
    return writers[writer_name](folder_name, boxes)

writers = dict(raw=RawBoxWriter(), voc=VocWriter)