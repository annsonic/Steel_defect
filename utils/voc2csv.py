import os
import xml.etree.ElementTree as ET
import pandas as pd


"""
[file tree]
- NEU-DET
    - train
    - validation
        - annotations
            - crazing_*.xml
            - inclusion_*.xml
            - patches_*.xml
            - pitted_surface_*.xml
            - rolled-in_scale_*.xml
            - scratches_*.xml
        - images
            - crazing
                - crazing_*.jpg
            - inclusion
                - inclusion_*.jpg
            - patches
                - patches_*.jpg
            - pitted_surface
                - pitted_surface_*.jpg
            - rolled-in_scale
                - rolled-in_scale_*.jpg
            - scratches
                - scratches_*.jpg
"""

ROOT_FOLDER_PATH = r'D:/Documents/Academic/workspace/python_prj/now/Steel_defect/dataset/NEU-DET'
TRAIN_FOLDER = 'train'
VALID_FOLDER = 'validation'
IMG_FOLDER = 'images'
ANNOT_FOLDER = 'annotations'
LABELS = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']


def read_xml(fp):
    """ Get the content of xml file.
    Args:
        fp (str): xml file path
    Returns:
        width (int): image width
        height (int): image height
        boxes (list): list of [class name (str), [xmin (int), xmax(int), ymin (int), ymax (int)]]
    """
    tree = ET.parse(fp)
    root = tree.getroot()

    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    boxes = []
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls in LABELS and not int(obj.find('difficult').text) == 1:
            xmlbox = obj.find('bndbox')
            x1x2y1y2 = [int(xmlbox.find(x).text) for x in ('xmin', 'xmax', 'ymin', 'ymax')]
            boxes.append((cls, x1x2y1y2))
    
    return width, height, boxes


def get_annotations(root, files):
    """ Aggregate the content of all xml files.
    Args:
        root (str): folder path of the xml files
        files (list): the xml file names (str)
    Returns:
        list_width (list): width (int) of each image
        list_height (list): height (int) of each image
        list_box (list): element is [class name (str), [xmin (int), xmax(int), ymin (int), ymax (int)]]
    """
    list_width = []
    list_height = []
    list_box = []
    for file in files:
        fp = os.path.join(root, file)
        width, height, boxes = read_xml(fp)
        list_width.append(width)
        list_height.append(height)
        list_box.append(boxes)
    
    return (list_width, list_height, list_box)


def create_summary_csv():
    """ Collect all annotations into the csv file. File: `annotations.csv` """
    
    # Storage
    list_path = []
    list_xmin = []
    list_ymin = []
    list_xmax = []
    list_ymax = []
    list_class_name = []

    # Step 1. Read every xml file 
    for subset in [TRAIN_FOLDER, VALID_FOLDER]:
        annot_folder = os.path.join(ROOT_FOLDER_PATH, subset, ANNOT_FOLDER)
        files = os.listdir(annot_folder)

        for file in files:
            fp = os.path.join(annot_folder, file)
            width, height, boxes = read_xml(fp)

            for cls, x1x2y1y2 in boxes:
                [xmin, xmax, ymin, ymax] = x1x2y1y2
                img_fn = file.replace('xml', 'jpg')
                # str_digits = file.split('.')[0]
                # label_folder = ''.join([i for i in str_digits if not i.isdigit()])
                # label_folder = label_folder.strip('_')
                fp = os.path.join(img_fn)
                list_path.append(fp)
                list_xmin.append(xmin)
                list_ymin.append(ymin)
                list_xmax.append(xmax)
                list_ymax.append(ymax)
                list_class_name.append(cls)

    # Summary
    dict_summary = {
        'file_path': list_path, 
        'xmin': list_xmin, 
        'ymin': list_ymin, 
        'xmax': list_xmax, 
        'ymax': list_ymax,
        'class_name': list_class_name
    }
    
    # Step 2. Remove duplicate columns
    df = pd.DataFrame.from_dict(dict_summary)
    df = df.drop_duplicates(['file_path', 'class_name', 'xmin', 'ymin', 'xmax', 'ymax'], keep='last')

    # Step 3. Save
    df.to_csv('for_opt.csv', index=False, header=False, encoding='utf-8-sig')


if __name__ == '__main__':
    create_summary_csv()