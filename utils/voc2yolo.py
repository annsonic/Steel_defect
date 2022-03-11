import xml.etree.ElementTree as ET
from tqdm import tqdm
from pathlib import Path
import os
import shutil


"""
[Source]
https://github.com/ultralytics/yolov5/blob/26f0415287b7fa333f559a8300cedc2274943ab6/data/VOC.yaml

[file tree]
- ROOT_FOLDER_PATH 
    - NEU-DET
        - train
        - validation
            - annotations
                - crazing_*.xml
                - inclusion_*.xml
                - patches_*.xml
                - pittes_surface_*.xml
                - rolled-in_scale_*.xml
                - scratches_*.xml
            - images
                - crazing
                    - crazing_*.jpg
                - inclusion
                    - inclusion_*.jpg
                - patches
                    - patches_*.jpg
                - pittes_surface
                    - pittes_surface_*.jpg
                - rolled-in_scale
                    - rolled-in_scale_*.jpg
                - scratches
                    - scratches_*.jpg
    - NEU-DET_YOLO
        - images
            -validation
                -*.jpg
            -train
                -*.jpg
        - labels
            -validation
                -*.txt
            -train
                -*.txt
- voc2yolo.py

[Usage]
Modify `ROOT_FOLDER_PATH` to yours.
Modify `COPY_IMG` if you want to copy images to new image folder.
Modify `LABELS` to your class names.
$ python voc2yolo.py

"""

ROOT_FOLDER_PATH = r'D:/Documents/Academic/workspace/python_prj/now/Steel_defect/dataset'
VOC_FOLDER_PATH = os.path.join(ROOT_FOLDER_PATH, 'NEU-DET') 
LABELS = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
COPY_IMG = False # If true, copy images to new image folder


def convert_label(in_fp, out_fp):
    """ Convert VOC format to YOLO format and save file. File: `{raw}.txt`
    Args:
        in_fp (str): file path of input xml
        out_fp (str): file path of output txt
    """
    def convert_box(size, box):
        """ Convert pixel coordinate to fractional.
        Args:
            size (tuple): (image width, image height)
            box (list): [xmin (float), xmax(float), ymin (float), ymax (float)]
        """
        dw, dh = 1. / size[0], 1. / size[1]
        x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
        return x * dw, y * dh, w * dw, h * dh

    in_file = open(in_fp)
    out_file = open(out_fp, 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls in LABELS and not int(obj.find('difficult').text) == 1:
            xmlbox = obj.find('bndbox')
            bb = convert_box((w, h), [float(xmlbox.find(x).text) for x in ('xmin', 'xmax', 'ymin', 'ymax')])
            cls_id = LABELS.index(cls)  # class id
            out_file.write(" ".join([str(a) for a in (cls_id, *bb)]) + '\n')


# Convert
for image_set in ['train', 'validation']:
    # Output path
    imgs_path = Path(f'{ROOT_FOLDER_PATH}/NEU-DET_YOLO/images/{image_set}')
    lbs_path = Path(f'{ROOT_FOLDER_PATH}/NEU-DET_YOLO/labels/{image_set}')
    imgs_path.mkdir(exist_ok=True, parents=True)
    lbs_path.mkdir(exist_ok=True, parents=True)

    # Input image list
    # image_ids = open(f'VOC{year}/ImageSets/Main/{image_set}.txt').read().strip().split()
    annot_folder = os.path.join(VOC_FOLDER_PATH, image_set, 'annotations_clean')
    xml_files = os.listdir(annot_folder)

    for xml_fn in tqdm(xml_files, desc=f'{image_set}'):
        if COPY_IMG:
            # Get input image path
            img_fn = xml_fn.replace('xml', 'jpg')
            str_digits = img_fn.split('.')[0]
            label_folder = ''.join([i for i in str_digits if not i.isdigit()])
            label_folder = label_folder.strip('_')
            img_fp = os.path.join(VOC_FOLDER_PATH, image_set, 'images', label_folder, img_fn)

            # Copy image
            shutil.copy(src=img_fp, dst=str(imgs_path))

        # Get input xml path
        xml_fp = Path(f'{annot_folder}/{xml_fn}')
        # Get output txt path
        xml_out_fp = lbs_path / xml_fn
        txt_out_fp = xml_out_fp.with_suffix('.txt')

        # Convert labels to YOLO format
        convert_label(xml_fp, txt_out_fp)
        