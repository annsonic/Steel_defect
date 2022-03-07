import xml.etree.ElementTree as ET
from tqdm import tqdm
from pathlib import Path
import os
import shutil
import json


"""
[Source]
https://github.com/yukkyo/voc2coco/blob/master/voc2coco.py

[file tree]
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
- NEU-DET_COCO
    - images
        - *.jpg
    - annotations
        - train.json
        - val.json
- voc2coco.py

[Usage]
Modify `ROOT_FOLDER_PATH`, `VOC_FOLDER_PATH`
$ python voc2coco.py

"""

ROOT_FOLDER_PATH = r'D:/Documents/Academic/workspace/python_prj/now/Steel_defect/dataset'
VOC_FOLDER_PATH = os.path.join(ROOT_FOLDER_PATH, 'NEU-DET') 
LABELS = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
COPY_IMG = True


def get_coco_annotation_from_obj(obj, label2id):
    """
    Args:
        obj (ElementTree): xml file content
        label2id (dict): mapping label name to id
    Returns:
        ann (dict): bbox info
    """
    label = obj.findtext('name')
    assert label in label2id, f"Error: {label} is not in label2id !"
    category_id = label2id[label]
    bndbox = obj.find('bndbox')
    xmin = int(float(bndbox.findtext('xmin'))) - 1
    ymin = int(float(bndbox.findtext('ymin'))) - 1
    xmax = int(float(bndbox.findtext('xmax')))
    ymax = int(float(bndbox.findtext('ymax')))
    assert xmax > xmin and ymax > ymin, f"Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
    o_width = xmax - xmin
    o_height = ymax - ymin
    ann = {
        'area': o_width * o_height,
        'iscrowd': 0,
        'bbox': [xmin, ymin, o_width, o_height],
        'category_id': category_id,
        'ignore': 0,
        'segmentation': []  # This script is not for segmentation
    }
    return ann
    

def convert_label(annotation_paths, label2id, output_jsonpath):
    """ Read all xml files and parse content to the json file.
    Args:
        annotation_paths (list): list of xml file path
        label2id (dict): mapping label name to id
        output_jsonpath (str): json file path

    """
    # Initialize output content
    output_json_dict = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": []
    }

    img_id = 1 # START_IMAGE_ID
    bnd_id = 1  # START_BOUNDING_BOX_ID

    for a_path in tqdm(annotation_paths):
        # Read annotation xml
        ann_tree = ET.parse(a_path)
        ann_root = ann_tree.getroot()

        # Part: images
        filename = ann_root.findtext('filename')
        if not filename.endswith('.jpg'):
            filename += '.jpg'
        size = ann_root.find('size')
        width = int(size.findtext('width'))
        height = int(size.findtext('height'))
        img_info = {
            'file_name': filename,
            'height': height,
            'width': width,
            'id': img_id
        }
        output_json_dict['images'].append(img_info)

        # Part: annotations
        for obj in ann_root.findall('object'):
            ann = get_coco_annotation_from_obj(obj=obj, label2id=label2id)
            ann.update({'image_id': img_id, 'id': bnd_id})
            output_json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1
        
        img_id += 1

    # Part: categories
    for label, label_id in label2id.items():
        category_info = {'supercategory': 'none', 'id': label_id, 'name': label}
        output_json_dict['categories'].append(category_info)

    # Write output
    with open(output_jsonpath, 'w') as f:
        output_json = json.dumps(output_json_dict)
        f.write(output_json)

def main():
    # Output path
    imgs_path = Path(f'{ROOT_FOLDER_PATH}/NEU-DET_COCO/images/')
    lbs_path = Path(f'{ROOT_FOLDER_PATH}/NEU-DET_COCO/annotations/')
    imgs_path.mkdir(exist_ok=True, parents=True)
    lbs_path.mkdir(exist_ok=True, parents=True)

    # Label IDs
    labels_ids = list(range(1, len(LABELS)+1))
    label2id = dict(zip(LABELS, labels_ids))
    
    # Convert
    for image_set in ['train', 'validation']:
        
        # Input image list
        # image_ids = open(f'VOC{year}/ImageSets/Main/{image_set}.txt').read().strip().split()
        annot_folder = os.path.join(VOC_FOLDER_PATH, image_set, 'annotations')
        xml_files = os.listdir(annot_folder)

        # Input annotation list
        annotation_paths = []

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
            annotation_paths.append(xml_fp)

        # Get output txt path
        json_out_fp = lbs_path / image_set
        json_out_fp = json_out_fp.with_suffix('.json')

        # Convert labels to YOLO format
        convert_label(annotation_paths, label2id, json_out_fp)


if __name__ == '__main__':
    main()