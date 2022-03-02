import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, ElementTree
from tqdm import tqdm
from pathlib import Path
import os
from operator import itemgetter

"""
[raw file tree]
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
- clean_labels.py

[raw file tree]
- NEU-DET
    - train
    - validation
        - annotations_clean
            - crazing_*.xml
            - inclusion_*.xml
            - patches_*.xml
            - pittes_surface_*.xml
            - rolled-in_scale_*.xml
            - scratches_*.xml

[Usage]
Modify `ROOT_FOLDER_PATH`, `DILATED_WIDTH`, `OVERLAP_FRACTION`
$ python clean_labels.py

"""

# Parent folder of `train` and `validation` folder
ROOT_FOLDER_PATH = r'D:/Documents/Academic/workspace/python_prj/now/Steel_defect/dataset/NEU-DET'
# Class labels
LABELS = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
# For merging near-by boxes, DILATED_WIDTH ~ 0.5 * gap_between_boxes 
DILATED_WIDTH = 10
# If overlapped area > OVERLAP_FRACTION * min_box_area, merge boxes
OVERLAP_FRACTION = 0.4
DICT_FOLDER = {
    'crazing': 'cz',
    'inclusion': 'in',
    'patches': 'pa', 
    'pitted_surface': 'ps', 
    'rolled-in_scale': 'rs', 
    'scratches': 'sc'
}


def read_xml(in_fp):
    """ Read content of a xml file.
    Args:
        in_fp (str): xml file path
    Returns:
        w (int): image width
        h (int): image height
        dict_group (dict): key is class name (str), 
                           value is list of [xmin (int), xmax(int), ymin (int), ymax (int)]
    
    """
    in_file = open(in_fp)

    # Read xml file
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    dict_group = {}
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls in LABELS and not int(obj.find('difficult').text) == 1:
            xmlbox = obj.find('bndbox')
            bb = [int(xmlbox.find(x).text) for x in ('xmin', 'xmax', 'ymin', 'ymax')]
            if cls in dict_group:
                dict_group[cls].append(bb)
            else:
                dict_group[cls] = [bb]
    return w, h, dict_group


def dilate_rectangle(box, width, height):
    """ Dilate the rectangle by DILATED_WIDTH.
    Args:
        box (list): [xmin (int), xmax(int), ymin (int), ymax (int)]
        width (int): image width, prevent dilation from exceeding width
        height (int): image height, prevent dilation from exceeding height
    Returns:
        box (list): [xmin (int), xmax(int), ymin (int), ymax (int)]

    """
    xmin = max(0, box[0] - DILATED_WIDTH)
    xmax = min(width, box[1] + DILATED_WIDTH)
    ymin = max(0, box[2] - DILATED_WIDTH)
    ymax = min(height, box[3] + DILATED_WIDTH)

    return [xmin, xmax, ymin, ymax]


def compute_overlapped_area(boxa, boxb, dilated=False, width=0, height=0):
    """
    Args:
        boxa (list): [xmin (int), xmax(int), ymin (int), ymax (int)]
        boxb (list): [xmin (int), xmax(int), ymin (int), ymax (int)]
        dilated (bool): whether to dilate the boxes
        width (int): image width, prevent dilation from exceeding width
        height (int): image height, prevent dilation from exceeding height
    Returns:
        area (int): overlapped area

    """
    # Scan along x-axis
    list_sort_xmin = sorted([boxa, boxb], key=itemgetter(0))

    # Copy
    box1 = list_sort_xmin[0]
    box2 = list_sort_xmin[1]

    # Get left-side x1 and right-side x2
    x1 = box2[0]
    x2 = box1[1]
    if x2 < x1: # No intersection
        return 0
    
    if dilated:
        box1 = dilate_rectangle(box1, width, height)
        box2 = dilate_rectangle(box1, width, height)
    
    # Get upper-side y1 and bottom-side y2
    if box2[2] < box1[3]:
        y1, y2 = box2[2], box1[3]
    else:
        y1, y2 = box1[2], box2[3]
    
    # Calculate intersection area
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    return area


def area(box):
    """ Compute the area of the box. """
    width = box[1] - box[0]
    height = box[3] - box[2]
    return width * height


def merge(box1, box2):
    """ Union two rectangles. 
    Args:
        box1 (list): [xmin (int), xmax(int), ymin (int), ymax (int)]
        box2 (list): [xmin (int), xmax(int), ymin (int), ymax (int)]
    Returns:
        box (list): [xmin (int), xmax(int), ymin (int), ymax (int)]

    """
    
    x1 = min(box1[0], box2[0])
    x2 = max(box1[1], box2[1])
    y1 = min(box1[2], box2[2])
    y2 = max(box1[3], box2[3])
    
    return [x1, x2, y1, y2]


def union_boxes(list_boxes, dilated, width, height):
    """ Iteratively union rectangles.
    Inspired from 
    https://stackoverflow.com/questions/48557865/merging-overlapping-axis-aligned-rectangles
    
    Args:
        list_boxes (list): list of [xmin (int), xmax(int), ymin (int), ymax (int)]
        dilated (bool): whether to dilate the boxes
        width (int): image width, prevent dilation from exceeding width
        height (int): image height, prevent dilation from exceeding height
    Returns:
        list_boxes (list): list of [xmin (int), xmax(int), ymin (int), ymax (int)]

    """

    # Scan along x-axis
    list_sort_xmin = sorted(list_boxes, key=itemgetter(0))

    n_box = len(list_boxes)
    for i in range(n_box-1):
        if list_sort_xmin[i] is None:
            continue

        # Shallow copy
        box = list_sort_xmin[i][:]

        # The enlarged box may intersect with previous boxes
        list_index_review = []

        # Examine the rest boxes
        for j in range(n_box):
            if list_sort_xmin[j] is None or j == i:
                continue

            overlapped_area = compute_overlapped_area(box, 
                list_sort_xmin[j], dilated, width, height)

            min_area = min(area(box), area(list_sort_xmin[j]))
            thresh = 1 if dilated else (OVERLAP_FRACTION * min_area)
            
            if overlapped_area > thresh:
                # Enlarge box
                box = merge(box, list_sort_xmin[j])
                
                # Eraze box_j
                list_sort_xmin[j] = None

                # Check previous boxes
                for k in list_index_review[::-1]:
                    if list_sort_xmin[k] is None:
                        continue

                    overlapped_area = compute_overlapped_area(box, 
                        list_sort_xmin[k], dilated, width, height)
                    min_area = min(area(box), area(list_sort_xmin[k]))
                    thresh = 1 if dilated else (OVERLAP_FRACTION * min_area)
                    
                    if overlapped_area > thresh:
                        # Enlarge box
                        box = merge(box, list_sort_xmin[k])
                        
                        # Eraze box_k
                        list_sort_xmin[k] = None
            else:
                list_index_review.append(j)
        
        # Update
        list_sort_xmin[i] = box[:]

    # Remove non-exist boxes
    list_boxes = [box for box in list_sort_xmin if box is not None]

    return list_boxes


def indent(elem, level=0):
    """ Pretty print with new line and indentation. 
    source: https://stackoverflow.com/questions/52595012/xml-elementtree-with-pretty-print
    Args:
        elem (Element): xml contents reference
        level (int): indention scale

    """
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def write_xml(output_path, filename, 
              width, height, list_bbox):
    """ Write `root_path/annotations_clean/filename+'.xml` file.

    Args:
        output_path (str): output folder of xml file
        filename (str): file name without extension
        width (int): image width
        height (int): image height
        list_bbox (list): list of [class_name, xmin (int), xmax(int), ymin (int), ymax (int)]
    
    """
    # Extract folder name from file_name -> row index
    folder_name = ''.join([i for i in filename if not i.isdigit()])
    folder_name = folder_name.strip('_')

    root = Element('annotation')
    SubElement(root, 'folder').text = DICT_FOLDER[folder_name]
    SubElement(root, 'filename').text = filename + '.jpg'
    source = SubElement(root, 'source')
    SubElement(source, 'database').text = 'NEU-DET'

    # Size
    size = SubElement(root, 'size')
    SubElement(size, 'width').text = str(width)
    SubElement(size, 'height').text = str(height)
    SubElement(size, 'depth').text = '1'
    SubElement(root, 'segmented').text = '0'

    # Boxes
    for (class_name, xmin, xmax, ymin, ymax) in list_bbox:
        # xmin, ymin, xmax, ymax = entry
        
        obj = SubElement(root, 'object')
        SubElement(obj, 'name').text = class_name
        SubElement(obj, 'pose').text = 'Unspecified'
        SubElement(obj, 'truncated').text = '0'
        SubElement(obj, 'difficult').text = '0'

        bbox = SubElement(obj, 'bndbox')
        SubElement(bbox, 'xmin').text = str(xmin)
        SubElement(bbox, 'ymin').text = str(ymin)
        SubElement(bbox, 'xmax').text = str(xmax)
        SubElement(bbox, 'ymax').text = str(ymax)

    tree = ElementTree(root)
    indent(root)
    xml_filename = os.path.join(output_path, filename+'.xml')
    tree.write(xml_filename)


def clean(xml_files, annot_folder, output_folder, image_set):
    """ Clean the annotations by merging near boxes.
    Args:
        xml_files (list): list of xml file names
        annot_folder (str): parent folder of input xml files
        output_folder (str): parent folder of output xml files
        image_set (str): train or validation
    
    """
    for xml_fn in tqdm(xml_files, desc=f'{image_set}'):
        # Get input xml path
        in_fp = Path(f'{annot_folder}/{xml_fn}')
        w, h, dict_group = read_xml(in_fp)
        
        for class_name in dict_group:
            if class_name in ['patches', 'scratches'] or len(dict_group[class_name]) < 2:
                # Ignore this group
                continue

            if class_name in ['crazing', 'pitted_surface', 'rolled-in_scale']:
                dilated = True
            else:
                dilated = False

            list_boxes = union_boxes(dict_group[class_name], dilated, w, h)
            dict_group[class_name] = list_boxes

        # Convert dict to list
        list_bbox = []
        for class_name in dict_group:
            for box in dict_group[class_name]:
                list_bbox.append([class_name])
                list_bbox[-1].extend(box)

        # Write xml in pretty format
        write_xml(output_folder, xml_fn.split('.')[0], 
              w, h, list_bbox)


def main():
    for image_set in ['train', 'validation']:
        # Get file name list
        annot_folder = os.path.join(ROOT_FOLDER_PATH, image_set, 'annotations')
        xml_files = os.listdir(annot_folder)

        # Create output folder
        output_folder = os.path.join(ROOT_FOLDER_PATH, image_set, 'annotations_clean')
        lbs_path = Path(output_folder)
        lbs_path.mkdir(exist_ok=True, parents=True)

        clean(xml_files, annot_folder, output_folder, image_set)


if __name__ == '__main__':
    main()
