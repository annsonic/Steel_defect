import os
from re import sub
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sn
import cv2
import random


"""
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
- statistics.py

[Usage]
Modify `ROOT_FOLDER_PATH`
$ python statistics.py

"""
ROOT_FOLDER_PATH = r'D:/Documents/Academic/workspace/python_prj/now/Steel_defect/dataset/NEU-DET'
TRAIN_FOLDER = 'train'
VALID_FOLDER = 'validation'
IMG_FOLDER = 'images'
ANNOT_FOLDER = 'annotations_clean'
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
    list_subset = []
    list_name = []
    list_width = []
    list_height = []
    list_class_name = []
    list_xmin = []
    list_ymin = []
    list_xmax = []
    list_ymax = []

    # Step 1. Read every xml file 
    for subset in [TRAIN_FOLDER, VALID_FOLDER]:
        annot_folder = os.path.join(ROOT_FOLDER_PATH, subset, ANNOT_FOLDER)
        files = os.listdir(annot_folder)

        for file in files:
            fp = os.path.join(annot_folder, file)
            width, height, boxes = read_xml(fp)

            for cls, x1x2y1y2 in boxes:
                [xmin, xmax, ymin, ymax] = x1x2y1y2

                list_subset.append(subset)
                list_name.append(file)
                list_width.append(width)
                list_height.append(height)
                list_class_name.append(cls)
                list_xmin.append(xmin)
                list_ymin.append(ymin)
                list_xmax.append(xmax)
                list_ymax.append(ymax)

    # Summary
    dict_summary = {
        'subset': list_subset,
        'file_name': list_name, 
        'width': list_width, 
        'height': list_height, 
        'class_name': list_class_name,
        'xmin': list_xmin, 
        'ymin': list_ymin, 
        'xmax': list_xmax, 
        'ymax': list_ymax
    }
    
    # Step 2. Remove duplicate columns
    df = pd.DataFrame.from_dict(dict_summary)
    df_dup = df[df[['file_name', 'class_name', 'xmin','ymin','xmax', 'ymax']].duplicated()]
    print("Duplicate box:\n", df_dup)
    df = df.drop_duplicates(['file_name', 'class_name', 'xmin','ymin','xmax', 'ymax'], keep='last')

    # Step 3. Append 5 columns: (xmin+xmax)/2; (ymin+ymax)/2; xmax-xmin; ymax-ymin; aspect_ratio
    df['xc'] = df.apply(lambda x: mean(x['xmin'], x['xmax']), axis=1)
    df['yc'] = df.apply(lambda x: mean(x['ymin'], x['ymax']), axis=1)
    df['w'] = df.apply(lambda x: difference(x['xmin'], x['xmax']), axis=1)
    df['h'] = df.apply(lambda x: difference(x['ymin'], x['ymax']), axis=1)
    df['aspect_ratio'] = df.apply(lambda x: aspect_ratio(x['w'], x['h']), axis=1)
    # Step 3. Save
    df.to_csv('annotations.csv', index=False, encoding='utf-8-sig')


def plot_class_per_image():
    """ Plot histogram: number of box in an image. File: `n_class_per_image.png`"""

    df = pd.read_csv('annotations.csv', encoding='utf-8-sig')
    sub_series = df.value_counts(subset=['file_name'], sort=False)
    counts = sub_series.to_numpy()
    bin = np.max(counts) - np.min(counts)

    probs, bins, patches = plt.hist(counts, bin, density=True, facecolor='g', alpha=0.75)
    n = 0 # index of bins
    dx = (bins[1] - bins[0]) / 2 # bin width
    # Annotate probability value on the histogram chart
    for fr, center, patch in zip(probs, bins, patches):
        plt.annotate("{:.2f}".format(probs[n]),
                    xy = (center+dx, 0),          # top left corner of the histogram bar
                    xytext = (0,0.2),             # offsetting label position above its bar
                    textcoords = "offset points", # Offset (in points) from the *xy* value
                    ha = 'center', va = 'bottom'
                    )
        n += 1

    plt.xlabel('Boxes per image')
    plt.ylabel('Probability')
    plt.title('Histogram of Boxes per image')
    plt.xlim(0, bin+1)
    plt.ylim(0, 0.5)
    plt.grid(True)
    plt.savefig('n_class_per_image.png')
    plt.close()
    

def autolabel(ax, rects):
    """Attach a text label above each bar in *rects*, displaying its height.
    Args:
        ax (matplotlib.axes): Current figure
        rects (matplotlib.container.BarContainer): Container for the bar plots

    """
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def plot_class_count():
    """ Plot histogram: number of box per class. File: `n_class.png`"""

    df = pd.read_csv('annotations.csv', encoding='utf-8-sig')
    train_df = df[df['subset']==TRAIN_FOLDER]
    valid_df = df[df['subset']==VALID_FOLDER]
    
    train_series = train_df.value_counts(subset=['class_name'], sort=False)
    valid_series = valid_df.value_counts(subset=['class_name'], sort=False)

    train_counts = train_series.to_numpy()
    valid_counts = valid_series.to_numpy()

    classes = train_series.index.get_level_values('class_name').values
    
    x = np.arange(len(classes))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots(figsize=(8,5))
    rects1 = ax.bar(x - width/2, train_counts, width, label=TRAIN_FOLDER)
    rects2 = ax.bar(x + width/2, valid_counts, width, label=VALID_FOLDER)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Count')
    ax.set_title('Box count by dataset and class')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    autolabel(ax, rects1)
    autolabel(ax, rects2)
    fig.tight_layout()
    plt.grid(True)
    plt.savefig('n_class.png')
    plt.close()


def mean(p1, p2):
    """ Return the average value of the input points.
    Args:
        p1 (Numpy array): point [x, y]
        p2 (Numpy array): point [x, y]
    """
    return (p1 + p2) / 2.0


def difference(p1, p2):
    """ Return the difference value of the input points.
    Args:
        p1 (Numpy array): point [x, y]
        p2 (Numpy array): point [x, y]
    """
    return p2 - p1


def aspect_ratio(w, h):
    """ Return the width-to-height ratio.
    Args:
        w (int): image width
        h (int): image height
    """
    return w/float(h)


def plot_pairplot_asr():
    """ Plot correlation of xc, yc, w, h and distribution of aspect ratio. 
    File: `{subset}_{label}_correlation.png`
    File: `{subset}_{label}_aspect_ratio.png`
    """

    df = pd.read_csv('annotations.csv', encoding='utf-8-sig')

    # Step 2. Split by dataset
    for subset in [TRAIN_FOLDER, VALID_FOLDER]:
        sub_df = df[df['subset']==subset]
        # Step 3. Split by class
        for label in LABELS:
            class_df = sub_df[sub_df['class_name']==label]

            # Step 4. For each class, plot correlation of xc, yc, w, h
            sn.pairplot(class_df[['xc','yc','w', 'h']], 
                        corner=True, 
                        diag_kind='auto', 
                        kind='hist', 
                        diag_kws=dict(bins=10), 
                        plot_kws=dict(pmax=0.9))
            # plt.savefig('labels_correlogram.jpg', dpi=200)
            plt.title(label)
            plt.savefig(f'{subset}_{label}_correlation.png')
            plt.close()

            # Step 5. For each class, plot aspect-ratio distribution
            class_df.hist(column=["aspect_ratio"],
                            bins=50, grid=True)
            plt.title(label)
            plt.xlabel('Aspect ratio')
            plt.ylabel('Count')
            plt.savefig(f'{subset}_{label}_aspect_ratio.png')
            plt.close()


def plot_co_occurrence():
    """ Plot correlation of class. File: `{subset}_co_occurrence.png`"""

    df = pd.read_csv('annotations.csv', encoding='utf-8-sig')

    # Step 2. Split by dataset
    for subset in [TRAIN_FOLDER, VALID_FOLDER]:
        sub_df = df[df['subset']==subset]

        # Step 3. Create subplots
        fig, ax = plt.subplots(len(LABELS), len(LABELS), 
                                figsize = (20, 22), sharex=True, sharey=True)
        for i, row_name in enumerate(LABELS): # Row
            for j, col_name in enumerate(LABELS): # Column
                ax[i][j].set_xlim([0, 200])
                ax[i][j].set_ylim([0, 200])
                if i == 0:
                    ax[i][j].set_title(f"co-{col_name}")
                if j == 0:
                    ax[i][j].set_ylabel(row_name, fontsize=10, rotation=15, labelpad=30)
        
        # Step 4. Iterate each image, find different box name.
        for index, row in sub_df.iterrows():
            # Extract class name from file_name -> row index
            str_digits = row['file_name'].split('.')[0]
            row_name = ''.join([i for i in str_digits if not i.isdigit()])
            row_name = row_name.strip('_')
            row_idx = LABELS.index(row_name)
            # Take Box class name -> column index
            col_idx = LABELS.index(row['class_name'])
            # Step 5. Put point on corresponding scatter plot
            ax[row_idx][col_idx].plot(row['xc'], row['yc'], 'o', markersize=2, markeredgecolor='b')
        
        plt.savefig(f'{subset}_co_occurrence.png')
        plt.close()


def plot_anchor_shape():
    """ Plot the shape of each box. File: `{subset}_{label}_anchor.png`"""

    df = pd.read_csv('annotations.csv', encoding='utf-8-sig')

    # Step 2. Split by dataset
    for subset in [TRAIN_FOLDER, VALID_FOLDER]:
        sub_df = df[df['subset']==subset]

        # Step 3. Split by class
        for label in LABELS:
            class_df = sub_df[sub_df['class_name']==label]
            # Step 4. Create figure and axes
            fig, ax = plt.subplots()
            plt.xlim(-1, 200)
            plt.ylim(-1, 200)

            # Step 5. Iterate each image, shift the box to center of frame.
            for index, row in class_df.iterrows():
                x = 100 - row['w'] / 2.0
                y = 100 - row['h'] / 2.0
                # Step 6. Create a Rectangle patch
                rect = patches.Rectangle(xy=(x, y), width=row['w'], height=row['h'], 
                        linewidth=1, edgecolor='r', facecolor='none', alpha=0.3)

                # Add the patch to the Axes
                ax.add_patch(rect)

            plt.savefig(f'{subset}_{label}_anchor.png')
            plt.close()


def plot_brightness_histogram():
    """ Plot the grayscale histogram. File: `{subset}_{label}_brightness.png`"""

    df = pd.read_csv('annotations.csv', encoding='utf-8-sig')
    # Step 2. Split by dataset
    for subset in [TRAIN_FOLDER, VALID_FOLDER]:
        sub_df = df[df['subset']==subset]
        # Step 3. Split by class
        for label in LABELS:
            class_df = sub_df[sub_df['class_name']==label]

            img_stack = []
             # Iterate each annotation
            for index, row in class_df.iterrows():
                # Step 4. Read image
                img_fn = row['file_name'].replace('xml', 'jpg')
                str_digits = row['file_name'].split('.')[0]
                label_folder = ''.join([i for i in str_digits if not i.isdigit()])
                label_folder = label_folder.strip('_')
                fp = os.path.join(ROOT_FOLDER_PATH, subset, IMG_FOLDER, label_folder, img_fn)
                img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)

                # Step 5. Crop RoI
                x1 = int(row['xmin'])
                y1 = int(row['ymin'])
                x2 = int(row['xmax'])
                y2 = int(row['ymax'])
                img = img[y1:y2, x1:x2]

                # Stack image to channel
                img_stack.extend(img.ravel())
            
            # Step 6. Plot histogram
            plt.hist(img_stack, bins=256, range=[0,256], 
                    color='k', density=True, lw=0)
            plt.title(f'Grayscale Histogram of {label}')
            plt.grid()
            plt.savefig(f'{subset}_{label}_brightness.png')
            plt.close()


def plot_random_sample_img():
    """ Select 5 * 6 images and display. File: `sample.png`"""

    df = pd.read_csv('annotations.csv', encoding='utf-8-sig')

    # Step 2. Create subplots
    fig, ax = plt.subplots(5, len(LABELS), 
                            figsize = (10, 9), sharex=True, sharey=True)
    for i in range(5): # Row
        for j, col_name in enumerate(LABELS): # Column
            ax[i][j].set_xlim([0, 200])
            ax[i][j].set_ylim([0, 200])
            if i == 0:
                ax[i][j].set_title(f"{col_name}")

    # Step 3. Split by class
    for j, label in enumerate(LABELS):
        class_df = df[df['class_name']==label]
        # Step 4. Random select images
        list_indices = class_df.index.values.tolist()
        indices = random.sample(list_indices, k=5)

        for i, index in enumerate(indices):
            # Step 5. Read image
            img_fn = class_df['file_name'][index].replace('xml', 'jpg')
            str_digits = class_df['file_name'][index].split('.')[0]
            label_folder = ''.join([i for i in str_digits if not i.isdigit()])
            label_folder = label_folder.strip('_')
            fp = os.path.join(ROOT_FOLDER_PATH, class_df['subset'][index], IMG_FOLDER, label_folder, img_fn)
            img = cv2.imread(fp)

            # Step 6. Draw bounding boxes
            img_df = df[df['file_name'] == class_df['file_name'][index]]
            for _, row in img_df.iterrows():
                x1 = int(row['xmin'])
                y1 = int(row['ymin'])
                x2 = int(row['xmax'])
                y2 = int(row['ymax'])
                cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 0))
            # Step 7. Display
            ax[i][j].imshow(img)
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])
    
    plt.tight_layout()
    plt.savefig('sample.png')
    plt.close()


if __name__ == '__main__':
    # create_summary_csv()
    # plot_class_per_image()
    # plot_class_count()
    plot_pairplot_asr()
    # plot_co_occurrence()
    # plot_anchor_shape()
    # plot_brightness_histogram()
    # plot_random_sample_img()

    pass