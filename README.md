## Introduction
This project train object detection models to detect the defects on the hot-rolled steel surface. 
The models are trained and evaluated on `NEU-DET` dataset.
I practice training the `YOLOv5` and `RetinaNet` model, with the following techniques: 
anchor optimization, data augmentation (ElasticTransform, GridMask), label-smoothing and Adaptive Training Sample Selection.

From my result, I noticed that:
1. If I union the boxes of crazing class, the AP would improve a lot.
I suppose the labeling rule of crazing class should adopt a large bounding box because there 
are not many distict boundaries between the crazes in `NEU-DET` images.

2. The AP score diverges: patches class is always doing well; rolled-in scale class is below average. The sooner the strong classes are recognized well in the early training period, the smaller score the weak classes will get. I have tried 
* Label Smoothing
* Focal loss
* More data augmentation
* Cosine learning rate

but couldn't improve weak classes' score.
## Performance
There are six types of surface defects in the NEU-DET dataset: crazing (Cr), inclusion (In), patches (Pa),  pitted surface (Ps), rolled-in scale (Rs), and scratches (Sc). 
The dataset includes 300 grayscale samples in each class of surface defects, splitted in 80:20 ratio.
### Table-1 AP@.5 validation results of different models on NEU-DET dataset.
NMS conf-thres=0.001, NMS iou-thres=0.5
|Types|[1]SSD300|[1]YOLO-V3*| YOLOv5s(A)| YOLOv5s(B)|RetinaNet(ATSS)|
|:------:|:------:|:-------:|:---------:|:---------:|:---------:| 
| Cr     | 0.411  |  0.389  |   0.527   |   0.960   |   0.470   | 
| In     | 0.796  |  0.737  |   0.787   |   0.611   |   0.795   | 
| Pa     | 0.839  |  0.935  |   0.945   |   0.882   |   0.952   | 
| Ps     | 0.839  |  0.748  |   0.854   |   0.908   |   0.852   | 
| Rs     | 0.621  |  0.607  |   0.581   |   0.521   |   0.550   | 
| Sc     | 0.836  |  0.914  |   0.805   |   0.656   |   0.569   | 
| mAP    | 0.714  |  0.722  |   0.750   |   0.756   |   0.698   | 
| FPS    |  37.6  |   64.5  |     -     |     -     |     -     |

※ My experiment with RetinaNet(ATSS) is not finished yet. 
The hyperparameters need to evolve well. It takes time ... 
### Table-2 My Training details
|      Type         | YOLOv5s(A)| YOLOv5s(B)|RetinaNet(ATSS)|
|:-----------------:|:---------:|:---------:|:---------:|
|Input image size   |    320    |    320    |    320    |
|Backbone           |CSPdarknet+SPP|CSPdarknet+SPP|Resnet-18|
|label-smoothing    |           |     √     |           |
|Modified labeling  |           |('Cr', 'In', 'Ps', 'Rs')|           |
|Anchor optimization|  Kmeans   |   Kmeans  |differential evolution|
|Cosine learning rate|          |     √     |     √     |
|Warming UP         | 2 epochs  | 2 epochs  | 2 epochs  |
|Image weighting    |           |     √     |           |
|IOU GIOU DIOU CIOU |     √     |     √     |     √     |
|Focal Loss         |           |           |     √     |
|MedianBlur         |           |     √     |     √     |
|RandomBrightnessContrast|           |     √     |           |
|RandomGamma        |           |     √     |           |
|ImageCompression   |           |     √     |           |
|RandomRotate90     |           |     √     |           |
|GaussNoise         |           |     √     |     √     |
|MultiplicativeNoise|           |     √     |           |
|ElasticTransform   |           |     √     |           |
|GridDistortion     |           |     √     |           |
|OpticalDistortion  |           |     √     |           |
|GridMask           |           |     √     |           |
|Mosaic Augment     |     √     |     √     |     √     |
|Color Jitter       |           |           |     √     |
|Perspective Transform|     √     |     √     |     √     |
|Translation Transform|     √     |     √     |           |
|Scaling Transform  |     √     |     √     |     √     |
|Shearing Transform |     √     |     √     |           |
|Rotation           |     √     |     √     |           |
|Vertical Flip      |     √     |     √     |           |
|Horizontal Flip    |     √     |     √     |     √     |
|RandomHSV        |           |           |     √     |
|RandCrop         |           |           |     √     |


## Installation
Python>=3.7.0 and PyTorch>=1.7
### Folders
```
- Base folder
	- anchor-optimization (Optimize anchor of RetinaNet)
	- atssv1 (RetinaNet with ATSS project)
	- utils (Some data format converter)
		- clean_labels.py (union near boxes, only support VOC format)
		- statistics.py (plot statics of box annotations)
		- voc2coc.py (convert VOC to COCO style)
		- voc2csv.py (convert VOC to keras-retina style)
		- voc2yolo.py (convert VOC to YOLO style)
	- yolov5 (YOLOv5 project)
```
### Setup YOLOv5 (PyTorch project)
```
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```

### Setup RetinaNet(ATSS) (PyTorch project)
```
pip install cython
pip install 'git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI'
```

### Setup anchor-optimization for RetinaNet (TensorFlow/Keras project)

Note: The dependency keras-retina is imcompatible with TensorFlow 2.8.0, Keras 2.8.0 and Numpy 1.21.5.
It works with keras 2.4 and tensorflow 2.3.0.

1. `git clone https://github.com/martinzlocha/anchor-optimization.git`

2. `cd anchor-optimization`

3. `pip install .`

4. `python setup.py build_ext --inplace`

## Usage

### YOLOv5 part
* Convert dataset format (VOC to YOLO)
1. Modify `ROOT_FOLDER_PATH` and `COPY_IMG` variable in `voc2yolo.py`
2. run `voc2yolo.py`
3. generated `NEU-DET_YOLO` folder will be under your `ROOT_FOLDER_PATH`

* Anchor optimization
1. modify `anchor_t` in hyp.neu.yaml
2. run tain.py with `--img-size <int>` argument

* Train
1. `cd yolov5`
2.

```
python train.py \
--weights 'yolov5s.pt' \
--cfg './models/yolov5s.yaml' \
--data './data/neu.yaml' \
--hyp './data/hyp.neu.yaml' \
--epochs 90 \
--batch-size 256 \
--img 320 \
--cache \
--image-weights \
--label-smoothing 0.1 \
--cos-lr \
--device '0' \
--project <path for your result> \
--exist-ok 
```

* Evaluate
1. `cd yolov5`
2. 

```
python val.py \
--weights <your ckpt file path> \
--data './data/neu.yaml' \
--img 320 \
--task 'val' \
--conf-thres 0.3 \
--iou-thres 0.2 \
--device '0' \
--project <path for your result> \
--save-txt \
--save-conf \
--verbose 
```

### RetinaNet part
* Convert dataset format (VOC to COCO)

1. Modify `ROOT_FOLDER_PATH` and `COPY_IMG` variable in `voc2yolo.py`
2. run `voc2yolo.py`
3. generated `NEU-DET_COCO` folder will be under your `ROOT_FOLDER_PATH`

* Anchor optimization

1. `anchor-optimization <your csv file> --image-min-side <training image size> `

&emsp;&emsp;Note: you should put the csv file and the class_id text file in the image folder of your dataset.

2. Put the displayed `anchor_scales` and `anchor_ratios` values to your cfg yaml file.

&emsp;&emsp;Note: the cfg yaml file template is `atssv1/config/neu.yaml`.

* Train

1. `cd atssv1`
2. modify `main.py` (modify `cfg_path`) and your cfg yaml file.
3. `python main.py`
4. displaying mAP results, for example:<br>
```
epoch: 0|match_num:165|loss:2.2716|cls:1.1108|box:0.7987|iou:0.3622
```

* Evaluate

1. `cd atssv1`
2. modify `eval.py` (modify `cfg_path`) and your cfg yaml file (modify `ckpt_name`).
3. `python eval.py`
4. displaying mAP results, for example:<br>
```
Start evaluating...
100% 90/90 [03:31<00:00,  2.35s/it]
	Single classid=0, ap50: 0.25902219641001
	Single classid=1, ap50: 0.7291185018137647
	Single classid=2, ap50: 0.9126426816465034
	Single classid=3, ap50: 0.7710457516339869
	Single classid=4, ap50: 0.4761086432442042
	Single classid=5, ap50: 0.47500818551404606
******************** eval start ********************
epoch:  2|mp:56.9954|mr:70.1405|map50:60.3824|map:27.5001
******************** eval end ********************
```

## Reference

[1] Xupeng Kou, Shuaijun Liu, Kaiqiang Cheng, Ye Qian, Development of a YOLO-V3-based model for detecting defects on steel strip surface, Measurement, Volume 182, 2021, 109454, ISSN 0263-2241, https://doi.org/10.1016/j.measurement.2021.109454.

## Acknowledgement

[1] Anchor optimization for RetinaNet [link](https://github.com/martinzlocha/anchor-optimization/)

[2] ATSS_RetinaNet [link](https://github.com/liangheming/atssv1)

[3] YOLOv5 [link](https://github.com/ultralytics/yolov5)