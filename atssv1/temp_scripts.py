import torch
from nets.retinanet import RetinaNet
from datasets.coco import COCODataSets
from torch.utils.data.dataloader import DataLoader

if __name__ == '__main__':
    dataset = COCODataSets(img_root="/home/huffman/data/val2017",
                           annotation_path="/home/huffman/data/annotations/instances_val2017.json",
                           use_crowd=True,
                           augments=True,
                           remove_blank=True,
                           max_thresh=640
                           )
    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=1, collate_fn=dataset.collect_fn)
    net = RetinaNet(**{"dist_train": False})
    for img_input, targets, batch_len in dataloader:
        ret = net(img_input, targets={"target": targets, "batch_len": batch_len})
        print(ret)
        break
