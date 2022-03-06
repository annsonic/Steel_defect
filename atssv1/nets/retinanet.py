import torch
import math
from nets import resnet
from torch import nn
from nets.common import FPN, CGR, CBR
from losses.atss_retina import BoxCoder, ATSSRetinaLoss
from torchvision.ops.boxes import nms


def non_max_suppression(prediction: torch.Tensor,
                        conf_thresh=0.05,
                        iou_thresh=0.5,
                        max_det=300,
                        max_box=2048,
                        max_layer_num=1000
                        ):
    """
    :param max_layer_num:
    :param prediction:
    :param conf_thresh:
    :param iou_thresh:
    :param max_det:
    :param max_box:
    :return: (x1,y1,x2,y2,score,cls_id)
    """
    for i in range(len(prediction)):
        if prediction[i].dtype == torch.float16:
            prediction[i] = prediction[i].float()
    bs = prediction[0].shape[0]
    out = [None] * bs
    for bi in range(bs):
        batch_predicts_list = [torch.zeros(size=(0, 6), device=prediction[0].device).float()] * len(prediction)
        for lj in range(len(prediction)):
            one_layer_bath_predict = prediction[lj][bi]
            reg_predicts = one_layer_bath_predict[:, :4]
            cls_predicts = one_layer_bath_predict[:, 4:].sigmoid()
            center_predicts = cls_predicts[:, 0]
            cls_predicts = cls_predicts[:, 1:]

            max_val, max_idx = cls_predicts.max(dim=1)
            valid_bool_idx = max_val > conf_thresh
            if valid_bool_idx.sum() == 0:
                continue
            valid_val = max_val[valid_bool_idx] * center_predicts[valid_bool_idx]
            sorted_idx = valid_val.argsort(descending=True)
            valid_val = valid_val[sorted_idx]
            valid_box = reg_predicts[valid_bool_idx, :][sorted_idx]
            valid_cls = max_idx[valid_bool_idx][sorted_idx]
            if 0 < max_layer_num < valid_box.shape[0]:
                valid_val = valid_val[:max_layer_num]
                valid_box = valid_box[:max_layer_num, :]
                valid_cls = valid_cls[:max_layer_num]
            batch_predicts = torch.cat([valid_box, valid_val[:, None], valid_cls[:, None]], dim=-1)
            batch_predicts_list[lj] = batch_predicts
        x = torch.cat(batch_predicts_list, dim=0)
        if x.shape[0] == 0:
            continue
        c = x[:, 5:6] * max_box
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = nms(boxes, scores, iou_thresh)
        if i.shape[0] > max_det:
            i = i[:max_det]
        out[bi] = x[i]
    return out


class Scale(nn.Module):
    def __init__(self, init_val=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(data=init_val), requires_grad=True)

    def forward(self, x):
        return x * self.scale


class SequenceCNR(nn.Module):
    def __init__(self,
                 in_channel,
                 inner_channel,
                 kennel_size=3,
                 stride=1,
                 num=4,
                 padding=None,
                 bias=True,
                 block_type='CGR'):
        super(SequenceCNR, self).__init__()
        self.bones = list()
        for i in range(num):
            if i == 0:
                block = eval(block_type)(in_channel, inner_channel, kennel_size, stride, padding=padding, bias=bias)
            else:
                block = eval(block_type)(inner_channel, inner_channel, kennel_size, stride, padding=padding, bias=bias)
            self.bones.append(block)
        self.bones = nn.Sequential(*self.bones)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.bones(x)


class RetinaClsHead(nn.Module):
    def __init__(self,
                 in_channel=256,
                 num_anchors=9, num_cls=80):
        super(RetinaClsHead, self).__init__()
        self.num_anchors = num_anchors
        self.num_cls = num_cls
        self.cls = nn.Conv2d(in_channel, self.num_anchors * self.num_cls, 3, 1, 1)
        nn.init.normal_(self.cls.weight, std=0.01)
        nn.init.constant_(self.cls.bias, -math.log((1 - 0.01) / 0.01))

    def forward(self, x):
        x = self.cls(x)
        bs, _, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous() \
            .view(bs, h, w, self.num_anchors, self.num_cls) \
            .view(bs, -1, self.num_cls)
        return x


class RetinaRegHead(nn.Module):
    def __init__(self, in_channel=256, num_anchors=9):
        super(RetinaRegHead, self).__init__()
        self.num_anchors = num_anchors
        self.reg = nn.Conv2d(in_channel, self.num_anchors * 4, 3, 1, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.reg(x)
        bs, _, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous() \
            .view(bs, h, w, self.num_anchors, 4) \
            .view(x.size(0), -1, 4)
        return x


class IOUPredHead(nn.Module):
    def __init__(self, in_channel=256, num_anchors=9):
        super(IOUPredHead, self).__init__()
        self.num_anchors = num_anchors
        self.iou_head = nn.Conv2d(in_channel, self.num_anchors, 3, 1, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.iou_head(x)
        bs, _, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous() \
            .view(bs, h, w, self.num_anchors, 1) \
            .view(x.size(0), -1, 1)
        return x


class RetinaHead(nn.Module):
    def __init__(self, in_channel,
                 inner_channel,
                 anchor_sizes,
                 anchor_scales,
                 anchor_ratios,
                 strides,
                 num_cls=80,
                 num_convs=4,
                 layer_num=5,
                 block_type="CGR",
                 on_reg=True):
        super(RetinaHead, self).__init__()
        self.num_cls = num_cls
        self.layer_num = layer_num
        self.anchor_sizes = anchor_sizes
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.strides = strides
        self.on_reg = on_reg

        self.anchor_nums = len(self.anchor_scales) * len(self.anchor_ratios)
        self.scales = nn.ModuleList([Scale(init_val=1.0) for _ in range(self.layer_num)])
        self.anchors = [torch.zeros(size=(0, 4))] * self.layer_num
        self.box_coder = BoxCoder()
        self.cls_bones = SequenceCNR(in_channel, inner_channel,
                                     kennel_size=3, stride=1,
                                     num=num_convs, block_type=block_type)
        self.reg_bones = SequenceCNR(in_channel, inner_channel,
                                     kennel_size=3, stride=1,
                                     num=num_convs, block_type=block_type)
        self.cls_head = RetinaClsHead(inner_channel, self.anchor_nums, num_cls)
        self.reg_head = RetinaRegHead(inner_channel, self.anchor_nums)
        self.iou_head = IOUPredHead(inner_channel, self.anchor_nums)

    def build_anchors_delta(self, size=32.):
        """
        :param size:
        :return: [anchor_num, 4]
        """
        scales = torch.tensor(self.anchor_scales).float()
        ratio = torch.tensor(self.anchor_ratios).float()
        scale_size = (scales * size)
        w = (scale_size[:, None] * ratio[None, :].sqrt()).view(-1) / 2
        h = (scale_size[:, None] / ratio[None, :].sqrt()).view(-1) / 2
        delta = torch.stack([-w, -h, w, h], dim=1)
        return delta

    def build_anchors(self, feature_maps):
        """
        :param feature_maps:
        :return: list(anchor) anchor:[all,4] (x1,y1,x2,y2)
        """
        assert self.layer_num == len(feature_maps)
        assert len(self.anchor_sizes) == len(feature_maps)
        assert len(self.anchor_sizes) == len(self.strides)
        anchors = list()
        for stride, size, feature_map in zip(self.strides, self.anchor_sizes, feature_maps):
            # 9*4
            anchor_delta = self.build_anchors_delta(size)
            _, _, ny, nx = feature_map.shape
            yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
            # h,w,4
            grid = torch.stack([xv, yv, xv, yv], 2).float()
            anchor = (grid[:, :, None, :] + 0.5) * stride + anchor_delta[None, None, :, :]
            anchor = anchor.view(-1, 4)
            anchors.append(anchor)
        return anchors

    def forward(self, xs):
        cls_outputs = list()
        reg_outputs = list()
        iou_outputs = list()
        for j, x in enumerate(xs):
            cls_tower = self.cls_bones(x)
            reg_tower = self.reg_bones(x)
            cls_outputs.append(self.cls_head(cls_tower))
            reg_outputs.append(self.scales[j](self.reg_head(reg_tower)))
            if self.on_reg:
                iou_outputs.append(self.iou_head(reg_tower))
            else:
                iou_outputs.append(self.iou_head(cls_tower))
        if self.anchors[0] is None or self.anchors[0].shape[0] != cls_outputs[0].shape[1]:
            with torch.no_grad():
                anchors = self.build_anchors(xs)
                assert len(anchors) == len(self.anchors)
                for i, anchor in enumerate(anchors):
                    self.anchors[i] = anchor.to(xs[0].device)
        if self.training:
            return cls_outputs, reg_outputs, iou_outputs, self.anchors
        else:
            predicts_list = list()
            for cls_out, reg_out, iou_out, anchor in zip(cls_outputs, reg_outputs, iou_outputs, self.anchors):
                scale_reg = self.box_coder.decoder(reg_out, anchor)
                predicts_out = torch.cat([scale_reg, iou_out, cls_out], dim=-1)
                predicts_list.append(predicts_out)
            return predicts_list


default_cfg = {
    "num_cls": 80,
    "anchor_sizes": [32., 64., 128., 256., 512.],
    "anchor_scales": [2 ** 0, ],
    "anchor_ratios": [1., ],
    "strides": [8, 16, 32, 64, 128],
    "backbone": "resnet18",
    "pretrained": True,
    "fpn_channel": 256,
    "head_conv_num": 4,
    "block_type": "CGR",
    "on_reg": True,
    # loss
    "top_k": 9,
    "iou_loss_weight": 0.5,
    "reg_loss_weight": 1.3,
    "alpha": 0.25,
    "gamma": 2.0,
    "iou_type": "giou",
    "iou_loss_type": "centerness",
    # predicts
    "conf_thresh": 0.01,
    "nms_iou_thresh": 0.5,
    "max_det": 300,

}


class RetinaNet(nn.Module):
    def __init__(self, **kwargs):
        self.cfg = {**default_cfg, **kwargs}
        super(RetinaNet, self).__init__()
        self.backbones = getattr(resnet, self.cfg['backbone'])(pretrained=self.cfg['pretrained'])
        c3, c4, c5 = self.backbones.inner_channels
        self.neck = FPN(c3, c4, c5, self.cfg['fpn_channel'])
        self.head = RetinaHead(in_channel=self.cfg['fpn_channel'],
                               inner_channel=self.cfg['fpn_channel'],
                               num_cls=self.cfg['num_cls'],
                               num_convs=self.cfg['head_conv_num'],
                               layer_num=5,
                               anchor_sizes=self.cfg['anchor_sizes'],
                               anchor_scales=self.cfg['anchor_scales'],
                               anchor_ratios=self.cfg['anchor_ratios'],
                               strides=self.cfg['strides'],
                               block_type=self.cfg['block_type'],
                               on_reg=self.cfg['on_reg'])
        self.loss = ATSSRetinaLoss(
            top_k=self.cfg['top_k'],
            alpha=self.cfg['alpha'],
            gamma=self.cfg['gamma'],
            iou_loss_weight=self.cfg['iou_loss_weight'],
            reg_loss_weight=self.cfg['reg_loss_weight'],
            iou_type=self.cfg['iou_type'],
            iou_loss_type=self.cfg['iou_loss_type'],
            anchor_num_per_loc=self.head.anchor_nums
        )

    def forward(self, x, targets=None):
        c3, c4, c5 = self.backbones(x)
        p3, p4, p5, p6, p7 = self.neck([c3, c4, c5])
        out = self.head([p3, p4, p5, p6, p7])
        ret = dict()
        if self.training:
            assert targets is not None
            cls_outputs, reg_outputs, iou_outputs, anchors = out
            all_cls_loss, all_box_loss, all_iou_loss, num_pos = self.loss(
                cls_outputs, reg_outputs, iou_outputs, anchors, targets)
            ret['cls_loss'] = all_cls_loss
            ret['box_loss'] = all_box_loss
            ret['iou_loss'] = all_iou_loss
            ret['match_num'] = num_pos
        else:
            _, _, h, w = x.shape
            for pred in out:
                pred[:, [0, 2]] = pred[:, [0, 2]].clamp(min=0, max=w)
                pred[:, [1, 3]] = pred[:, [0, 2]].clamp(min=0, max=h)
            predicts = non_max_suppression(out,
                                           conf_thresh=self.cfg['conf_thresh'],
                                           iou_thresh=self.cfg['nms_iou_thresh'],
                                           max_det=self.cfg['max_det']
                                           )
            ret['predicts'] = predicts
        return ret

# if __name__ == '__main__':
#     input_tensor = torch.rand(size=(4, 3, 640, 640)).float()
#     net = RetinaNet(backbone="resnet18")
#     mcls_output, mreg_output, manchor = net(input_tensor, 1)
#     for cls_out, reg_out, anchor_out in zip(mcls_output, mreg_output, manchor):
#         print(cls_out.shape, reg_out.shape, anchor_out.shape)
# # out = net(input_tensor)
# for item in out:
#     print(item.shape)
