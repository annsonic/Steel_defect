import torch
from utils.boxs_utils import box_iou
from losses.commons import focal_loss, IOULoss, BoxSimilarity

INF = 1e8


class BoxCoder(object):
    def __init__(self, weights=None):
        super(BoxCoder, self).__init__()
        if weights is None:
            weights = [0.1, 0.1, 0.2, 0.2]
        self.weights = torch.tensor(data=weights, requires_grad=False)

    def encoder(self, anchors, gt_boxes):
        """
        :param gt_boxes:[box_num, 4]
        :param anchors: [box_num, 4]
        :return:
        """
        if self.weights.device != anchors.device:
            self.weights = self.weights.to(anchors.device)
        anchors_wh = anchors[..., [2, 3]] - anchors[..., [0, 1]]
        anchors_xy = anchors[..., [0, 1]] + 0.5 * anchors_wh
        gt_wh = (gt_boxes[..., [2, 3]] - gt_boxes[..., [0, 1]]).clamp(min=1.0)
        gt_xy = gt_boxes[..., [0, 1]] + 0.5 * gt_wh
        delta_xy = (gt_xy - anchors_xy) / anchors_wh
        delta_wh = (gt_wh / anchors_wh).log()

        delta_targets = torch.cat([delta_xy, delta_wh], dim=-1) / self.weights

        return delta_targets

    def decoder(self, predicts, anchors):
        """
        :param predicts: [anchor_num, 4] or [bs, anchor_num, 4]
        :param anchors: [anchor_num, 4]
        :return: [anchor_num, 4] (x1,y1,x2,y2)
        """
        if self.weights.device != anchors.device:
            self.weights = self.weights.to(anchors.device)
        anchors_wh = anchors[:, [2, 3]] - anchors[:, [0, 1]]
        anchors_xy = anchors[:, [0, 1]] + 0.5 * anchors_wh
        scale_reg = predicts * self.weights
        scale_reg[..., :2] = anchors_xy + scale_reg[..., :2] * anchors_wh
        scale_reg[..., 2:] = scale_reg[..., 2:].exp() * anchors_wh
        scale_reg[..., :2] -= (0.5 * scale_reg[..., 2:])
        scale_reg[..., 2:] = scale_reg[..., :2] + scale_reg[..., 2:]

        return scale_reg


class ATSSMatcher(object):
    def __init__(self, top_k, anchor_num_per_loc):
        self.top_k = top_k
        self.anchor_num_per_loc = anchor_num_per_loc

    def __call__(self, anchors, gt_boxes, num_anchor_per_layer):
        """
        :param anchors:
        :param targets:
        :param num_anchor_per_layer:
        :return:
        """
        ret_list = list()
        anchor_xy = (anchors[:, :2] + anchors[:, 2:]) / 2.0
        for bid, gt in enumerate(gt_boxes):
            if len(gt) == 0:
                continue
            start_idx = 0
            candidate_idxs = list()
            gt_xy = (gt[:, [1, 2]] + gt[:, [3, 4]]) / 2.0
            distances = (anchor_xy[:, None, :] - gt_xy[None, :, :]).pow(2).sum(-1).sqrt()
            anchor_gt_iou = box_iou(anchors, gt[:, 1:])
            for num_anchor in num_anchor_per_layer:
                distances_per_level = distances[start_idx:start_idx + num_anchor]
                top_k = min(self.top_k * self.anchor_num_per_loc, num_anchor)
                _, topk_idxs_per_level = distances_per_level.topk(top_k, dim=0, largest=False)
                candidate_idxs.append(topk_idxs_per_level + start_idx)
                start_idx += num_anchor
            candidate_idxs = torch.cat(candidate_idxs, dim=0)
            candidate_ious = anchor_gt_iou.gather(dim=0, index=candidate_idxs)
            iou_mean_per_gt = candidate_ious.mean(0)
            iou_std_per_gt = candidate_ious.std(0)
            iou_thresh_per_gt = iou_mean_per_gt + iou_std_per_gt
            is_pos = candidate_ious >= iou_thresh_per_gt[None, :]
            candidate_xy = anchor_xy[candidate_idxs]
            lt = candidate_xy - gt[None, :, [1, 2]]
            rb = gt[None, :, [3, 4]] - candidate_xy
            is_in_gts = torch.cat([lt, rb], dim=-1).min(-1)[0] > 0.01
            is_pos = is_pos & is_in_gts
            gt_idx = torch.arange(len(gt))[None, :].repeat((len(candidate_idxs), 1))
            match = torch.full_like(anchor_gt_iou, fill_value=-INF)
            match[candidate_idxs[is_pos], gt_idx[is_pos]] = anchor_gt_iou[candidate_idxs[is_pos], gt_idx[is_pos]]
            val, match_gt_idx = match.max(dim=1)
            match_gt_idx[val == -INF] = -1
            ret_list.append((bid, match_gt_idx))
        return ret_list


class ATSSRetinaLoss(object):
    def __init__(self, top_k,
                 anchor_num_per_loc,
                 alpha=0.25,
                 gamma=2.0,
                 iou_type="giou",
                 iou_loss_type="centerness",
                 iou_loss_weight=0.5,
                 reg_loss_weight=1.0):
        self.top_k = top_k
        self.alpha = alpha
        self.gamma = gamma
        self.iou_type = iou_type
        self.iou_loss_type = iou_loss_type
        self.iou_loss_weight = iou_loss_weight
        self.reg_loss_weight = reg_loss_weight
        self.matcher = ATSSMatcher(self.top_k, anchor_num_per_loc)
        self.box_coder = BoxCoder()
        self.iou_loss = IOULoss(iou_type=iou_type)
        self.box_similarity = BoxSimilarity(iou_type="iou")
        self.bce = torch.nn.BCEWithLogitsLoss(reduction="sum")

    @staticmethod
    def build_centerness(reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                     (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def __call__(self, cls_predicts, reg_predicts, iou_predicts, anchors, targets):
        """
        :param cls_predicts:
        :param reg_predicts:
        :param iou_predicts:
        :param anchors:
        :param targets:
        :return:
        """
        num_anchors_per_level = [len(item) for item in anchors]
        cls_predicts = torch.cat([item for item in cls_predicts], dim=1)
        reg_predicts = torch.cat([item for item in reg_predicts], dim=1)
        all_anchors = torch.cat([item for item in anchors])
        iou_predicts = torch.cat([item for item in iou_predicts], dim=1)
        gt_boxes = targets['target'].split(targets['batch_len'])
        matches = self.matcher(all_anchors, gt_boxes, num_anchors_per_level)
        match_bidx = list()
        match_anchor_idx = list()
        match_gt_idx = list()
        for bid, match in matches:
            anchor_idx = (match >= 0).nonzero(as_tuple=False).squeeze(-1)
            match_anchor_idx.append(anchor_idx)
            match_gt_idx.append(match[anchor_idx])
            match_bidx.append(bid)
        if cls_predicts.dtype == torch.float16:
            cls_predicts = cls_predicts.float()
        if iou_predicts.dtype == torch.float16:
            iou_predicts = iou_predicts.float()
        cls_batch_idx = sum([[i] * len(j) for i, j in zip(match_bidx, match_anchor_idx)], [])
        cls_anchor_idx = torch.cat(match_anchor_idx)
        cls_label_idx = torch.cat([gt_boxes[i][:, 0][j].long() for i, j in zip(match_bidx, match_gt_idx)])
        num_pos = len(cls_batch_idx)
        cls_targets = torch.zeros_like(cls_predicts)
        cls_targets[cls_batch_idx, cls_anchor_idx, cls_label_idx] = 1.0
        all_cls_loss = focal_loss(cls_predicts.sigmoid(), cls_targets, alpha=self.alpha,
                                  gamma=self.gamma).sum() / num_pos
        all_box_targets = torch.cat([gt_boxes[i][:, 1:][j] for i, j in zip(match_bidx, match_gt_idx)], dim=0)
        all_box_predicts = self.box_coder.decoder(reg_predicts[cls_batch_idx, cls_anchor_idx],
                                                  all_anchors[cls_anchor_idx])
        if self.iou_loss_type == "centerness":
            anchor_xy = (all_anchors[cls_anchor_idx][:, [0, 1]] + all_anchors[cls_anchor_idx][:, [2, 3]]) / 2.0
            reg_targets = torch.cat([anchor_xy - all_box_targets[:, [0, 1]], all_box_targets[:, [2, 3]] - anchor_xy],
                                    dim=-1)
            all_iou_targets = self.build_centerness(reg_targets)
        elif self.iou_loss_type == "iou":
            all_iou_targets = self.box_similarity(all_box_predicts.detach(), all_box_targets)
        else:
            raise NotImplementedError("iou_loss_type: {:s} is not support now".format(self.iou_loss_type))
        all_iou_loss = self.iou_loss_weight * self.bce(
            iou_predicts[cls_batch_idx, cls_anchor_idx, 0], all_iou_targets) / num_pos
        all_box_loss = self.reg_loss_weight * (
                self.iou_loss(all_box_predicts, all_box_targets) * all_iou_targets).sum() / (all_iou_targets.sum())

        return all_cls_loss, all_box_loss, all_iou_loss, num_pos
