from ..builder import DETECTORS
from .two_stage import TwoStageDetector
from mmdet.core import (bbox2roi, bbox_mapping, merge_aug_bboxes, merge_aug_bboxes_sparse,
                        merge_aug_masks, multiclass_nms, multiclass_nms_sparse)
from mmdet.core import bbox2result, bbox2roi, bbox_xyxy_to_cxcywh

@DETECTORS.register_module()
class ProgressiveProposalHighlight(TwoStageDetector):

    def __init__(self, *args, **kwargs):
        super(ProgressiveProposalHighlight, self).__init__(*args, **kwargs)
        
    def extract_feat(self, img):
        x = self.backbone(img)
        xs, proposal_tokens = x[0], x[1]
        if self.with_neck:
            x = self.neck(xs)
        return x, proposal_tokens
    
        def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        
        x, proposal_tokens = self.extract_feat(img)
        proposal_boxes, proposal_features, imgs_whwh = \
            self.rpn_head.forward_train(x, img_metas, proposal_features=proposal_tokens)
        roi_losses = self.roi_head.forward_train(
            x[0],
            proposal_boxes,
            proposal_features,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore=gt_bboxes_ignore,
            imgs_whwh=imgs_whwh)
        return roi_losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img)
        if len(x) == 2:
#             proposal_boxes, proposal_features, imgs_whwh = \
#             proposal_boxes, proposal_features, imgs_whwh, mask_features = \
            proposal_boxes, proposal_features, imgs_whwh = \
                self.rpn_head.simple_test_rpn(x[0], img_metas, proposal_features=x[1])
            bbox_results = self.roi_head.simple_test(
                x[0],
                proposal_boxes,
                proposal_features,
                img_metas,
                imgs_whwh=imgs_whwh,
                rescale=rescale,)
#                 mask_features=mask_features)
            return bbox_results
        elif len(x) == 3:
#             proposal_boxes, proposal_features, imgs_whwh, mask_features = \
            proposal_boxes, proposal_features, imgs_whwh = \
                self.rpn_head.forward_train(x[0], img_metas, proposal_features=x[1])
            bbox_results = self.roi_head.simple_test(
                x[0],
                proposal_boxes,
                proposal_features,
                img_metas,
                imgs_whwh=imgs_whwh,
                rescale=rescale,
#                 mask_features=mask_features,
                attns_maps=x[2]
            )
            return bbox_results
        
        else:
            x = self.extract_feat(img)
            proposal_boxes, proposal_features, imgs_whwh = \
                self.rpn_head.simple_test_rpn(x, img_metas)
            bbox_results = self.roi_head.simple_test(
                x,
                proposal_boxes,
                proposal_features,
                img_metas,
                imgs_whwh=imgs_whwh,
                rescale=rescale)
            return bbox_results
        
    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        aug_bboxes = []
        aug_scores = []
        
        xs = self.extract_feats(imgs)
        for x, img_meta in zip(xs, img_metas):
            proposal_boxes, proposal_features, imgs_whwh = \
                self.rpn_head.forward_train(x[0], img_meta, proposal_features=x[1])
            bboxes, scores = self.roi_head.aug_test(
                x[0],
                proposal_boxes,
                proposal_features,
                img_meta,
                imgs_whwh=imgs_whwh,
                rescale=False,
            )            
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)
        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes_sparse(
            aug_bboxes, aug_scores, img_metas, None)
        det_bboxes, det_labels = multiclass_nms_sparse(merged_bboxes, merged_scores,
                                                0.05,
                                                dict(type='soft_nms', iou_threshold=0.65),
                                                max_num=100)
        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   80)
        return [bbox_results]
        

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        # backbone
        x = self.extract_feat(img)
        # rpn
        num_imgs = len(img)
        dummy_img_metas = [
            dict(img_shape=(800, 1333, 3)) for _ in range(num_imgs)
        ]
        proposal_boxes, proposal_features, imgs_whwh = \
            self.rpn_head.simple_test_rpn(x, dummy_img_metas)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposal_boxes,
                                               proposal_features,
                                               dummy_img_metas)
        return roi_outs