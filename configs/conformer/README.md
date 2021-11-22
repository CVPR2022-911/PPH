# Conformer: Local Features Coupling Global Representations for Visual Recognition

## Introduction

<!-- [ALGORITHM] -->

```latex
@article{peng2021conformer,
      title={Conformer: Local Features Coupling Global Representations for Visual Recognition}, 
      author={Zhiliang Peng and Wei Huang and Shanzhi Gu and Lingxi Xie and Yaowei Wang and Jianbin Jiao and Qixiang Ye},
      journal={arXiv preprint arXiv:2105.03889},
      year={2021},
}
```

## Results and models

### Mask R-CNN

| Backbone | Pretrain    | Lr schd | Multi-scale crop     |   FP16   |Mem (GB) | Inf time (fps) | box AP | mask AP |  Config  |   Download  |
| :------: | :---------: | :-----: | :-------------------:| :------: |:------: | :------------: | :----: | :-----: | :------: |  :--------: |
|  Conformer-S-P32  | ImageNet-1K |    1x   |        no            |    no    |   7.6   |                |  42.7  |  39.3   | [config](./mask_rcnn_swin-t-p4-w7_fpn_1x_coco.py)             | [model](https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.pth)  &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937.log.json) |
