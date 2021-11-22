#!/usr/bin/env bash

#pip install mmcv-full==1.3.8 --no-cache-dir
#pip install -r requirements/build.txt
#pip install -v -e .

# dataset path
#ln -s /userhome/MSCOCO2017/annotations data/coco/annotations
#ln -s /userhome/MSCOCO2017/images data/coco/train2017
#ln -s /userhome/MSCOCO2017/images data/coco/val2017

export OMP_NUM_THREADS=1
GPU_NUM=8

# CONFIG="configs/conformer/cascade_mask_rcnn_conformer-s-p32_fpn_ms_3x_coco.py"
CONFIG="configs/pph/pph_conformer_small_p32_fpn_mstrain_crop_100tokens_3x_coco.py"

WORK_DIR='./work_dir/pph_conformer_small_p32_fpn_mstrain_crop_100tokens_3x_coco'


# python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port=50040 --use_env ./tools/test.py \
#   ${CONFIG} ${WORK_DIR}/latest.pth --launcher pytorch --eval bbox

python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port=50040 --use_env ./tools/train.py ${CONFIG} \
     --work-dir ${WORK_DIR} --gpus ${GPU_NUM}  --launcher pytorch
    
# --resume-from ${WORK_DIR}/epoch_22.pth
