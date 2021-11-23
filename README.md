# CVPR2022-911

## Introduction
The code includes training and inference procedures for **Progressive Proposal Highlight for Object Detection**.

## Installation
```
conda create -n pph python=3.7
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.8 --no-cache-dir
#pip install -r requirements/build.txt
pip install -v -e .
```

## Set Data Path
```
ln -s /userhome/MSCOCO2017/annotations data/coco/annotations
ln -s /userhome/MSCOCO2017/train2017 data/coco/train2017
ln -s /userhome/MSCOCO2017/val2017 data/coco/val2017
```

## Training and Evaluate on validation dataset 

Create a training and inference shell script contains following command.

```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
GPUS=8

WORK_DIR="./work_dir/pph_conformer_small_p32_fpn_mstrain_crop_100tokens_3x_coco"
CONFIG="configs/pph/pph_conformer_small_p32_fpn_mstrain_crop_100tokens_3x_coco.py"

python -m torch.distributed.launch --nproc_per_node=${GPUS} --master_port=50040 --use_env ./tools/train.py ${CONFIG} \
     --work-dir ${WORK_DIR} --gpus ${GPU_NUM}  --launcher pytorch
```
