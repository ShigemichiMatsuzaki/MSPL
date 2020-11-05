CUDA_VISIBLE_DEVICES=0 python train_segmentation.py \
  --dataset city \
  --data-path ./vision_datasets/cityscapes/ \
  --savedir /tmp/runs/ \
  --batch-size 8 \
  --crop-size 512 256 \
  --ignore-idx 255 \
  --model unet \
  --s 2.0 \
  --lr 0.003 \
  --scheduler hybrid \
  --clr-max 61 \
  --normalize true \
  --use-aux true \
  --epochs 500
  #--label-conversion true \

# CUDA_VISIBLE_DEVICES=0 python train_segmentation.py \
#   --dataset greenhouse \
#   --data-path ./vision_datasets/greenhouse/ \
#   --batch-size 30 \
#   --crop-size 480 256 \
#   --ignore-idx 4 \
#   --model espnetv2 \
#   --s 2.0 \
#   --lr 0.009 \
#   --scheduler hybrid \
#   --clr-max 61 \
#   --epochs 500
