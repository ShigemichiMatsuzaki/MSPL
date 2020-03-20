CUDA_VISIBLE_DEVICES=0 python train_segmentation.py \
  --dataset greenhouse \
  --data-path ./vision_datasets/greenhouse/ \
  --batch-size 30 \
  --crop-size 480 256 \
  --ignore-idx 4 \
  --model espdnet \
  --s 2.0 \
  --lr 0.009 \
  --scheduler hybrid \
  --clr-max 61 \
  --use-depth true \
  --epochs 500

#CUDA_VISIBLE_DEVICES=0 python train_segmentation.py \
#  --dataset greenhouse \
#  --data-path ./vision_datasets/greenhouse/ \
#  --batch-size 30 \
#  --crop-size 480 256 \
#  --ignore-idx 4 \
#  --model espnetv2 \
#  --s 2.0 \
#  --lr 0.009 \
#  --scheduler hybrid \
#  --clr-max 61 \
#  --epochs 150
#
