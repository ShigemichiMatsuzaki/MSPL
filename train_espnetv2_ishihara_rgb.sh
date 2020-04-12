# CUDA_VISIBLE_DEVICES=0 python train_segmentation.py \
#   --dataset ishihara \
#   --data-path ./vision_datasets/ishihara_rgbd/ \
#   --batch-size 30 \
#   --crop-size 400 304 \
#   --ignore-idx 255 \
#   --model espnetv2 \
#   --s 2.0 \
#   --lr 0.0005 \
#   --scheduler hybrid \
#   --clr-max 61 \
#   --epochs 500

CUDA_VISIBLE_DEVICES=0 python train_segmentation.py \
  --dataset ishihara \
  --data-path ./vision_datasets/ishihara_rgbd/ \
  --batch-size 22 \
  --crop-size 400 304 \
  --ignore-idx 255 \
  --model espdnet \
  --s 2.0 \
  --lr 0.0005 \
  --scheduler hybrid \
  --clr-max 61 \
  --use-depth true \
  --epochs 500

