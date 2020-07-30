CUDA_VISIBLE_DEVICES=0 python train_segmentation.py \
  --dataset forest \
  --data-path ./vision_datasets/camvid/ \
  --savedir /tmp/runs/ \
  --batch-size 24 \
  --crop-size 480 288 \
  --ignore-idx 255 \
  --model espdnetue \
  --s 2.0 \
  --lr 0.009 \
  --scheduler hybrid \
  --clr-max 61 \
  --normalize true \
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
