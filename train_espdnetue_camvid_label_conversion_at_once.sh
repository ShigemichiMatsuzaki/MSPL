CUDA_VISIBLE_DEVICES=1 python train_segmentation_label_conversion.py \
  --dataset camvid \
  --data-path ./vision_datasets/camvid/ \
  --savedir /tmp/runs/results_segmentation/ \
  --batch-size 20 \
  --crop-size 480 288 \
  --model espdnetue \
  --s 2.0 \
  --lr 0.009 \
  --scheduler hybrid \
  --clr-max 61 \
  --epochs 200 \
  --label-conversion true


#  --finetune true \
#  --weights /tmp/runs/results_segmentation/model_espdnetue_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_rgb/20200609-221003/espdnetue_2.0_480_best.pth \
#CUDA_VISIBLE_DEVICES=1 python train_segmentation_label_conversion.py \
#  --dataset camvid \
#  --data-path ./vision_datasets/camvid/ \
#  --savedir /tmp/runs/results_segmentation/ \
#  --batch-size 20 \
#  --crop-size 480 288 \
#  --model espdnet \
#  --s 2.0 \
#  --lr 0.009 \
#  --scheduler hybrid \
#  --clr-max 61 \
#  --epochs 200 \
#  --label-conversion true
#
#CUDA_VISIBLE_DEVICES=1 python train_segmentation_label_conversion.py \
#  --dataset camvid \
#  --data-path ./vision_datasets/camvid/ \
#  --savedir /tmp/runs/results_segmentation/ \
#  --batch-size 20 \
#  --crop-size 480 288 \
#  --model espnetv2 \
#  --s 2.0 \
#  --lr 0.009 \
#  --scheduler hybrid \
#  --clr-max 61 \
#  --epochs 200 \
#  --label-conversion true

  #--weights /tmp/runs/results_segmentation/model_espdnet_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_rgb/20200513-172834/espdnet_2.0_480_best.pth \

