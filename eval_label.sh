#!/bin/bash
# ESPNet-CamVid
path_list_esp_cv=(\
	model_espdnetue_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_rgb/20200630-111157/espdnetue_2.0_480_best.pth \
	model_espdnetue_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_rgb/20200714-204420/espdnetue_2.0_480_best.pth \
	model_espdnetue_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_2.0_1.5_rgb/20200715-111451/espdnetue_2.0_480_best.pth \
	model_espdnetue_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_1.0_1.0_rgb/20200715-164256/espdnetue_2.0_480_best.pth \
	model_espdnetue_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_rgb/20200715-230509/espdnetue_2.0_480_best.pth \
)
# ESPNet-Cityscapes
path_list_esp_cs=(\
	model_espdnetue_city/s_2.0_sch_hybrid_loss_ce_res_512_sc_0.25_0.5_rgb/20200715-170836/espdnetue_2.0_512_best.pth \
	model_espdnetue_city/s_2.0_sch_hybrid_loss_ce_res_512_sc_0.25_0.5_rgb/20200717-161219/espdnetue_2.0_512_best.pth \
)
# ESPNet-Freiburg-Forest
path_list_esp_ff=(\
  model_espdnetue_forest/s_2.0_sch_hybrid_loss_ce_res_480_sc_1.0_1.0_rgb/20200729-181614/espdnetue_2.0_480_best.pth	\
)
# Deeplab-CamVid
path_list_dl_cv=(\
	model_deeplabv3_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_rgb/20200710-185848/deeplabv3_2.0_480_best.pth \ 
	model_deeplabv3_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_rgb/20200710-191117/deeplabv3_2.0_480_best.pth \
	model_deeplabv3_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_rgb/20200711-141634/deeplabv3_2.0_480_best.pth \
	model_deeplabv3_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_rgb/20200713-150748/deeplabv3_2.0_480_best.pth \
)
# Deeplab-Cityscapes
path_list_dl_cs=(\
	model_deeplabv3_city/s_2.0_sch_hybrid_loss_ce_res_512_sc_0.25_0.5_rgb/20200714-172451/deeplabv3_2.0_512_best.pth \
)

for path in "${path_list_esp_ff[@]}"; 
do
CUDA_VISIBLE_DEVICES=0 python eval_label.py \
  --data-path ./vision_datasets/camvid/ \
  --savedir /tmp/runs/eval_label \
  --batch-size 24 \
  --crop-size 480 256 \
  --dataset forest \
  --model espdnetue \
  --finetune /tmp/runs/$path
done


  #--model espdnetue \
  #--finetune /tmp/runs/model_espdnetue_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_2.0_1.5_rgb/20200715-111451/espdnetue_2.0_480_best.pth \
  #--model espdnetue \
  #--finetune /tmp/runs/model_espdnetue_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_rgb/20200715-230509/espdnetue_2.0_480_best.pth \
  #--model deeplabv3 \
  #--finetune /tmp/runs/model_deeplabv3_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_rgb/20200713-150748/deeplabv3_2.0_480_best.pth \

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
