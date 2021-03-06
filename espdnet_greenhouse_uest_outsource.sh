mr=0.1
tgt_port=0.5
#for mr in 0.1 0.3 0.5; do
#for tgt_port in 0.5; do

# Uncertainty + NID 
#CUDA_VISIBLE_DEVICES=0 python uest_seg.py \
#    --random-mirror \
#    --test-scale 1.0 \
#    --rm-prob \
#    --test-flipping \
#    --num-classes 5 \
#    --learning-rate 0.00001 \
#    --save /tmp/runs/uest \
#    --data-path ./vision_datasets/ \
#    --data-src greenhouse \
#    --data-src-list ./vision_datasets/camvid/train_camvid.txt \
#    --data-tgt-train-list ./vision_datasets/greenhouse/train_greenhouse_more.txt \
#    --data-tgt-test-list ./vision_datasets/greenhouse/val_greenhouse_more.txt \
#    --batch-size 24 \
#    --gpu 0 \
#    --model espdnetue \
#    --restore-from /tmp/runs/model_espdnetue_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_rgb/20200630-111157/espdnetue_5_2.0_480_best.pth \
#    --runs-root /tmp/runs/uest \
#    --epr 5 \
#    --num-rounds 20 \
#    --use-uncertainty true \
#    --use-nid true \
#    --nid-bin 64 \
#    --outsource camvid \
#    --outsource-weights /tmp/runs/model_espdnet_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_rgb/20200630-161402/espdnet_2.0_480_best.pth

#    --use-depth true \
#    --trainable-fusion true \

# Uncertainty
CUDA_VISIBLE_DEVICES=0 python uest_seg.py \
    --random-mirror \
    --test-scale 1.0 \
    --rm-prob \
    --test-flipping \
    --num-classes 5 \
    --learning-rate 0.00001 \
    --save /tmp/runs/uest \
    --data-path ./vision_datasets/ \
    --data-src greenhouse \
    --data-src-list ./vision_datasets/camvid/train_camvid.txt \
    --data-tgt-train-list ./vision_datasets/greenhouse/train_greenhouse_more.lst \
    --data-tgt-test-list ./vision_datasets/greenhouse/val_greenhouse_more.lst \
    --batch-size 24 \
    --gpu 0 \
    --model espdnetue \
    --restore-from /tmp/runs/model_espdnetue_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_rgb/20200715-230509/espdnetue_2.0_480_best.pth \
    --runs-root /tmp/runs/uest \
    --epr 5 \
    --num-rounds 10 \
    --use-uncertainty true \
    --os-model espdnetue \
    --outsource forest \
    --os-weights /tmp/runs/model_espdnetue_forest/s_2.0_sch_hybrid_loss_ce_res_480_sc_1.0_1.0_rgb/20200729-181614/espdnetue_2.0_480_best.pth


#    --os-weights /tmp/runs/model_espdnetue_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_1.0_1.0_rgb/20200715-164256/espdnetue_2.0_480_best.pth
#    --outsource cityscapes \
#    --os-weights /tmp/runs/model_espdnetue_city/s_2.0_sch_hybrid_loss_ce_res_512_sc_0.25_0.5_rgb/20200717-161219/espdnetue_2.0_512_best.pth
#	model_espdnetue_city/s_2.0_sch_hybrid_loss_ce_res_512_sc_0.25_0.5_rgb/20200715-170836/espdnetue_2.0_512_best.pth \
#	model_espdnetue_city/s_2.0_sch_hybrid_loss_ce_res_512_sc_0.25_0.5_rgb/20200717-161219/espdnetue_2.0_512_best.pth \
#	model_deeplabv3_city/s_2.0_sch_hybrid_loss_ce_res_512_sc_0.25_0.5_rgb/20200714-172451/deeplabv3_2.0_512_best.pth \


##    --os-weights /tmp/runs/model_espdnet_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_1.0_1.0_rgb/20200715-165603/espdnet_2.0_480_best.pth
#    --os-weights /tmp/runs/model_espdnetue_city/s_2.0_sch_hybrid_loss_ce_res_512_sc_0.25_0.5_rgb/20200715-170836/espdnetue_2.0_512_best.pth
#    --os-weights /tmp/runs/model_espdnetue_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_1.0_1.0_rgb/20200715-164256/espdnetue_2.0_480_best.pth
#    --os-weights /tmp/runs/model_espdnetue_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_2.0_1.5_rgb/20200715-111451/espdnetue_2.0_480_best.pth
#    --os-weights /tmp/runs/model_espdnetue_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_rgb/20200714-204420/espdnetue_2.0_480_best.pth
#    --os-weights /tmp/runs/model_deeplabv3_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_rgb/20200713-150748/deeplabv3_2.0_480_best.pth
    

# --outsource-weights /tmp/runs/model_espdnet_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_rgb/20200630-161402/espdnet_2.0_480_best.pth
# --outsource-weights /tmp/runs/model_deeplabv3_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_rgb/20200710-185848/deeplabv3_2.0_480_best.pth
# --outsource-weights /tmp/runs/model_deeplabv3_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_rgb/20200711-141634/deeplabv3_2.0_480_best.pth
# CE + NID
#CUDA_VISIBLE_DEVICES=1 python uest_seg.py \
#    --random-mirror \
#    --test-scale 1.0 \
#    --rm-prob \
#    --test-flipping \
#    --num-classes 5 \
#    --learning-rate 0.00001 \
#    --save /tmp/runs/results_segmentation/uest \
#    --data-path ./vision_datasets/ \
#    --data-src greenhouse \
#    --data-src-list ./vision_datasets/camvid/train_camvid.txt \
#    --data-tgt-train-list ./vision_datasets/greenhouse/train_greenhouse_more.txt \
#    --data-tgt-test-list ./vision_datasets/greenhouse/val_greenhouse_more.txt \
#    --batch-size 32 \
#    --gpu 0 \
#    --model espdnetue \
#    --restore-from /tmp/runs/results_segmentation/model_espdnetue_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_rgb/20200603-141809/espdnetue_2.0_480_best.pth \
#    --runs-root /tmp/runs/results_segmentation/uest \
#    --epr 5 \
#    --num-rounds 20 \
#    --use-nid true \
#    --nid-bin 64 \
#    --outsource camvid \
#    --outsource-weights ./results_segmentation/model_espdnet_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_rgb_/20200420-095339/espdnet_2.0_480_best.pth
#
## CE
#CUDA_VISIBLE_DEVICES=1 python uest_seg.py \
#    --random-mirror \
#    --test-scale 1.0 \
#    --rm-prob \
#    --test-flipping \
#    --num-classes 5 \
#    --learning-rate 0.00001 \
#    --save /tmp/runs/results_segmentation/uest \
#    --data-path ./vision_datasets/ \
#    --data-src greenhouse \
#    --data-src-list ./vision_datasets/camvid/train_camvid.txt \
#    --data-tgt-train-list ./vision_datasets/greenhouse/train_greenhouse_more.txt \
#    --data-tgt-test-list ./vision_datasets/greenhouse/val_greenhouse_more.txt \
#    --batch-size 32 \
#    --gpu 0 \
#    --model espdnetue \
#    --restore-from /tmp/runs/results_segmentation/model_espdnetue_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_rgb/20200603-141809/espdnetue_2.0_480_best.pth \
#    --runs-root /tmp/runs/results_segmentation/uest \
#    --epr 5 \
#    --num-rounds 20 \
#    --outsource camvid \
#    --outsource-weights ./results_segmentation/model_espdnet_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_rgb_/20200420-095339/espdnet_2.0_480_best.pth
#
