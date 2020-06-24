mr=0.1
tgt_port=0.5
#for mr in 0.1 0.3 0.5; do
#for tgt_port in 0.5; do

# Uncertainty + NID 
CUDA_VISIBLE_DEVICES=0 python uest_seg.py \
    --random-mirror \
    --test-scale 1.0 \
    --rm-prob \
    --test-flipping \
    --num-classes 5 \
    --learning-rate 0.00001 \
    --save /tmp/runs/results_segmentation/uest \
    --data-path ./vision_datasets/ \
    --data-src greenhouse \
    --data-src-list ./vision_datasets/camvid/train_camvid.txt \
    --data-tgt-train-list ./vision_datasets/greenhouse/train_greenhouse_more.txt \
    --data-tgt-test-list ./vision_datasets/greenhouse/val_greenhouse_more.txt \
    --batch-size 24 \
    --gpu 0 \
    --model espdnetue \
    --restore-from /tmp/runs/results_segmentation/model_espdnetue_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_rgb/20200610-133810/espdnetue_2.0_480_best.pth \
    --runs-root /tmp/runs/results_segmentation/uest \
    --epr 5 \
    --num-rounds 20 \
    --use-uncertainty true \
    --use-nid true \
    --nid-bin 64 \
    --use-depth true\
    --trainable-fusion true\
    --outsource camvid \
    --outsource-weights /tmp/runs/results_segmentation/model_espdnet_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_rgb/20200608-184240/espdnet_2.0_480_best.pth

# Uncertainty
CUDA_VISIBLE_DEVICES=0 python uest_seg.py \
    --random-mirror \
    --test-scale 1.0 \
    --rm-prob \
    --test-flipping \
    --num-classes 5 \
    --learning-rate 0.00001 \
    --save /tmp/runs/results_segmentation/uest \
    --data-path ./vision_datasets/ \
    --data-src greenhouse \
    --data-src-list ./vision_datasets/camvid/train_camvid.txt \
    --data-tgt-train-list ./vision_datasets/greenhouse/train_greenhouse_more.txt \
    --data-tgt-test-list ./vision_datasets/greenhouse/val_greenhouse_more.txt \
    --batch-size 24 \
    --gpu 0 \
    --model espdnetue \
    --restore-from /tmp/runs/results_segmentation/model_espdnetue_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_rgb/20200610-133810/espdnetue_5_2.0_480_best.pth \
    --runs-root /tmp/runs/results_segmentation/uest \
    --epr 5 \
    --num-rounds 20 \
    --use-uncertainty true \
    --use-depth true\
    --trainable-fusion true\
    --outsource camvid \
    --outsource-weights /tmp/runs/results_segmentation/model_espdnet_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_rgb/20200608-184240/espdnet_2.0_480_best.pth

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
