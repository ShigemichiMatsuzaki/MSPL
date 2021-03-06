# Uncertainty + NID 
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
    --data-tgt-train-list ./vision_datasets/greenhouse/train_cucumber_r.lst \
    --data-tgt-test-list ./vision_datasets/greenhouse/val_cucumber_r.lst \
    --batch-size 24 \
    --gpu 0 \
    --model espdnetue \
    --restore-from /tmp/runs/model_espdnetue_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_rgb/20200630-111157/espdnetue_5_2.0_480_best.pth \
    --runs-root /tmp/runs/uest \
    --epr 5 \
    --num-rounds 15 \
    --use-uncertainty true \
    --outsource camvid \
    --outsource-weights /tmp/runs/model_espdnet_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_rgb/20200630-161402/espdnet_2.0_480_best.pth

# Uncertainty
#CUDA_VISIBLE_DEVICES=0 python uest_seg.py \
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
#   --data-tgt-train-list ./vision_datasets/greenhouse/train_cucumber_r.lst \
#   --data-tgt-test-list ./vision_datasets/greenhouse/val_cucumber_r.lst \
#    --batch-size 32 \
#    --gpu 0 \
#    --model espdnetue \
#    --restore-from /tmp/runs/results_segmentation/model_espdnetue_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_rgb/20200523-200621/espdnetue_2.0_480_best.pth \
#    --runs-root /tmp/runs/results_segmentation/uest \
#    --epr 5 \
#    --num-rounds 15 \
#    --use-uncertainty true \
#    --outsource camvid \
#    --outsource-weights ./results_segmentation/model_espdnet_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_rgb_/20200420-095339/espdnet_2.0_480_best.pth
