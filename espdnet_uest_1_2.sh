mr=0.1
tgt_port=0.5

# Uncertainty
CUDA_VISIBLE_DEVICES=0 python uest_seg_multi_os.py \
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
    --data-tgt-train-list ./vision_datasets/greenhouse/train_greenhouse2.lst \
    --data-tgt-test-list ./vision_datasets/greenhouse/val_greenhouse2.lst \
    --batch-size 24 \
    --gpu 0 \
    --model espdnetue \
    --restore-from /tmp/runs/model_espdnetue_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_rgb/20200715-230509/espdnetue_2.0_480_best.pth \
    --runs-root /tmp/runs/uest \
    --epr 5 \
    --num-rounds 10 \
    --use-uncertainty true \
    --outsource1 greenhouse \
    --os-model1  espdnetue \
    --os-weights1 /tmp/runs/uest/model_espdnetue_greenhouse/s_2.0_res_480_uest_rgb_os_camvid_forest_cityscapes_ue/20200802-114205/espdnetue_best.pth

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
