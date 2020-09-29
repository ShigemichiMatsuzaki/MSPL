
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
    --data-tgt-train-list ./vision_datasets/greenhouse/train_cucumber_r.lst \
    --data-tgt-test-list ./vision_datasets/greenhouse/val_cucumber_r.lst \
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
#    --outsource camvid \
#    --os-model espdnetue \
#    --os-weights /tmp/runs/uest/model_espdnetue_greenhouse/s_2.0_res_480_uest_rgb_os_camvid_ue/20200702-010334/espdnetue_best.pth