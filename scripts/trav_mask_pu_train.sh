for lr in 0.0005 0.0003 0.0001 0.00005 0.00003 0.00001; do
CUDA_VISIBLE_DEVICES=0 python trav_mask_pu_train.py \
    --num-classes 5 \
    --learning-rate $lr \
    --save /tmp/runs/uest/trav \
    --data-train-list ./vision_datasets/traversability_mask/greenhouse_b_train.lst \
    --data-test-list ./vision_datasets/traversability_mask/greenhouse_b_val.lst \
    --optimizer SGD \
    --batch-size 64 \
    --gpu 0 \
    --model espdnetue \
    --restore-from /tmp/runs/uest/for-paper/trav/model_espdnetue_greenhouse/s_2.0_res_480_uest_rgb_os_camvid_cityscapes_forest_ue/20201204-152737/espdnetue_ep_1.pth \
    --epoch 200 \
    --use-uncertainty true
done