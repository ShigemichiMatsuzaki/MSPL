CUDA_VISIBLE_DEVICES=0 python trav_mask_train.py \
    --num-classes 5 \
    --learning-rate 0.0005 \
    --save /tmp/runs/uest/for-paper/trav \
    --data-trav-list ./vision_datasets/traversability_mask/greenhouse_b_train.lst \
    --optimizer Adam \
    --batch-size 64 \
    --gpu 0 \
    --model espdnetue \
    --restore-from /tmp/runs/uest/for-paper/trav/model_espdnetue_greenhouse/s_2.0_res_480_uest_rgb_os_camvid_cityscapes_forest_ue/20201204-152737/espdnetue_ep_1.pth \
    --epoch 500 \
    --use-uncertainty true

CUDA_VISIBLE_DEVICES=0 python trav_mask_train.py \
    --num-classes 5 \
    --learning-rate 0.0005 \
    --save /tmp/runs/uest/for-paper/trav \
    --data-trav-list ./vision_datasets/traversability_mask/greenhouse_b_train.lst \
    --optimizer SGD \
    --batch-size 64 \
    --gpu 0 \
    --model espdnetue \
    --restore-from /tmp/runs/uest/for-paper/trav/model_espdnetue_greenhouse/s_2.0_res_480_uest_rgb_os_camvid_cityscapes_forest_ue/20201204-152737/espdnetue_ep_1.pth \
    --epoch 500 \
    --use-uncertainty true