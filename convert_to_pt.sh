restore_from="/tmp/runs/uest/for-paper/trav/model_espdnetue_greenhouse/s_2.0_res_480_uest_rgb_os_camvid_cityscapes_forest_ue/20201204-152737/espdnetue_ep_1.pth"
trav_module_weights="/tmp/runs/uest/trav/model_espdnetue_greenhouse/20210115-145605/espdnetue_best.pth"

CUDA_VISIBLE_DEVICES=0 python convert_to_pt.py \
    --num-classes 5 \
    --model-name espdnetue_trav \
    --weights $restore_from \
    --trav-module-weights $trav_module_weights \
    --use-uncertainty true
 