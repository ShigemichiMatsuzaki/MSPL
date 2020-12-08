restore_from="/tmp/runs/uest/for-paper/trav/model_espdnetue_greenhouse/20201205-204050/espdnetue_best.pth"

CUDA_VISIBLE_DEVICES=0 python convert_to_pt.py \
    --num-classes 5 \
    --model espdnetue \
    --weights $restore_from \
    --use-uncertainty true
 