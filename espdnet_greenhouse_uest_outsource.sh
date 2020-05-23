mr=0.1
tgt_port=0.5
#for mr in 0.1 0.3 0.5; do
#for tgt_port in 0.5; do
CUDA_VISIBLE_DEVICES=1 python uest_seg.py \
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
    --data-tgt-train-list ./vision_datasets/greenhouse/train_greenhouse_more_less.txt \
    --data-tgt-test-list ./vision_datasets/greenhouse/val_greenhouse.txt \
    --batch-size 32 \
    --gpu 0 \
    --model espdnetue \
    --restore-from /tmp/runs/results_segmentation/model_espdnet_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_rgb/20200513-204023/espdnet_2.0_480_best.pth \
    --runs-root /tmp/runs/results_segmentation/uest \
    --power 0.0 \
    --epr 20 \
    --num-rounds 5 \
    --init-src-port 0.01 \
    --max-src-port 0.01 \
    --init-tgt-port $tgt_port \
    --max-tgt-port 0.5 \
    --tgt-port-step 0.0 \
    --src-port-step 0.0 \
    --mr-weight-kld $mr \
    --use-nid true \
    --nid-bin 256 \
    --outsource camvid \
    --outsource-weights ./results_segmentation/model_espdnet_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_rgb_/20200420-095339/espdnet_2.0_480_best.pth
    #--use-depth true \
    #--trainable-fusion true
#  done
#done

