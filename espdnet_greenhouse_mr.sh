CUDA_VISIBLE_DEVICES=0 python crst_seg.py \
    --random-mirror \
    --test-scale 1.0 \
    --rm-prob \
    --test-flipping \
    --num-classes 5 \
    --learning-rate 0.00001 \
    --save ./results_segmentation/ \
    --data-path ./vision_datasets/ \
    --data-src greenhouse \
    --data-src-list ./vision_datasets/greenhouse/train_greenhouse_gt.txt \
    --data-tgt-train-list ./vision_datasets/greenhouse/train_greenhouse_more.txt \
    --data-tgt-test-list ./vision_datasets/greenhouse/val_greenhouse.txt \
    --batch-size 20 \
    --gpu 0 \
    --model espdnet \
    --restore-from ./results_segmentation/model_espdnet_greenhouse/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0/20200417-093057/espdnet_2.0_480_best.pth \
    --runs-root ./results_segmentation \
    --power 0.0 \
    --num-rounds 8 \
    --init-src-port 1.0 \
    --max-src-port 1.0 \
    --init-tgt-port 0.5 \
    --max-tgt-port 1.0 \
    --tgt-port-step 0.0 \
    --src-port-step 0.0 \
    --mr-weight-kld 0.4 \
    --trainable-fusion true \
    --use-depth true
