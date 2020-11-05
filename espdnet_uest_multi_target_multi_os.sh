train_list="train_1_2.lst"
val_list="val_greenhouse2.lst"
restore_from="/tmp/runs/model_espdnetue_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_rgb/20200715-230509/espdnetue_2.0_480_best.pth"
outsource1=camvid 
outsource2=cityscapes
outsource3=forest
os_model1=espdnetue 
os_model2=espdnetue 
os_model3=espdnetue 
os_weights1="/tmp/runs/model_espdnetue_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_1.0_1.0_rgb/20200715-164256/espdnetue_2.0_480_best.pth"
os_weights2="/tmp/runs/model_espdnetue_city/s_2.0_sch_hybrid_loss_ce_res_512_sc_0.25_0.5_rgb/20200717-161219/espdnetue_2.0_512_best.pth"
os_weights3="/tmp/runs/model_espdnetue_forest/s_2.0_sch_hybrid_loss_ce_res_480_sc_1.0_1.0_rgb/20200729-181614/espdnetue_2.0_480_best.pth"

# Single source
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
    --data-tgt-train-list ./vision_datasets/greenhouse/$train_list \
    --data-tgt-test-list ./vision_datasets/greenhouse/$val_list \
    --batch-size 24 \
    --gpu 0 \
    --model espdnetue \
    --restore-from /tmp/runs/model_espdnetue_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_rgb/20200715-230509/espdnetue_2.0_480_best.pth \
    --runs-root /tmp/runs/uest \
    --epr 5 \
    --num-rounds 10 \
    --use-uncertainty true \
    --conservative-label true \
    --label-update 3 \
    --merge-label-policy all \
    --os-model1 $os_model1\
    --outsource1 $outsource1\
    --os-weights1 $os_weights1
#    --os-model3 espdnetue \
#    --outsource3 forest \
#    --os-weights3 /tmp/runs/model_espdnetue_forest/s_2.0_sch_hybrid_loss_ce_res_480_sc_1.0_1.0_rgb/20200729-181614/espdnetue_2.0_480_best.pth \

# Multi-source (Camvid-Cityscapes)
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
    --data-tgt-train-list ./vision_datasets/greenhouse/$train_list \
    --data-tgt-test-list ./vision_datasets/greenhouse/$val_list \
    --batch-size 24 \
    --gpu 0 \
    --model espdnetue \
    --restore-from $restore_from \
    --runs-root /tmp/runs/uest \
    --epr 5 \
    --num-rounds 10 \
    --use-uncertainty true \
    --conservative-label true \
    --label-update 3 \
    --merge-label-policy all \
    --os-model1 $os_model1 \
    --outsource1 $outsource1\
    --os-weights1 $os_weights1 \
    --os-model2 $os_model2 \
    --outsource2 $outsource2 \
    --os-weights2 $os_weights2
#    --os-model3 espdnetue \
#    --outsource3 forest \
#    --os-weights3 /tmp/runs/model_espdnetue_forest/s_2.0_sch_hybrid_loss_ce_res_480_sc_1.0_1.0_rgb/20200729-181614/espdnetue_2.0_480_best.pth \

# Multi-source (Camvid-Cityscapes-forest)
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
    --data-tgt-train-list ./vision_datasets/greenhouse/$train_list \
    --data-tgt-test-list ./vision_datasets/greenhouse/$val_list \
    --batch-size 24 \
    --gpu 0 \
    --model espdnetue \
    --restore-from $restore_from \
    --runs-root /tmp/runs/uest \
    --epr 5 \
    --num-rounds 10 \
    --use-uncertainty true \
    --conservative-label true \
    --label-update 3 \
    --merge-label-policy all \
    --os-model1 $os_model1 \
    --outsource1 $outsource1\
    --os-weights1 $os_weights1 \
    --os-model2 $os_model2 \
    --outsource2 $outsource2 \
    --os-weights2 $os_weights2 \
    --os-model3 $os_model3 \
    --outsource3 $outsource3 \
    --os-weights3 $os_weights3




#    --os-model1 deeplabv3 \
#    --outsource1 camvid \
#    --os-weights1 /tmp/runs/model_deeplabv3_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_rgb/20200713-150748/deeplabv3_2.0_480_best.pth \
#    --os-model2 deeplabv3 \
#    --outsource2 cityscapes \
#    --os-weights2 /tmp/runs/model_deeplabv3_city/s_2.0_sch_hybrid_loss_ce_res_512_sc_0.25_0.5_rgb/20200714-172451/deeplabv3_2.0_512_best.pth




#    --os-weights1 /tmp/runs/model_unet_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_1.0_1.0_rgb/20200728-201827/unet_2.0_480_best.pth \
#    --outsource cityscapes \
#    --os-weights /tmp/runs/model_espdnetue_city/s_2.0_sch_hybrid_loss_ce_res_512_sc_0.25_0.5_rgb/20200717-161219/espdnetue_2.0_512_best.pth
#	model_espdnetue_city/s_2.0_sch_hybrid_loss_ce_res_512_sc_0.25_0.5_rgb/20200715-170836/espdnetue_2.0_512_best.pth \
#	model_espdnetue_city/s_2.0_sch_hybrid_loss_ce_res_512_sc_0.25_0.5_rgb/20200717-161219/espdnetue_2.0_512_best.pth \
#	model_deeplabv3_city/s_2.0_sch_hybrid_loss_ce_res_512_sc_0.25_0.5_rgb/20200714-172451/deeplabv3_2.0_512_best.pth \


##    --os-weights /tmp/runs/model_espdnet_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_1.0_1.0_rgb/20200715-165603/espdnet_2.0_480_best.pth
#    --os-weights /tmp/runs/model_espdnetue_city/s_2.0_sch_hybrid_loss_ce_res_512_sc_0.25_0.5_rgb/20200715-170836/espdnetue_2.0_512_best.pth
#    --os-weights /tmp/runs/model_espdnetue_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_1.0_1.0_rgb/20200715-164256/espdnetue_2.0_480_best.pth
#    --os-weights /tmp/runs/model_espdnetue_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_2.0_1.5_rgb/20200715-111451/espdnetue_2.0_480_best.pth
#    --os-weights /tmp/runs/model_espdnetue_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_rgb/20200714-204420/espdnetue_2.0_480_best.pth
#    --os-weights /tmp/runs/model_deeplabv3_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_rgb/20200713-150748/deeplabv3_2.0_480_best.pth
    

# --outsource-weights /tmp/runs/model_espdnet_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_rgb/20200630-161402/espdnet_2.0_480_best.pth
# --outsource-weights /tmp/runs/model_deeplabv3_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_rgb/20200710-185848/deeplabv3_2.0_480_best.pth
# --outsource-weights /tmp/runs/model_deeplabv3_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_rgb/20200711-141634/deeplabv3_2.0_480_best.pth