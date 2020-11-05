train_list_1="train_greenhouse_more.lst"
val_list_1="val_greenhouse_more.lst"
train_list_2="train_greenhouse2_occluded_removed.lst"
val_list_2="val_greenhouse2.lst"
train_list_3="train_cucumber_r.lst"
val_list_3="val_cucumber_r.lst"
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

batch_size=64
# learning_rate=0.0001
learning_rate=0.0005

train_list_list=( $train_list_1 $train_list_2 $train_list_3 )
val_list_list=( $val_list_1 $val_list_2 $val_list_3 )

source_list=( $outsource1 $outsource2 $outsource3 )
model_list=( $os_model1 $os_model2 $os_model3 )
weights_list=( $os_weights1 $os_weights2 $os_weights3 )
class_weighting_list=( "flat" "flat" "normal" )

# Iterate over all target datasets
for ((j=0; j<3; j++)) do
# Greenhouse C : CV+CS+FR
CUDA_VISIBLE_DEVICES=0 python uest_seg_multi_os.py \
    --test-scale 1.0 \
    --test-flipping \
    --num-classes 5 \
    --learning-rate $learning_rate \
    --save /tmp/runs/uest/for-paper \
    --data-src greenhouse \
    --data-src-list ./vision_datasets/camvid/train_camvid.txt \
    --data-tgt-train-list ./vision_datasets/greenhouse/${train_list_list[j]} \
    --data-tgt-test-list ./vision_datasets/greenhouse/${val_list_list[j]} \
    --batch-size $batch_size \
    --gpu 0 \
    --model espdnetue \
    --class-weighting ${class_weighting_list[j]} \
    --restore-from $restore_from \
    --runs-root /tmp/runs/uest/for-paper \
    --epr 5 \
    --num-rounds 10 \
    --use-uncertainty true \
    --conservative-label true \
    --label-update 3 \
    --merge-label-policy all \
    --os-model1 $os_model1 \
    --outsource1 $outsource1 \
    --os-weights1 $os_weights1 \
    --os-model2 $os_model2 \
    --outsource2 $outsource2 \
    --os-weights2 $os_weights2 \
    --os-model3 $os_model3 \
    --outsource3 $outsource3 \
    --os-weights3 $os_weights3

# Greenhouse C : Double
for ((i=0; i<3; i++)) do
i_next=`expr $i + 1`
i_next=`expr $i_next % 3`
CUDA_VISIBLE_DEVICES=0 python uest_seg_multi_os.py \
    --test-scale 1.0 \
    --test-flipping \
    --num-classes 5 \
    --learning-rate $learning_rate \
    --save /tmp/runs/uest/for-paper \
    --data-src greenhouse \
    --data-src-list ./vision_datasets/camvid/train_camvid.txt \
    --data-tgt-train-list ./vision_datasets/greenhouse/${train_list_list[j]} \
    --data-tgt-test-list ./vision_datasets/greenhouse/${val_list_list[j]} \
    --batch-size $batch_size \
    --gpu 0 \
    --model espdnetue \
    --class-weighting ${class_weighting_list[j]} \
    --restore-from $restore_from \
    --runs-root /tmp/runs/uest/for-paper \
    --epr 5 \
    --num-rounds 10 \
    --use-uncertainty true \
    --conservative-label true \
    --label-update 3 \
    --merge-label-policy all \
    --os-model1 ${model_list[i]} \
    --outsource1 ${source_list[i]} \
    --os-weights1 ${weights_list[i]} \
    --os-model2 ${model_list[i_next]} \
    --outsource2 ${source_list[i_next]} \
    --os-weights2 ${weights_list[i_next]}
done

# Greenhouse C : Single
for ((i=0; i<r3; i++)) do
CUDA_VISIBLE_DEVICES=0 python uest_seg_multi_os.py \
    --test-scale 1.0 \
    --test-flipping \
    --num-classes 5 \
    --learning-rate $learning_rate \
    --save /tmp/runs/uest/for-paper \
    --data-src greenhouse \
    --data-src-list ./vision_datasets/camvid/train_camvid.txt \
    --data-tgt-train-list ./vision_datasets/greenhouse/${train_list_list[j]} \
    --data-tgt-test-list ./vision_datasets/greenhouse/${val_list_list[j]} \
    --batch-size $batch_size \
    --gpu 0 \
    --model espdnetue \
    --class-weighting ${class_weighting_list[j]} \
    --restore-from $restore_from \
    --runs-root /tmp/runs/uest/for-paper \
    --epr 5 \
    --num-rounds 10 \
    --use-uncertainty true \
    --conservative-label true \
    --label-update 3 \
    --merge-label-policy all \
    --os-model1  ${model_list[i]} \
    --outsource1 ${source_list[i]} \
    --os-weights1 ${weights_list[i]}
done

done