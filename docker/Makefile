NAME=pytorch-edgenets
#VERSION=torch1.2.0-py3.6
VERSION=20.03-py3
CONTAINER_NAME=pytorch-edgenets

build : 
	docker build -t $(NAME):$(VERSION) .

restart: stop start

start:
	docker start $(CONTAINER_NAME)

contener=`docker ps -a -q`
image=`docker images | awk '/^<none>/ { print $$3 }'`
	
clean:
	@if [ "$(image)" != "" ] ; then \
		docker rmi $(image); \
	fi
	@if [ "$(contener)" != "" ] ; then \
		docker rm $(contener); \
	fi
	
stop:
	docker stop $(CONTAINER_NAME)
	
rm:
	docker rm -f $(CONTAINER_NAME)
attach:
	docker start $(CONTAINER_NAME) && docker exec -it $(CONTAINER_NAME) /bin/bash
	
logs:
	docker logs $(CONTAINER_NAME)

#
# Execusion commands
#
train-espdnet-camvid:
	docker run -it \
    		--gpus="device=0" \
    		-v /home/aisl/matsuzaki/docker/edgenets-docker/espdnet/:/root/espdnet/ \
    		-v /data/aisl/matsuzaki/dataset:/tmp/dataset \
    		-v /data/aisl/matsuzaki/runs/:/tmp/runs/ \
		--rm \
		--shm-size 5G \
		--workdir /root/espdnet/ \
		--name train-espdnet-camvid \
		$(NAME):$(VERSION) \
    sh /root/espdnet/train_espdnet_camvid.sh

train-espdnetue-greenhouse:
	docker run -it \
    		--gpus="device=2" \
    		-v /home/aisl/matsuzaki/docker/edgenets-docker/espdnet/:/root/espdnet/ \
    		-v /data/aisl/matsuzaki/dataset:/tmp/dataset \
    		-v /data/aisl/matsuzaki/runs/:/tmp/runs/ \
		--rm \
		--shm-size 5G \
		--workdir /root/espdnet/ \
		--name train-espdnetue-greenhouse \
		$(NAME):$(VERSION) \
    sh /root/espdnet/train_espdnetue_greenhouse.sh

train-espdnetue-greenhouse-2:
	docker run -it \
    		--gpus="device=2" \
    		-v /home/aisl/matsuzaki/docker/edgenets-docker/espdnet/:/root/espdnet/ \
    		-v /data/aisl/matsuzaki/dataset:/tmp/dataset \
    		-v /data/aisl/matsuzaki/runs/:/tmp/runs/ \
		--rm \
		--shm-size 5G \
		--workdir /root/espdnet/ \
		--name train-espdnetue-greenhouse-2 \
		$(NAME):$(VERSION) \
    sh /root/espdnet/train_espdnetue_greenhouse.sh

train-deeplabv3-camvid:
	docker run -it \
    		--gpus="device=2" \
    		-v /home/aisl/matsuzaki/docker/edgenets-docker/espdnet/:/root/espdnet/ \
    		-v /data/aisl/matsuzaki/dataset:/tmp/dataset \
    		-v /data/aisl/matsuzaki/runs/:/tmp/runs/ \
		--rm \
		--shm-size 5G \
		--workdir /root/espdnet/ \
		--name train-deeplabv3-camvid \
		$(NAME):$(VERSION) \
    sh /root/espdnet/train_deeplabv3_camvid.sh

train-deeplabv3-camvid-2:
	docker run -it \
    		--gpus="device=2" \
    		-v /home/aisl/matsuzaki/docker/edgenets-docker/espdnet/:/root/espdnet/ \
    		-v /data/aisl/matsuzaki/dataset:/tmp/dataset \
    		-v /data/aisl/matsuzaki/runs/:/tmp/runs/ \
		--rm \
		--shm-size 5G \
		--workdir /root/espdnet/ \
		--name train-deeplabv3-camvid-2 \
		$(NAME):$(VERSION) \
    sh /root/espdnet/train_deeplabv3_camvid.sh

train-deeplabv3-cityscapes:
	docker run -it \
    		--gpus="device=2" \
    		-v /home/aisl/matsuzaki/docker/edgenets-docker/espdnet/:/root/espdnet/ \
    		-v /data/aisl/matsuzaki/dataset:/tmp/dataset \
    		-v /data/aisl/matsuzaki/runs/:/tmp/runs/ \
		--rm \
		--shm-size 5G \
		--workdir /root/espdnet/ \
		--name train-deeplabv3-cityscapes \
		$(NAME):$(VERSION) \
    sh /root/espdnet/train_deeplabv3_cityscapes.sh

train-espdnetue-camvid:
	docker run -it \
    		--gpus="device=0" \
    		-v /home/aisl/matsuzaki/docker/edgenets-docker/espdnet/:/root/espdnet/ \
    		-v /data/aisl/matsuzaki/dataset:/tmp/dataset \
    		-v /data/aisl/matsuzaki/runs/:/tmp/runs/ \
		--rm \
		--shm-size 5G \
		--workdir /root/espdnet/ \
		--name train-espdnetue-camvid \
		$(NAME):$(VERSION) \
    sh /root/espdnet/train_espdnetue_camvid.sh

train-espdnetue-forest:
	docker run -it \
    		--gpus="device=0" \
    		-v /home/aisl/matsuzaki/docker/edgenets-docker/espdnet/:/root/espdnet/ \
    		-v /data/aisl/matsuzaki/dataset:/tmp/dataset \
    		-v /data/aisl/matsuzaki/runs/:/tmp/runs/ \
		--rm \
		--shm-size 5G \
		--workdir /root/espdnet/ \
		--name train-espdnetue-forest \
		$(NAME):$(VERSION) \
    sh /root/espdnet/train_espdnetue_forest.sh

eval-label:
	docker run -it \
    		--gpus="device=2" \
    		-v /home/aisl/matsuzaki/docker/edgenets-docker/espdnet/:/root/espdnet/ \
    		-v /data/aisl/matsuzaki/dataset:/tmp/dataset \
    		-v /data/aisl/matsuzaki/runs/:/tmp/runs/ \
		--rm \
		--shm-size 5G \
		--workdir /root/espdnet/ \
		--name label-eval \
		$(NAME):$(VERSION) \
    bash /root/espdnet/eval_label.sh

train-espdnetue-camvid-label-conv:
	docker run -it \
    		--gpus="device=2" \
    		-v /home/aisl/matsuzaki/docker/edgenets-docker/espdnet/:/root/espdnet/ \
    		-v /data/aisl/matsuzaki/dataset:/tmp/dataset \
    		-v /data/aisl/matsuzaki/runs/:/tmp/runs/ \
		--rm \
		--shm-size 5G \
		--workdir /root/espdnet/ \
		--name train-espdnetue-camvid-label-conv \
		$(NAME):$(VERSION) \
    sh /root/espdnet/train_espdnetue_camvid_label_conversion_at_once.sh

train-espdnetue-cityscapes:
	docker run -it \
    		--gpus="device=2" \
    		-v /home/aisl/matsuzaki/docker/edgenets-docker/espdnet/:/root/espdnet/ \
    		-v /data/aisl/matsuzaki/dataset:/tmp/dataset \
    		-v /data/aisl/matsuzaki/runs/:/tmp/runs/ \
		--rm \
		--shm-size 5G \
		--workdir /root/espdnet/ \
		--name train-espdnetue-cityscapes \
		$(NAME):$(VERSION) \
    sh /root/espdnet/train_espdnetue_city.sh

train-unet-camvid:
	docker run -it \
    		--gpus="device=2" \
    		-v /home/aisl/matsuzaki/docker/edgenets-docker/espdnet/:/root/espdnet/ \
    		-v /data/aisl/matsuzaki/dataset:/tmp/dataset \
    		-v /data/aisl/matsuzaki/runs/:/tmp/runs/ \
		--rm \
		--shm-size 5G \
		--workdir /root/espdnet/ \
		--name train-unet-camvid \
		$(NAME):$(VERSION) \
    sh /root/espdnet/train_unet_camvid.sh

train-unet-cityscapes:
	docker run -it \
    		--gpus="device=2" \
    		-v /home/aisl/matsuzaki/docker/edgenets-docker/espdnet/:/root/espdnet/ \
    		-v /data/aisl/matsuzaki/dataset:/tmp/dataset \
    		-v /data/aisl/matsuzaki/runs/:/tmp/runs/ \
		--rm \
		--shm-size 5G \
		--workdir /root/espdnet/ \
		--name train-unet-cityscapes \
		$(NAME):$(VERSION) \
    sh /root/espdnet/train_unet_city.sh


train-espdnet-uest-greenhouse-multi-os:
	docker run -it \
    	--gpus="device=2" \
    	-v /home/aisl/matsuzaki/docker/edgenets-docker/espdnet/:/root/espdnet/ \
    	-v /data/aisl/matsuzaki/dataset:/tmp/dataset \
    	-v /data/aisl/matsuzaki/runs/:/tmp/runs/ \
		--rm \
		--shm-size 5G \
		--workdir /root/espdnet/ \
		--name train-espdnet-uest-greenhouse-multi-os \
		$(NAME):$(VERSION) \
        bash /root/espdnet/espdnet_greenhouse_uest_multi_os.sh

train-espdnet-uest-greenhouse-multi-os-2:
	docker run -it \
    	--gpus="device=2" \
    	-v /home/aisl/matsuzaki/docker/edgenets-docker/espdnet/:/root/espdnet/ \
    	-v /data/aisl/matsuzaki/dataset:/tmp/dataset \
    	-v /data/aisl/matsuzaki/runs/:/tmp/runs/ \
		--rm \
		--shm-size 5G \
		--workdir /root/espdnet/ \
		--name train-espdnet-uest-greenhouse-multi-os-2 \
		$(NAME):$(VERSION) \
        sh /root/espdnet/espdnet_greenhouse_uest_multi_os.sh


train-espdnet-uest-greenhouse-outsource:
	docker run -it \
    	--gpus="device=2" \
    	-v /home/aisl/matsuzaki/docker/edgenets-docker/espdnet/:/root/espdnet/ \
    	-v /data/aisl/matsuzaki/dataset:/tmp/dataset \
    	-v /data/aisl/matsuzaki/runs/:/tmp/runs/ \
		--rm \
		--shm-size 5G \
		--workdir /root/espdnet/ \
		--name train-espdnet-uest-greenhouse-outsource \
		$(NAME):$(VERSION) \
        sh /root/espdnet/espdnet_greenhouse_uest_outsource.sh

train-espdnet-uest-greenhouse2-outsource:
	docker run -it \
    	--gpus="device=2" \
    	-v /home/aisl/matsuzaki/docker/edgenets-docker/espdnet/:/root/espdnet/ \
    	-v /data/aisl/matsuzaki/dataset:/tmp/dataset \
    	-v /data/aisl/matsuzaki/runs/:/tmp/runs/ \
		--rm \
		--shm-size 5G \
		--workdir /root/espdnet/ \
		--name train-espdnet-uest-greenhouse2-outsource \
		$(NAME):$(VERSION) \
        sh /root/espdnet/espdnet_greenhouse2_uest_outsource.sh

train-espdnet-uest-cucumber-outsource:
	docker run -it \
    	--gpus="device=2" \
    	-v /home/aisl/matsuzaki/docker/edgenets-docker/espdnet/:/root/espdnet/ \
    	-v /data/aisl/matsuzaki/dataset:/tmp/dataset \
    	-v /data/aisl/matsuzaki/runs/:/tmp/runs/ \
		--rm \
		--shm-size 5G \
		--workdir /root/espdnet/ \
		--name train-espdnet-uest-cucumber-outsource \
		$(NAME):$(VERSION) \
        sh /root/espdnet/espdnet_cucumber_uest_outsource.sh

train-espdnet-uest-1-2:
	docker run -it \
    	--gpus="device=2" \
    	-v /home/aisl/matsuzaki/docker/edgenets-docker/espdnet/:/root/espdnet/ \
    	-v /data/aisl/matsuzaki/dataset:/tmp/dataset \
    	-v /data/aisl/matsuzaki/runs/:/tmp/runs/ \
		--rm \
		--shm-size 5G \
		--workdir /root/espdnet/ \
		--name train-espdnet-uest-1-2 \
		$(NAME):$(VERSION) \
        sh /root/espdnet/espdnet_uest_1_2.sh

train-espdnet-uest-1-2-2:
	docker run -it \
    	--gpus="device=2" \
    	-v /home/aisl/matsuzaki/docker/edgenets-docker/espdnet/:/root/espdnet/ \
    	-v /data/aisl/matsuzaki/dataset:/tmp/dataset \
    	-v /data/aisl/matsuzaki/runs/:/tmp/runs/ \
		--rm \
		--shm-size 5G \
		--workdir /root/espdnet/ \
		--name train-espdnet-uest-1-2-2 \
		$(NAME):$(VERSION) \
        sh /root/espdnet/espdnet_uest_1_2.sh

train-espdnet-uest-1-3:
	docker run -it \
    	--gpus="device=2" \
    	-v /home/aisl/matsuzaki/docker/edgenets-docker/espdnet/:/root/espdnet/ \
    	-v /data/aisl/matsuzaki/dataset:/tmp/dataset \
    	-v /data/aisl/matsuzaki/runs/:/tmp/runs/ \
		--rm \
		--shm-size 5G \
		--workdir /root/espdnet/ \
		--name train-espdnet-uest-1-3 \
		$(NAME):$(VERSION) \
        sh /root/espdnet/espdnet_uest_1_3.sh

train-espdnet-uest-2-1:
	docker run -it \
    	--gpus="device=2" \
    	-v /home/aisl/matsuzaki/docker/edgenets-docker/espdnet/:/root/espdnet/ \
    	-v /data/aisl/matsuzaki/dataset:/tmp/dataset \
    	-v /data/aisl/matsuzaki/runs/:/tmp/runs/ \
		--rm \
		--shm-size 5G \
		--workdir /root/espdnet/ \
		--name train-espdnet-uest-2-1 \
		$(NAME):$(VERSION) \
        sh /root/espdnet/espdnet_uest_2_1.sh

train-espdnet-uest-2-3:
	docker run -it \
    	--gpus="device=2" \
    	-v /home/aisl/matsuzaki/docker/edgenets-docker/espdnet/:/root/espdnet/ \
    	-v /data/aisl/matsuzaki/dataset:/tmp/dataset \
    	-v /data/aisl/matsuzaki/runs/:/tmp/runs/ \
		--rm \
		--shm-size 5G \
		--workdir /root/espdnet/ \
		--name train-espdnet-uest-2-3 \
		$(NAME):$(VERSION) \
        sh /root/espdnet/espdnet_uest_2_3.sh

train-espdnet-uest-3-1:
	docker run -it \
    	--gpus="device=2" \
    	-v /home/aisl/matsuzaki/docker/edgenets-docker/espdnet/:/root/espdnet/ \
    	-v /data/aisl/matsuzaki/dataset:/tmp/dataset \
    	-v /data/aisl/matsuzaki/runs/:/tmp/runs/ \
		--rm \
		--shm-size 5G \
		--workdir /root/espdnet/ \
		--name train-espdnet-uest-3-1 \
		$(NAME):$(VERSION) \
        sh /root/espdnet/espdnet_uest_3_1.sh

train-espdnet-uest-12:
	docker run -it \
    	--gpus="device=2" \
    	-v /home/aisl/matsuzaki/docker/edgenets-docker/espdnet/:/root/espdnet/ \
    	-v /data/aisl/matsuzaki/dataset:/tmp/dataset \
    	-v /data/aisl/matsuzaki/runs/:/tmp/runs/ \
		--rm \
		--shm-size 5G \
		--workdir /root/espdnet/ \
		--name train-espdnet-uest-12 \
		$(NAME):$(VERSION) \
        sh /root/espdnet/espdnet_uest_12.sh

train-espdnet-uest-23:
	docker run -it \
    	--gpus="device=2" \
    	-v /home/aisl/matsuzaki/docker/edgenets-docker/espdnet/:/root/espdnet/ \
    	-v /data/aisl/matsuzaki/dataset:/tmp/dataset \
    	-v /data/aisl/matsuzaki/runs/:/tmp/runs/ \
		--rm \
		--shm-size 5G \
		--workdir /root/espdnet/ \
		--name train-espdnet-uest-23 \
		$(NAME):$(VERSION) \
        sh /root/espdnet/espdnet_uest_23.sh


train-espdnet-uest-31:
	docker run -it \
    	--gpus="device=2" \
    	-v /home/aisl/matsuzaki/docker/edgenets-docker/espdnet/:/root/espdnet/ \
    	-v /data/aisl/matsuzaki/dataset:/tmp/dataset \
    	-v /data/aisl/matsuzaki/runs/:/tmp/runs/ \
		--rm \
		--shm-size 5G \
		--workdir /root/espdnet/ \
		--name train-espdnet-uest-31 \
		$(NAME):$(VERSION) \
        sh /root/espdnet/espdnet_uest_31.sh

trav-mask-train:
	docker run -it \
    	--gpus="device=2" \
    	-v /home/aisl/matsuzaki/docker/edgenets-docker/espdnet/:/root/espdnet/ \
    	-v /data/aisl/matsuzaki/dataset:/tmp/dataset \
    	-v /data/aisl/matsuzaki/runs/:/tmp/runs/ \
		--rm \
		--shm-size 5G \
		--workdir /root/espdnet/ \
		--name trav-mask-train \
		$(NAME):$(VERSION) \
        sh /root/espdnet/trav_mask_train.sh

trav-mask-pu-train:
	docker run -it \
    	--gpus="device=2" \
    	-v /home/aisl/matsuzaki/docker/edgenets-docker/espdnet/:/root/espdnet/ \
    	-v /data/aisl/matsuzaki/dataset:/tmp/dataset \
    	-v /data/aisl/matsuzaki/runs/:/tmp/runs/ \
		--rm \
		--shm-size 5G \
		--workdir /root/espdnet/ \
		--name trav-mask-pu-train \
		$(NAME):$(VERSION) \
        sh /root/espdnet/scripts/trav_mask_pu_train.sh

trav-mask-pu-train-2:
	docker run -it \
    	--gpus="device=2" \
    	-v /home/aisl/matsuzaki/docker/edgenets-docker/espdnet/:/root/espdnet/ \
    	-v /data/aisl/matsuzaki/dataset:/tmp/dataset \
    	-v /data/aisl/matsuzaki/runs/:/tmp/runs/ \
		--rm \
		--shm-size 5G \
		--workdir /root/espdnet/ \
		--name trav-mask-pu-train-2 \
		$(NAME):$(VERSION) \
        sh /root/espdnet/scripts/trav_mask_pu_train.sh


deeplab-test:
	docker run -it \
    	--gpus="device=2" \
    	-v /home/aisl/matsuzaki/docker/edgenets-docker/espdnet/:/root/espdnet/ \
    	-v /data/aisl/matsuzaki/dataset:/tmp/dataset \
    	-v /data/aisl/matsuzaki/runs/:/tmp/runs/ \
		--rm \
		--shm-size 5G \
		--workdir /root/espdnet/ \
		--name deeplab-test \
		$(NAME):$(VERSION) \
		python /root/espdnet/model/segmentation/deeplabv3.py

visualize:
	docker run -it \
    	--gpus="device=2" \
    	-v /home/aisl/matsuzaki/docker/edgenets-docker/espdnet/:/root/espdnet/ \
    	-v /data/aisl/matsuzaki/dataset:/tmp/dataset \
		--rm \
		--shm-size 5G \
		--workdir /root/espdnet/ \
		--name visualize \
		$(NAME):$(VERSION) \
		python /root/espdnet/visualize_features.py

convert-to-pt:
	docker run -it \
    	--gpus="device=2" \
    	-v /home/aisl/matsuzaki/docker/edgenets-docker/espdnet/:/root/espdnet/ \
    	-v /data/aisl/matsuzaki/runs/:/tmp/runs/ \
		--rm \
		--shm-size 5G \
		--workdir /root/espdnet/ \
		--name visualize \
		$(NAME):$(VERSION) \
		bash /root/espdnet/convert_to_pt.sh