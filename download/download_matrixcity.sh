#!/bin/bash
# download MatrixCity dataset contraining multi-view stereo data for 3D reconstruction and multi-view stereo research.
# includes four compressed packages: train.zip (1.2G), test.zip (0.3G), predict.zip (0.2G), and readme.zip (1.72K   
# we use hf-mirror.com to replace the huggingface.co to speed download matrixcity dataset


# 1.download aerial part of small city
SPLITS=('train' 'test' 'pose')
for SPLIT in "${SPLITS[@]}"; do
    if [ "$SPLIT" == 'pose' ]; then
        CHUNKS=('block_A' 'block_B' 'block_C' 'block_D' 'block_E' 'block_all')
        for CHUNK in "${CHUNKS[@]}"; do
            echo "[$(date)] Downloading ${SPLIT}/${CHUNK}..."
            mkdir -p ./dataset/MatrixCity/small_city/aerial/${SPLIT}/${CHUNK}/
            wget -c https://hf-mirror.com/datasets/BoDai/MatrixCity/resolve/main/small_city/pose/${CHUNK}/transform_test.json?download=true \
                    -O ./dataset/MatrixCity/small_city/aerial/${SPLIT}/${CHUNK}/transform_test.json
            wget -c https://hf-mirror.com/datasets/BoDai/MatrixCity/resolve/main/small_city/pose/${CHUNK}/transform_train.json?download=true \
                    -O ./dataset/MatrixCity/small_city/aerial/${SPLIT}/${CHUNK}/transform_train.json
        done
    elif [ "$SPLIT" == 'test' ]; then
        CHUNKS=('block_1' 'block_2' 'block_3' 'block_4' 'block_5' 'block_6' 'block_7' 'block_8' 'block_9' 'block_10')
        for CHUNK in "${CHUNKS[@]}"; do
            echo "[$(date)] Downloading ${SPLIT}/${CHUNK}..."
            # download the tar file and extract file
            mkdir -p ./dataset/MatrixCity/small_city/aerial/${SPLIT}/${CHUNK}/
            wget -c https://hf-mirror.com/datasets/BoDai/MatrixCity/resolve/main/small_city/aerial/${SPLIT}/${CHUNK}_test.tar?download=true \
                    -O ./dataset/MatrixCity/small_city/aerial/${SPLIT}/${CHUNK}_test.tar
            tar -xvf ./dataset/MatrixCity/small_city/aerial/${SPLIT}/${CHUNK}_test.tar -C ./dataset/MatrixCity/small_city/aerial/${SPLIT}/${CHUNK}/
            rm ./dataset/MatrixCity/small_city/aerial/${SPLIT}/${CHUNK}_test.tar

            # download json file 
            wget -c https://hf-mirror.com/datasets/BoDai/MatrixCity/resolve/main/small_city/aerial/${SPLIT}/${CHUNK}_test/transforms.json?download=true \
                    -O ./dataset/MatrixCity/small_city/aerial/${SPLIT}/${CHUNK}/transforms.json
            wget -c https://hf-mirror.com/datasets/BoDai/MatrixCity/resolve/main/small_city/aerial/${SPLIT}/${CHUNK}_test/transforms_origin.json?download=true \
                    -O ./dataset/MatrixCity/small_city/aerial/${SPLIT}/${CHUNK}/transforms_origin.json
        done
    elif [ "$SPLIT" == 'train' ]; then
        CHUNKS=('block_1' 'block_2' 'block_3' 'block_4' 'block_5' 'block_6' 'block_7' 'block_8' 'block_9' 'block_10')
        for CHUNK in "${CHUNKS[@]}"; do
            echo "[$(date)] Downloading ${SPLIT}/${CHUNK}..."
            # download the tar file and extract file
            mkdir -p ./dataset/MatrixCity/small_city/aerial/${SPLIT}/${CHUNK}/
            wget -c https://hf-mirror.com/datasets/BoDai/MatrixCity/resolve/main/small_city/aerial/${SPLIT}/${CHUNK}.tar?download=true \
                    -O ./dataset/MatrixCity/small_city/aerial/${SPLIT}/${CHUNK}.tar
            tar -xvf ./dataset/MatrixCity/small_city/aerial/${SPLIT}/${CHUNK}.tar -C ./dataset/MatrixCity/small_city/aerial/${SPLIT}/${CHUNK}/
            rm ./dataset/MatrixCity/small_city/aerial/${SPLIT}/${CHUNK}.tar

            # download json file 
            wget -c https://hf-mirror.com/datasets/BoDai/MatrixCity/resolve/main/small_city/aerial/${SPLIT}/${CHUNK}/transforms.json?download=true \
                    -O ./dataset/MatrixCity/small_city/aerial/${SPLIT}/${CHUNK}/transforms.json
            wget -c https://hf-mirror.com/datasets/BoDai/MatrixCity/resolve/main/small_city/aerial/${SPLIT}/${CHUNK}/transforms_origin.json?download=true \
                    -O ./dataset/MatrixCity/small_city/aerial/${SPLIT}/${CHUNK}/transforms_origin.json
        done
    fi
done
