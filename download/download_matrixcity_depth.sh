#!/bin/bash
# download MatrixCity dataset contraining multi-view stereo data for 3D reconstruction and multi-view stereo research.
# we use hf-mirror.com to replace the huggingface.co to speed download matrixcity dataset


# 1.download depth part of small city about 2.75G
SPLITS=('train' 'test')
for SPLIT in "${SPLITS[@]}"; do
    if [ "$SPLIT" == 'test' ]; then
        CHUNKS=('block_1_test_depth' 'block_2_test_depth' 'block_3_test_depth' 'block_4_test_depth' 'block_5_test_depth' 'block_6_test_depth' 'block_7_test_depth' 'block_8_test_depth' 'block_9_test_depth' 'block_10_test_depth')
        for CHUNK in "${CHUNKS[@]}"; do
            echo "[$(date)] Downloading ${SPLIT}/${CHUNK}..."
            mkdir -p ./dataset/MatrixCity/small_city_depth/aerial/${SPLIT}/${CHUNK}/
            wget  https://hf-mirror.com/datasets/BoDai/MatrixCity/resolve/main/small_city_depth/aerial/${SPLIT}/${CHUNK}.tar?download=true \
                    -O ./dataset/MatrixCity/small_city_depth/aerial/${SPLIT}/${CHUNK}.tar
            tar -xvf ./dataset/MatrixCity/small_city_depth/aerial/${SPLIT}/${CHUNK}.tar -C ./dataset/MatrixCity/small_city_depth/aerial/${SPLIT}/
            rm ./dataset/MatrixCity/small_city_depth/aerial/${SPLIT}/${CHUNK}.tar
        done
    elif [ "$SPLIT" == 'train' ]; then
        CHUNKS=('block_1_depth' 'block_2_depth' 'block_3_depth' 'block_4_depth' 'block_5_depth' 'block_6_depth' 'block_7_depth' 'block_8_depth' 'block_9_depth' 'block_10_depth')
        for CHUNK in "${CHUNKS[@]}"; do
            echo "[$(date)] Downloading ${SPLIT}/${CHUNK}..."
            mkdir -p ./dataset/MatrixCity/small_city_depth/aerial/${SPLIT}/${CHUNK}/
            wget  https://hf-mirror.com/datasets/BoDai/MatrixCity/resolve/main/small_city_depth/aerial/${SPLIT}/${CHUNK}.tar?download=true \
                    -O ./dataset/MatrixCity/small_city_depth/aerial/${SPLIT}/${CHUNK}.tar
            tar -xvf ./dataset/MatrixCity/small_city_depth/aerial/${SPLIT}/${CHUNK}.tar -C ./dataset/MatrixCity/small_city_depth/aerial/${SPLIT}/
            rm ./dataset/MatrixCity/small_city_depth/aerial/${SPLIT}/${CHUNK}.tar
        done
    fi
done


# 2.download depth part of big city about 25.6G
SPLITS=('train' 'test')
for SPLIT in "${SPLITS[@]}"; do
    if [ $SPLIT == 'test' ]; then
        CHUNKS=('big_high_block_1_test_depth' 'big_high_block_2_test_depth' 'big_high_block_3_test_depth' 'big_high_block_4_test_depth' 'big_high_block_5_test_depth' 'big_high_block_6_test_depth')
        for CHUNK in "${CHUNKS[@]}"; do 
            echo "[$(date)] Downloading ${SPLIT}/${CHUNK}..."
            mkdir -p ./dataset/MatrixCity/big_city_depth/aerial/${SPLIT}/${CHUNK}/
            wget  https://hf-mirror.com/datasets/BoDai/MatrixCity/resolve/main/big_city_depth/aerial/${SPLIT}/${CHUNK}.tar?download=true \
                    -O ./dataset/MatrixCity/big_city_depth/aerial/${SPLIT}/${CHUNK}.tar
            tar -xvf ./dataset/MatrixCity/big_city_depth/aerial/${SPLIT}/${CHUNK}.tar -C ./dataset/MatrixCity/big_city_depth/aerial/${SPLIT}/
            rm ./dataset/MatrixCity/big_city_depth/aerial/${SPLIT}/${CHUNK}.tar
        done
    elif [ $SPLIT == 'train' ]; then
        CHUNKS=('big_high_block_1_depth' 'big_high_block_2_depth' 'big_high_block_3_depth' 'big_high_block_4_depth' 'big_high_block_5_depth' 'big_high_block_6_depth')
        for CHUNK in "${CHUNKS[@]}"; do 
            echo "[$(date)] Downloading ${SPLIT}/${CHUNK}..."
            mkdir -p ./dataset/MatrixCity/big_city_depth/aerial/${SPLIT}/${CHUNK}/
            wget  https://hf-mirror.com/datasets/BoDai/MatrixCity/resolve/main/big_city_depth/aerial/${SPLIT}/${CHUNK}.tar?download=true \
                    -O ./dataset/MatrixCity/big_city_depth/aerial/${SPLIT}/${CHUNK}.tar
            tar -xvf ./dataset/MatrixCity/big_city_depth/aerial/${SPLIT}/${CHUNK}.tar -C ./dataset/MatrixCity/big_city_depth/aerial/${SPLIT}/
            rm ./dataset/MatrixCity/big_city_depth/aerial/${SPLIT}/${CHUNK}.tar
        done
    fi
done
