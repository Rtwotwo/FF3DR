#!/bin/bash
# download MatrixCity dataset contraining multi-view stereo data for 3D reconstruction and multi-view stereo research.
# we use hf-mirror.com to replace the huggingface.co to speed download matrixcity dataset


# 1.download aerial part of small city about 31.3G
# SPLITS=('train' 'test' 'pose')
# for SPLIT in "${SPLITS[@]}"; do
#     if [ "$SPLIT" == 'pose' ]; then
#         CHUNKS=('block_A' 'block_B' 'block_C' 'block_D' 'block_E' 'block_all')
#         for CHUNK in "${CHUNKS[@]}"; do
#             echo "[$(date)] Downloading ${SPLIT}/${CHUNK}..."
#             mkdir -p ./dataset/MatrixCity/small_city/aerial/${SPLIT}/${CHUNK}/
#             wget  https://hf-mirror.com/datasets/BoDai/MatrixCity/resolve/main/small_city/aerial/pose/${CHUNK}/transforms_test.json?download=true \
#                     -O ./dataset/MatrixCity/small_city/aerial/${SPLIT}/${CHUNK}/transforms_test.json
#             wget  https://hf-mirror.com/datasets/BoDai/MatrixCity/resolve/main/small_city/aerial/pose/${CHUNK}/transforms_train.json?download=true \
#                     -O ./dataset/MatrixCity/small_city/aerial/${SPLIT}/${CHUNK}/transforms_train.json
#         done
#     elif [ "$SPLIT" == 'test' ]; then
#         CHUNKS=('block_1' 'block_2' 'block_3' 'block_4' 'block_5' 'block_6' 'block_7' 'block_8' 'block_9' 'block_10')
#         for CHUNK in "${CHUNKS[@]}"; do
#             echo "[$(date)] Downloading ${SPLIT}/${CHUNK}..."
#             # download the tar file and extract file
#             mkdir -p ./dataset/MatrixCity/small_city/aerial/${SPLIT}/${CHUNK}/
#             wget  https://hf-mirror.com/datasets/BoDai/MatrixCity/resolve/main/small_city/aerial/${SPLIT}/${CHUNK}_test.tar?download=true \
#                     -O ./dataset/MatrixCity/small_city/aerial/${SPLIT}/${CHUNK}_test.tar
#             tar -xvf ./dataset/MatrixCity/small_city/aerial/${SPLIT}/${CHUNK}_test.tar -C ./dataset/MatrixCity/small_city/aerial/${SPLIT}/
#             rm ./dataset/MatrixCity/small_city/aerial/${SPLIT}/${CHUNK}_test.tar

#             # download json file 
#             wget  https://hf-mirror.com/datasets/BoDai/MatrixCity/resolve/main/small_city/aerial/${SPLIT}/${CHUNK}_test/transforms.json?download=true \
#                     -O ./dataset/MatrixCity/small_city/aerial/${SPLIT}/${CHUNK}/transforms.json
#             wget  https://hf-mirror.com/datasets/BoDai/MatrixCity/resolve/main/small_city/aerial/${SPLIT}/${CHUNK}_test/transforms_origin.json?download=true \
#                     -O ./dataset/MatrixCity/small_city/aerial/${SPLIT}/${CHUNK}/transforms_origin.json
#         done
#     elif [ "$SPLIT" == 'train' ]; then
#         CHUNKS=('block_1' 'block_2' 'block_3' 'block_4' 'block_5' 'block_6' 'block_7' 'block_8' 'block_9' 'block_10')
#         for CHUNK in "${CHUNKS[@]}"; do
#             echo "[$(date)] Downloading ${SPLIT}/${CHUNK}..."
#             # download the tar file and extract file
#             mkdir -p ./dataset/MatrixCity/small_city/aerial/${SPLIT}/${CHUNK}/
#             wget  https://hf-mirror.com/datasets/BoDai/MatrixCity/resolve/main/small_city/aerial/${SPLIT}/${CHUNK}.tar?download=true \
#                     -O ./dataset/MatrixCity/small_city/aerial/${SPLIT}/${CHUNK}.tar
#             tar -xvf ./dataset/MatrixCity/small_city/aerial/${SPLIT}/${CHUNK}.tar -C ./dataset/MatrixCity/small_city/aerial/${SPLIT}/
#             rm ./dataset/MatrixCity/small_city/aerial/${SPLIT}/${CHUNK}.tar

#             # download json file 
#             wget  https://hf-mirror.com/datasets/BoDai/MatrixCity/resolve/main/small_city/aerial/${SPLIT}/${CHUNK}/transforms.json?download=true \
#                     -O ./dataset/MatrixCity/small_city/aerial/${SPLIT}/${CHUNK}/transforms.json
#             wget  https://hf-mirror.com/datasets/BoDai/MatrixCity/resolve/main/small_city/aerial/${SPLIT}/${CHUNK}/transforms_origin.json?download=true \
#                     -O ./dataset/MatrixCity/small_city/aerial/${SPLIT}/${CHUNK}/transforms_origin.json
#         done
#     fi
# done


# 2.download aerial part of big city about 206G
SPLITS=('train')
for SPLIT in "${SPLITS[@]}"; do 
    if [ "$SPLIT" == 'test' ]; then
        CHUNKS=('big_high_block_1_test' 'big_high_block_2_test' 'big_high_block_3_test' 'big_high_block_4_test' 'big_high_block_5_test' 'big_high_block_6_test')
        for CHUNK in "${CHUNKS[@]}"; do
            echo "[$(date)] Downloading ${SPLIT}/${CHUNK}..."
            # download the tar file and extract file
            mkdir -p ./dataset/MatrixCity/big_city/aerial/${SPLIT}/${CHUNK}/
            wget  https://hf-mirror.com/datasets/BoDai/MatrixCity/resolve/main/big_city/aerial/${SPLIT}/${CHUNK}.tar?download=true \
                    -O ./dataset/MatrixCity/big_city/aerial/${SPLIT}/${CHUNK}.tar
            tar -xvf ./dataset/MatrixCity/big_city/aerial/${SPLIT}/${CHUNK}.tar -C ./dataset/MatrixCity/big_city/aerial/${SPLIT}/
            rm ./dataset/MatrixCity/big_city/aerial/${SPLIT}/${CHUNK}.tar

            # download json file 
            wget  https://hf-mirror.com/datasets/BoDai/MatrixCity/resolve/main/big_city/aerial/${SPLIT}/${CHUNK}/transforms.json?download=true \
                    -O ./dataset/MatrixCity/big_city/aerial/${SPLIT}/${CHUNK}/transforms.json
        done
    elif [ "$SPLIT" == 'train' ]; then
        CHUNKS=('big_high_block_1' 'big_high_block_2' 'big_high_block_3' 'big_high_block_4' 'big_high_block_5' 'big_high_block_6')
        for CHUNK in "${CHUNKS[@]}"; do
            if [ "$CHUNK" == 'big_high_block_1' ]; then 
                mkdir -p ./dataset/MatrixCity/big_city/aerial/${SPLIT}/${CHUNK}/
                PATCHS=("big_high_block_1.tar00" "big_high_block_1.tar01" "big_high_block_1.tar02" "big_high_block_1.tar03" "big_high_block_1.tar04")
                for PATCH in "${PATCHS[@]}"; do
                    echo "[$(date)] Downloading ${SPLIT}/${CHUNK}/${PATCH}..."
                    # download the tar file and extract file
                    wget  https://hf-mirror.com/datasets/BoDai/MatrixCity/resolve/main/big_city/aerial/${SPLIT}/${PATCH}?download=true \
                            -O ./dataset/MatrixCity/big_city/aerial/${SPLIT}/${PATCH}
                    tar -xvf ./dataset/MatrixCity/big_city/aerial/${SPLIT}/${CHUNK}/${PATCH} -C ./dataset/MatrixCity/big_city/aerial/${SPLIT}/${CHUNK}/
                    rm ./dataset/MatrixCity/big_city/aerial/${SPLIT}/${CHUNK}/${PATCH}
                done
            elif [ "$CHUNK" != 'big_high_block_1' ]; then 
                echo "[$(date)] Downloading ${SPLIT}/${CHUNK}..."
                # download the tar file and extract file
                mkdir -p ./dataset/MatrixCity/big_city/aerial/${SPLIT}/${CHUNK}/
                wget  https://hf-mirror.com/datasets/BoDai/MatrixCity/resolve/main/big_city/aerial/${SPLIT}/${CHUNK}.tar?download=true \
                        -O ./dataset/MatrixCity/big_city/aerial/${SPLIT}/${CHUNK}.tar
                tar -xvf ./dataset/MatrixCity/big_city/aerial/${SPLIT}/${CHUNK}.tar -C ./dataset/MatrixCity/big_city/aerial/${SPLIT}/
                rm ./dataset/MatrixCity/big_city/aerial/${SPLIT}/${CHUNK}.tar
            # download json file
            wget  https://hf-mirror.com/datasets/BoDai/MatrixCity/resolve/main/big_city/aerial/${SPLIT}/${CHUNK}/transforms.json?download=true \
                   -O ./dataset/MatrixCity/big_city/aerial/${SPLIT}/${CHUNK}/transforms.json
            fi
        done
    fi
done