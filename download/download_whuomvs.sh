# !/bin/bash
# download WHU-OMVS dataset contraining multi-view stereo data for 3D reconstruction and multi-view stereo research.
# includes four compressed packages: train.zip (67.1G), test.zip (22.1G), predict.zip (45.7G), and readme.zip (1.72K).


DATA_PATH=./dataset
mkdir -p $DATA_PATH && cd $DATA_PATH

# 1.download the training data
wget https://gpcv.whu.edu.cn/data/WHU_OMVS_dataset/train.zip
unzip $DATA_PATH/train.zip -d $DATA_PATH
rm $DATA_PATH/train.zip

# 2.download the testing data
wget https://gpcv.whu.edu.cn/data/WHU_OMVS_dataset/test.zip
unzip $DATA_PATH/test.zip -d $DATA_PATH
rm $DATA_PATH/test.zip

# 3.download the prediction data
wget https://gpcv.whu.edu.cn/data/WHU_OMVS_dataset/predict.zip
unzip $DATA_PATH/predict.zip -d $DATA_PATH
rm $DATA_PATH/predict.zip