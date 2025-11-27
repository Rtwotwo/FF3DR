# 🚀️ Project Guide 🚀️

## Dataset

The dataset used in the experiment is mainly the [WHU-OMVS Dataset](https://gpcv.whu.edu.cn/data/WHU_OMVS_dataset/WHU_dataset.htm), which includes four compressed packages: train.zip (67.1G), test.zip (22.1G), predict.zip (45.7G), and readme.zip (1.72K). Meanwhile, we provide [download\_utils.py](https://www.doubao.com/chat/utils/download_utils.py) to automatically download the files and save them in the data folder. The startup command is as follows:

```bash
# You can set your own save_dir pat
python utils/download_utils.py --save_dir ./data
```

To finish the feedforward 3D reconstruction task, we use DJI UAV to collect image with UAV's view to build the FF3R model. And before use UAV, the camera and imu should be calibrated, which depends on [imu_utils](https://github.com/gaowenliang/imu_utils.git) and [kalibr](https://github.com/ethz-asl/kalibr.git). Becanse of Ubuntu22.04 version, we use docker to install kalibr and imu_utils tools to avoid system conflicts.

```bash
# install ros1 on the ubuntu22.04
docker pull osrf/ros:noetic-desktop-full
# enter into the docker and install dependencics
docker run -it --name kalibr_noetic \
    -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY \
    -v $(pwd)/data:/data \
    ros:noetic-desktop-full bash

# install related tools
apt update && apt install -y \
    git \
    wget \
    vim \
    nano \
    build-essential \
    python3-pip \
    libyaml-cpp-dev \
    libceres-dev
# clone the kalibr and imu_utils tool
mkdir -p /catkin_ws/src
cd /catkin_ws/src
git clone https://github.com/ethz-asl/kalibr.git
git clone https://github.com/gaowenliang/imu_utils.git
cd /catkin_ws
catkin_make -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash
```

Next, you can process data in the docker container: 1.record or convert video/image data in the ros2 environment, use rosbags covert to convert the ros2bag into ROS1 bag; 2.make camera and imu calibration in the docker.

```bash
# record or convert  data
pip install rosbags
rosbags convert --src ros2_bag/ --dst ros1_bag.bag

# run camera and imu calibration
docker start -ai kalibr_noetic
rosrun kalibr kalibr_calibrate_cameras --bag /data/ros1_bag.bag ...
rosrun kalibr kalibr_calibrate_imu_camera --bag /data/ros1_bag.bag ...
rosrun imu_utils imu_an /data/imu.bag
```

