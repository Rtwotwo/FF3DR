# :rocket: Project Guide :rocket:

## 🎉️ Visualization 🎉️

Currently, we render the PolyTech scene in UrbanScene3D and display its gsplt rendering effect as well as the depth prediction effect of Depthanything3. Subsequently, we will measure the PSNR, SSIM and LPILS indices after rendering to provide a reference for evaluating the performance of the feedforward 3D reconstruction model in regressing 3D representations using aerial images captured from a drone perspective.

https://github.com/user-attachments/assets/6a492199-dad3-4d24-ac71-1cd50b3ce311 
https://github.com/user-attachments/assets/45336410-d84a-4ee9-9c13-7d573949a794 
https://github.com/user-attachments/assets/2e924ce4-9138-46af-9a60-1433de512088
https://github.com/user-attachments/assets/84ef0e96-fc51-419c-aef3-10113184370b
https://github.com/user-attachments/assets/3c3de141-3354-47df-8c6a-297ddbfbbd80
https://github.com/user-attachments/assets/f1aee625-c626-4989-8b96-a19ae786513b

## 🌍 Dataset 🌍

We have provided dataset download scripts for [WHU-OMVS](download/download_whuomvs.sh) and [MatrixCity-SmallCity](download/download_matrixcity.sh). You can run the scripts within the project folder to download the datasets. However, for the UrbanScene3D dataset, as official download links are provided, you can download the required parts from the corresponding [UrbanScene Baidu Netdisk](https://pan.baidu.com/s/1nqurXpbMzFo_-Cmf6eheOw?pwd=7zdg).

```bash
cd your/project/directory && mkdir dataset
bash ./download/download_matrixcity.sh
bash ./download/download_whuomvs.sh
```

### WHU-OMVS Dataset

The dataset used in the experiment is mainly the [WHU-OMVS Dataset](https://gpcv.whu.edu.cn/data/WHU_OMVS_dataset/WHU_dataset.htm), which includes four compressed packages: train.zip (67.1G), test.zip (22.1G), predict.zip (45.7G), and readme.zip (1.72K). Following the division of the WHU-MVS dataset, the data of Area #1/4/5/6 were used for the training. One slight difference is that both Area #2 and Area #3 were used for the testing in Liu and Ji (2020), whereas, in this work, Area #2 was treated as the validation region and Area #3 was the test region.

The simulated cameras were arranged following a typical oblique five-view camera system. There is no displacement among the five cameras, that is, the five cameras are positioned to share the same projection center while looking in different directions. Camera #3 looks straight down, while the remaining four cameras have a tilt angle of 40°. Specifically, Camera #1 and Camera #5 look forward and backward, respectively, while Camera #2 and Camera #4 look to the right and left, respectively.

<div align="center">
  <img src="assets/dataset_map.jpg" alt="dataset_map"> 
  <img src="assets/uav_cameras.jpg" alt="dataset_map">
</div>

### MatrixCity Dataset

[MatrixCity](https://city-super.github.io/matrixcity/) have constructed a large-scale, comprehensive, and high-quality synthetic dataset for urban-level neural rendering research. Leveraging Unreal Engine 5's city sample project, we have developed a pipeline that enables convenient collection of urban aerial and street-view images with authentic camera poses, while acquiring a range of additional data modalities. This pipeline also allows flexible control over environmental factors such as lighting, weather, pedestrians, and traffic flow, meeting the diverse requirements of tasks covering urban-level neural rendering and other related fields. The resulting pilot dataset, MatrixCity, contains 60,000 aerial images and 350,000 street-view images derived from two city maps covering a total area of 28 square kilometers.

<div align="center">
  <img src="assets/matrixcity_map.jpg" alt="dataset_map">
  <img src="assets/matrixcity_aera.jpg" alt="dataset_map">
</div>

## ❤️ Thanks ❤️

This project relies on outstanding open-source projects including [DepthAnything3](https://github.com/ByteDance-Seed/Depth-Anything-3), [VGGT-Long](https://github.com/DengKaiCQ/VGGT-Long) and [Ada-MVS](https://github.com/gpcv-liujin/Ada-MVS). We appreciate the work done by the authors of the aforementioned open-source projects.

