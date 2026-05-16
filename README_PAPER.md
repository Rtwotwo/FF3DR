# 第三章 方案设计与实现

## 3.1 任务需求与问题定义

### 3.1.1 任务背景

无人机倾斜摄影测量是当前城市三维重建的主要技术手段，其通过搭载多视角相机阵列（通常为5个视角：1个垂直向下、4个倾斜方向），沿预设航线连续采集地面影像序列，再经多视角立体匹配、密集点云生成、网格重建等流程，最终输出城市级三维模型。然而，传统重建流程依赖COLMAP等基于特征匹配与迭代优化的方法，存在重建耗时长（数小时至数天）、特征匹配失败导致重建空洞、对弱纹理/重复纹理区域鲁棒性差等根本性缺陷。

前馈式三维重建技术为解决上述问题提供了新思路。该方法通过端到端的深度学习架构，将场景的二维图像序列直接回归至三维几何表示（深度图、3D点云或3D点图），无需传统MVS中的特征匹配、视差筛选等迭代优化步骤，从而大幅提升重建效率与实时性。

### 3.1.2 核心需求

本项目的核心需求是：**采用前馈式三维重建技术，处理无人机单目或多目相机采集的连续帧图像序列，实现实时三维场景构造**。具体分解为以下子需求：

**（1）多视角长序列3D回归**：无人机采集的图像序列通常包含数百至上千帧连续图像（如WHU-OMVS数据集中5个相机×312帧=1560张图像），前馈模型需要在一次前向传播中将这些图像回归为统一坐标系下的3D几何表示。然而，现有前馈模型（DepthAnything3、Pi3、VGGT、MapAnything等）的输入长度受限于GPU显存，无法一次性处理全部帧。

**（2）跨模型统一推理框架**：不同前馈模型的输出格式各异——DepthAnything3输出深度图+位姿，Pi3输出局部3D点+相机矩阵，VGGT输出深度图+3D点图+位姿编码，MapAnything输出3D点图+置信度。需要设计统一的推理框架，屏蔽模型差异，使下游的坐标对齐、点云融合等流程与具体模型解耦。

**（3）长序列全局一致性**：分块推理导致各chunk处于不同的局部坐标系，且存在未知的尺度差异。需要设计鲁棒的chunk间坐标对齐算法，将所有局部3D表示统一到全局坐标系下，消除累积漂移和尺度不一致。

**（4）回环检测与全局优化**：无人机航拍常出现航线交叉，形成回环。需要检测回环约束并通过图优化消除长序列的累积误差，提升全局重建精度。

**（5）领域适应性微调**：通用前馈模型在无人机视角下存在深度估计偏差（如俯视视角的尺度歧义），需要利用无人机场景的深度真值对模型进行领域适应性微调。

### 3.1.3 问题形式化

将上述需求形式化为数学问题。给定 $N_c$ 个相机、$N_f$ 帧图像组成的图像集合 $\mathcal{I} = \{I_{c,f}\}_{c=1,f=1}^{N_c, N_f}$，目标是求解全局坐标系下的3D点云 $\mathcal{P} = \{p_i \in \mathbb{R}^3\}$，使得：

$$\mathcal{P}^* = \arg\min_{\mathcal{P}} \sum_{c,f} \sum_{(u,v) \in \Omega_{c,f}} \rho\left(\pi_{c,f}^{-1}\left(d_{c,f}(u,v), K_{c,f}, T_{c,f}\right), \mathcal{P}\right)$$

其中 $\pi_{c,f}^{-1}$ 为反投影函数，$d_{c,f}$ 为深度图，$K_{c,f}$ 为内参，$T_{c,f}$ 为外参，$\Omega_{c,f}$ 为有效像素集合，$\rho$ 为距离度量函数。

由于前馈模型无法一次性处理全部图像，将序列按滑动窗口切分为 $M$ 个chunk，每个chunk独立推理后需要对齐：

$$\mathcal{P}_{\text{global}} = \bigcup_{m=1}^{M} \left\{ s_m R_m p + t_m \mid p \in \mathcal{P}_m \right\}$$

其中 $(s_m, R_m, t_m) \in \text{Sim}(3)$ 为第 $m$ 个chunk到全局坐标系的相似变换。核心问题转化为：**如何鲁棒地估计相邻chunk间的Sim3变换，并消除累积误差？**

---

## 3.2 总体方案设计

### 3.2.1 系统架构

本系统采用"**分块推理—局部拼接—全局对齐—回环优化—评测输出**"的五阶段流水线架构，整体流程如图所示：

```
输入: N_c × N_f 张无人机图像序列
    │
    ▼
[阶段1] 长序列分块 (Chunking)
    │  滑动窗口切分, chunk_size=150, overlap=72
    ▼
[阶段2] 单Chunk多视角3D推理 (Per-Chunk Inference)
    │  前馈模型推理 → 深度图/3D点图 + 位姿 + 置信度
    │  帧间SE3拼接 → chunk内统一坐标系3D点云
    ▼
[阶段3] Chunk间SIM3全局对齐 (Global Alignment)
    │  重叠区域点云配准 → 相邻chunk间Sim3变换
    │  异常检测 → 累积变换 → 全局坐标系
    ▼
[阶段4] 回环检测与全局优化 (Loop Closure)
    │  视觉位置识别 → 回环约束 → Sim3图优化
    ▼
[阶段5] 评测与输出 (Evaluation)
    │  置信度过滤 → PLY输出 → DSM指标计算
    ▼
输出: reconstruction_merged.ply + 评测指标
```

### 3.2.2 模块划分

系统由以下核心模块组成：

| 模块 | 代码位置 | 功能 |
|------|---------|------|
| FF3DR 主控类 | `running/inference/run_whuomvs_inference.py` | 流水线调度、参数管理 |
| 前馈模型适配层 | `models/` | 四种模型的统一推理接口 |
| 深度反投影 | `loop_utils/alignment_torch.py` | 深度图→3D点云 |
| Sim3对齐 | `loop_utils/sim3utils.py` | 点云配准、变换累积 |
| 回环检测 | `loop_utils/loop_detector.py` | 视觉位置识别 |
| Sim3图优化 | `loop_utils/sim3loop.py` | 回环约束优化 |
| 评测指标 | `running/metrics/` | PAG/MAE/RMSE等指标计算 |
| LoRA微调 | `running/training/train_da3_lora.py` | DA3模型领域适应 |

### 3.2.3 设计原则

**（1）模型无关性**：通过适配器模式，将四种前馈模型的不同输出格式统一为6键字典 `{depth, conf, intrinsics, extrinsics, processed_images, world_points}`，下游所有流程仅依赖统一接口，与具体模型解耦。

**（2）分而治之**：将长序列切分为可管理的chunk，在chunk内部通过帧间SE3拼接实现局部一致性，在chunk之间通过Sim3对齐实现全局一致性，避免一次性处理全部图像的显存瓶颈。

**（3）鲁棒性优先**：在帧间拼接、chunk间对齐、回环检测等每个环节都设置异常检测机制，基于历史误差统计（中位数的3倍阈值）剔除异常变换，防止错误传播。

---

## 3.3 关键模块改进与设计

### 3.3.1 统一多模型推理适配层

#### 3.3.1.1 问题

四种前馈模型的输出格式存在本质差异：DepthAnything3输出深度图+位姿（需反投影），Pi3输出局部3D点+相机矩阵（需c2w→w2c转换），VGGT输出深度图+3D点图+位姿编码（需解码），MapAnything输出3D点图列表（需拼接）。若下游流程直接处理各模型的原始输出，将导致大量条件分支和重复代码。

#### 3.3.1.2 设计

采用**适配器模式 + 双向补全**策略，分两层实现输出格式统一：

**第一层：模型适配器 `_infer_*()`**。每个模型对应一个适配器方法，负责将模型原始输出转换为统一的6键字典。各适配器的关键转换逻辑如下：

- **DepthAnything3**：`conf - 1.0`（消除`expp1`激活的+1偏移），`world_points = None`（标记需后续反投影）
- **Pi3**：`depth = local_points[..., 2]`（取局部点z分量），`extrinsics = inv(c2w)[:3, :4]`，构造简化内参 $K = \text{diag}(f, f, 1)$ 其中 $f = \max(H, W)$
- **VGGT**：`pose_encoding_to_extri_intri()`（9维编码→矩阵），置信度回退 `depth_conf → world_points_conf`
- **MapAnything**：`cat_tensors()` 多视角拼接，`c2w → w2c` 转换

**第二层：格式补全 `_chunk_to_point_arrays()`**。该函数确保无论模型输出何种3D表示形式，都能补全为完整的4元组 `(world_points, conf, images, depth)`：

$$\text{world\_points} = \begin{cases} \text{model output} & \text{若模型直接输出3D点图} \\ \pi^{-1}(\text{depth}, K, T) & \text{若 world\_points = None} \end{cases}$$

$$\text{depth} = \begin{cases} \text{model output} & \text{若模型输出深度图} \\ \|\text{world\_points}\|_2 & \text{若 depth = None} \end{cases}$$

反投影函数 `depth_to_point_cloud_optimized_torch()` 的数学过程为：

$$\mathbf{X}_{\text{cam}} = K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} \cdot d(u,v), \quad \mathbf{X}_{\text{world}} = T_{\text{c2w}} \begin{bmatrix} \mathbf{X}_{\text{cam}} \\ 1 \end{bmatrix}$$

其中 $d(u,v)$ 为像素 $(u,v)$ 处的深度值，$K^{-1}$ 为内参逆矩阵，$T_{\text{c2w}} = T_{\text{w2c}}^{-1}$ 为世界到相机变换的逆。

### 3.3.2 长序列分块与帧间SE3拼接

#### 3.3.2.1 分块策略

给定总帧数 $N_f$，chunk大小 $C$，重叠帧数 $O$，步长 $S = C - O$，chunk数量 $M = \lceil (N_f - O) / S \rceil$。第 $m$ 个chunk覆盖帧范围 $[m \cdot S, \min(m \cdot S + C, N_f))$。重叠区域是后续chunk间坐标对齐的关键——两个相邻chunk共享 $O$ 帧的图像，其对应的3D点云应处于同一坐标系。

#### 3.3.2.2 帧间SE3拼接

以DepthAnything3的`framewise_multicam`模式为例，对chunk内逐帧推理后需要进行帧间坐标统一：

**Step 1：尺度归一化**。从外参矩阵提取相机中心 $\mathbf{c}_f$，计算相邻相机间距离作为rig尺度：

$$\sigma_f = \text{median}\left(\|\mathbf{c}_f - \mathbf{c}_{f-1}\|_2\right)$$

对相机中心和3D点云做归一化 $\hat{\mathbf{c}}_f = \mathbf{c}_f / \sigma_f$，$\hat{\mathbf{P}}_f = \mathbf{P}_f / \sigma_f$，消除模型预测的尺度歧义。

**Step 2：SE3估计**。对第 $f$ 帧（$f > 0$），通过Procrustes分析求解当前帧到前一帧的刚体变换：

$$\min_{R, \mathbf{t}} \sum_i \|\hat{\mathbf{c}}_f^{(i)} - (R \hat{\mathbf{c}}_{f-1}^{(i)} + \mathbf{t})\|^2$$

通过SVD分解求解：$H = \hat{\mathbf{c}}_{f-1}^T \hat{\mathbf{c}}_f$，$U \Sigma V^T = \text{SVD}(H)$，$R = V U^T$，$\mathbf{t} = \bar{\mathbf{c}}_f - R \bar{\mathbf{c}}_{f-1}$。

**Step 3：异常帧检测**。若某帧的对齐误差 $e_f > 3 \times \text{median}(e_{1:f-1})$，则拒绝该帧的变换，保持上一帧的累积变换不变。若整个chunk中被接受的变换数量不足15%，则回退到`anchor_stream`模式重新推理。

**Step 4：累积变换**。将逐帧的局部变换累积为相对于第一帧的全局变换：

$$R_{\text{cum}}^{(f)} = R_{\text{cum}}^{(f-1)} \cdot R_f, \quad \mathbf{t}_{\text{cum}}^{(f)} = R_{\text{cum}}^{(f-1)} \mathbf{t}_f + \mathbf{t}_{\text{cum}}^{(f-1)}$$

### 3.3.3 Chunk间SIM3全局对齐

#### 3.3.3.1 重叠区域点云配准

相邻chunk $m$ 和 $m+1$ 共享 $O$ 帧重叠区域。从两个chunk的重叠帧中分别提取锚点相机对应的3D点云 $\mathcal{P}_m^{\text{overlap}}$ 和 $\mathcal{P}_{m+1}^{\text{overlap}}$，以及对应的置信度 $w_m, w_{m+1}$。仅使用锚点相机（如俯视相机3）的点云进行配准，避免多视角方向混合导致配准不稳定。

#### 3.3.3.2 加权Sim3估计

通过置信度加权的点云配准求解相似变换 $(s, R, \mathbf{t})$：

$$\min_{s, R, \mathbf{t}} \sum_i w_i \| s R \mathbf{p}_m^{(i)} + \mathbf{t} - \mathbf{p}_{m+1}^{(i)} \|^2$$

求解过程分三步：

1. **质心计算**：$\bar{\mathbf{p}}_m = \frac{\sum w_i \mathbf{p}_m^{(i)}}{\sum w_i}$，$\bar{\mathbf{p}}_{m+1} = \frac{\sum w_i \mathbf{p}_{m+1}^{(i)}}{\sum w_i}$

2. **尺度估计**：$s = \sqrt{\frac{\sum w_i \|\mathbf{p}_{m+1}^{(i)} - \bar{\mathbf{p}}_{m+1}\|^2}{\sum w_i \|\mathbf{p}_m^{(i)} - \bar{\mathbf{p}}_m\|^2}}$

3. **旋转与平移**：对去质心、去尺度的点云做SVD：$H = \sum w_i (\mathbf{p}_m^{(i)} - \bar{\mathbf{p}}_m)(\mathbf{p}_{m+1}^{(i)} - \bar{\mathbf{p}}_{m+1})^T$，$U \Sigma V^T = \text{SVD}(H)$，$R = V U^T$，$\mathbf{t} = \bar{\mathbf{p}}_{m+1} - s R \bar{\mathbf{p}}_m$

#### 3.3.3.3 异常检测与累积

基于对齐误差的历史统计判断当前变换是否为异常值。若误差 $e_m > 3.2 \times \text{median}(e_{1:m-1})$，则回退到上一个稳定的变换。累积变换通过链式复合实现：

$$s_{\text{cum}}^{(m)} = s_{\text{cum}}^{(m-1)} \cdot s_m, \quad R_{\text{cum}}^{(m)} = R_{\text{cum}}^{(m-1)} \cdot R_m, \quad \mathbf{t}_{\text{cum}}^{(m)} = s_{\text{cum}}^{(m-1)} R_{\text{cum}}^{(m-1)} \mathbf{t}_m + \mathbf{t}_{\text{cum}}^{(m-1)}$$

#### 3.3.3.4 接缝精修

对相邻chunk的重叠区域进行局部SE3精配准，计算微小修正变换 $(\Delta R, \Delta \mathbf{t})$。仅当修正量满足以下条件时才应用：

$$\text{error} < 0.45, \quad |1 - s| < 0.03, \quad \theta(R) < 3°, \quad \|\mathbf{t}\| < 1.5$$

### 3.3.4 回环检测与Sim3图优化

#### 3.3.4.1 视觉位置识别

基于SALAD（Self-Attentive Local Aggregation Descriptor）模型进行回环检测。对每个chunk的关键帧提取全局描述子 $\mathbf{d}_m \in \mathbb{R}^{D}$，使用FAISS构建描述子索引，通过余弦相似度检索与当前帧最相似的历史帧。当相似度超过阈值且帧间距大于最小间距时，判定为回环候选。

#### 3.3.4.2 回环约束生成

对检测到的回环对 $(m, n)$，提取两个chunk重叠区域的3D点云，通过加权Sim3配准计算回环约束 $(s_{mn}, R_{mn}, \mathbf{t}_{mn})$。

#### 3.3.4.3 Sim3图优化

将所有chunk间的顺序约束和回环约束构建为Sim3位姿图，使用PyPose库进行图优化。优化目标为：

$$\min_{\{s_m, R_m, \mathbf{t}_m\}} \sum_{(m,n) \in \mathcal{E}} \left\| \log\left( T_n^{-1} T_m T_{mn} \right) \right\|^2_{\Sigma_{mn}}$$

其中 $\mathcal{E}$ 为约束边集合（包括顺序边和回环边），$T_m = (s_m, R_m, \mathbf{t}_m)$ 为第 $m$ 个chunk的Sim3位姿，$T_{mn}$ 为观测到的相对变换，$\Sigma_{mn}$ 为信息矩阵。优化变量在Sim3流形上参数化，通过高斯-牛顿法或Levenberg-Marquardt法求解。

### 3.3.5 DSM级评测指标体系

#### 3.3.5.1 深度图到DSM栅格化

评测不在深度图层面进行，而是将深度图反投影为3D点云后栅格化为DSM（Digital Surface Model）。具体流程为：

1. **3D反投影**：使用GT相机参数将预测深度图反投影为世界坐标3D点
2. **DSM栅格化**：将3D点投影到2D网格（GSD=0.2m），多帧多视角逐网格累加取均值
3. **DSM融合**：5个相机×数百帧的深度图融合为一张预测DSM

#### 3.3.5.2 指标定义

**PAG（Percentage of Accurate Grids）**：高程误差小于阈值 $\alpha$ 的网格点占全部有效网格点的百分比：

$$\text{PAG}_{\alpha} = \frac{|\{i : |z_i^{\text{pred}} - z_i^{\text{gt}}| < \alpha\}|}{N_{\text{valid}}} \times 100\%$$

分母 $N_{\text{valid}}$ 包含异常值，异常值直接降低PAG，惩罚不鲁棒的方法。

**MAE（Mean Absolute Error）**：排除异常值（误差>20m）后的平均绝对误差：

$$\text{MAE} = \frac{1}{N_{\text{within}}} \sum_{i \in \text{within}} |z_i^{\text{pred}} - z_i^{\text{gt}}|$$

**RMSE（Root Mean Square Error）**：排除异常值后的均方根误差：

$$\text{RMSE} = \sqrt{\frac{1}{N_{\text{within}}} \sum_{i \in \text{within}} (z_i^{\text{pred}} - z_i^{\text{gt}})^2}$$

此外还计算 abs_rel、sq_rel、rmse_log、log10、silog、$\delta_1, \delta_2, \delta_3$ 等深度估计领域标准指标。

---

## 3.4 数据处理与训练流程设计

### 3.4.1 数据集

本项目使用两个核心数据集：

**（1）WHU-OMVS数据集**：无人机倾斜摄影测量数据集，包含5个视角（1垂直+4倾斜）的连续帧图像序列，GSD为0.2m。训练集包含area1/area4/area5/area6，测试集包含area2/area3。每个场景提供GT DSM（.tif格式）、相机参数（.txt格式）和深度真值（.exr格式）。

**（2）MatrixCity数据集**：大规模合成城市数据集，包含big_city和small_city两个变体，提供航拍图像、相机位姿（transforms.json）和深度图（.exr格式），用于DA3模型的LoRA微调。

### 3.4.2 推理数据处理流程

#### 3.4.2.1 图像索引构建

扫描数据集目录，按相机ID组织图像路径。对每个相机目录，按文件名排序得到该相机的图像序列。支持通过 `camera_ids` 和 `num_cameras_to_use` 筛选使用的相机。

#### 3.4.2.2 图像预处理

各模型的预处理流程略有差异：

- **DepthAnything3**：resize到`process_res`（504），ImageNet归一化，外参归一化（第一帧为原点，平移除以中位距离）
- **Pi3**：resize到固定分辨率，转为张量 `[N, 3, H, W]`
- **VGGT**：`load_and_preprocess_images()` 标准化处理
- **MapAnything**：`preprocess_input_views_for_inference()` 处理可选几何输入

#### 3.4.2.3 后处理与格式统一

模型推理后经过两层格式归一化（详见3.3.1），最终输出4元组 `(world_points, conf, images, depth)`，供下游对齐和融合流程使用。

### 3.4.3 训练流程设计

#### 3.4.3.1 Ada-MVS训练流程

Ada-MVS采用级联MVS架构，在WHU-OMVS数据集上训练。数据处理流程为：

1. **数据加载**：`MVSDataset` 从 `viewpair.txt` 读取视角配对，加载5视角图像、相机参数、深度真值和掩码
2. **数据增强**：训练时随机颜色抖动
3. **图像归一化**：`center_image(mode='mean')` 做均值方差归一化
4. **坐标系变换**：Twc→Tcw（求逆），XrightYup→XrightYdown（乘对角矩阵 $O = \text{diag}(1,-1,-1)$）
5. **投影矩阵构建**：$\Pi = K[R|\mathbf{t}]$，多尺度缩放（4×/2×/1×）

损失函数为级联多阶段监督损失：

$$\mathcal{L}_{\text{total}} = \sum_{s=1}^{3} w_s \cdot \left( \mathcal{L}_{\text{depth}}^{(s)} + \mathcal{L}_{\text{pair}}^{(s)} \right)$$

其中 $w_s \in \{0.5, 1.0, 2.0\}$ 为阶段权重，$\mathcal{L}_{\text{depth}}^{(s)}$ 为融合深度的Smooth L1损失，$\mathcal{L}_{\text{pair}}^{(s)}$ 为各视角对深度的Smooth L1损失均值：

$$\mathcal{L}_{\text{depth}}^{(s)} = \text{SmoothL1}\left(\hat{d}^{(s)}[\mathcal{M}], d_{\text{gt}}^{(s)}[\mathcal{M}]\right)$$

$$\mathcal{L}_{\text{pair}}^{(s)} = \frac{1}{N_v - 1} \sum_{i=1}^{N_v - 1} \text{SmoothL1}\left(\hat{d}_i^{(s)}[\mathcal{M}], d_{\text{gt}}^{(s)}[\mathcal{M}]\right)$$

其中 $\mathcal{M}$ 为有效像素掩码，$N_v$ 为视角数。深度回归采用soft argmin：

$$\hat{d} = \sum_{k=1}^{D} \text{softmax}(\mathbf{c})_k \cdot d_k$$

其中 $\mathbf{c}$ 为正则化后的代价体，$d_k$ 为第 $k$ 个深度假设值。

#### 3.4.3.2 DA3 LoRA微调流程

为提升DepthAnything3在无人机视角下的深度估计精度，采用LoRA（Low-Rank Adaptation）进行参数高效微调。在MatrixCity数据集上训练，仅更新LoRA注入的低秩矩阵，冻结原始模型参数。

**LoRA原理**：对模型中的线性层 $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$，注入低秩分解：

$$W_0' = W_0 + \Delta W = W_0 + B A, \quad B \in \mathbb{R}^{d_{\text{out}} \times r}, \quad A \in \mathbb{R}^{r \times d_{\text{in}}}$$

其中 $r \ll \min(d_{\text{out}}, d_{\text{in}})$ 为LoRA秩，$A$ 初始化为零矩阵，$B$ 使用高斯初始化。训练时仅更新 $A$ 和 $B$，参数量从 $d_{\text{out}} \times d_{\text{in}}$ 降至 $r \times (d_{\text{out}} + d_{\text{in}})$。

**微调损失函数**为深度损失和位姿损失的加权和：

$$\mathcal{L} = \lambda_d \mathcal{L}_{\text{depth}} + \lambda_p \mathcal{L}_{\text{pose}}$$

深度损失包含三个分量：

$$\mathcal{L}_{\text{depth}} = \lambda_{\text{l1}} \cdot \text{L1}(\hat{d}, d_{\text{gt}}) + \lambda_{\text{silog}} \cdot \text{SiLog}(\hat{d}, d_{\text{gt}}) + \lambda_{\text{smooth}} \cdot \mathcal{L}_{\text{smooth}}$$

其中尺度不变对数损失（SiLog）定义为：

$$\text{SiLog} = \frac{1}{N} \sum_i (\log \hat{d}_i - \log d_i)^2 - \frac{\alpha}{N^2} \left(\sum_i (\log \hat{d}_i - \log d_i)\right)^2$$

边缘感知平滑损失定义为：

$$\mathcal{L}_{\text{smooth}} = \sum_{u,v} \left| \nabla_x \hat{d} \right| e^{-|\nabla_x I|} + \left| \nabla_y \hat{d} \right| e^{-|\nabla_y I|}$$

位姿损失包含旋转测地线距离和平移方向L1损失：

$$\mathcal{L}_{\text{pose}} = \lambda_R \cdot \text{arccos}\left(\frac{\text{tr}(R_{\text{pred}}^T R_{\text{gt}}) - 1}{2}\right) + \lambda_t \cdot \left\| \frac{\mathbf{t}_{\text{pred}}}{\|\mathbf{t}_{\text{pred}}\|} - \frac{\mathbf{t}_{\text{gt}}}{\|\mathbf{t}_{\text{gt}}\|} \right\|_1$$

---

## 3.5 实现环境与参数设置

### 3.5.1 硬件环境

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA A100 80GB × 1 |
| CPU | Intel Xeon |
| 内存 | 256GB DDR4 |
| 存储 | 4TB NVMe SSD |

### 3.5.2 软件环境

| 项目 | 版本 |
|------|------|
| 操作系统 | Ubuntu 20.04 LTS |
| Python | 3.8 / 3.10 |
| PyTorch | 2.0+ |
| CUDA | 11.8 |
| DinoV2 | ViT-L/14 |
| PyPose | 0.x (Sim3优化) |
| FAISS | 1.7+ (回环检索) |
| trimesh | 3.x (点云IO) |
| numba | 0.x (加速) |

### 3.5.3 推理参数设置

#### 3.5.3.1 通用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `chunk_size` | 150 | 每个chunk包含的帧数 |
| `overlap` | 72 | 相邻chunk的重叠帧数 |
| `conf_threshold_coef` | 1.5 | 置信度阈值系数，阈值 = mean(conf) × coef |
| `sample_ratio` | 4 | 点云下采样比率 |
| `delete_temp_files` | True | 是否删除中间预测文件 |

#### 3.5.3.2 模型特定参数

| 参数 | DepthAnything3 | Pi3 | VGGT | MapAnything |
|------|---------------|-----|------|-------------|
| 推理模式 | `framewise_multicam` | 默认 | 默认 | 默认 |
| 输入分辨率 | 504 | 原始 | 原始 | 512 |
| Backbone | DinoV2 ViT-L | DinoV2 ViT-L | DinoV2 ViT-L | DinoV2 ViT-L |
| 深度假设数 | — | — | — | — |

#### 3.5.3.3 对齐参数

| 参数 | 值 | 说明 |
|------|-----|------|
| SE3异常阈值 | $3 \times \text{median}(e)$ | 帧间拼接异常检测 |
| Sim3异常阈值 | $3.2 \times \text{median}(e)$ | chunk间对齐异常检测 |
| 接缝精修误差阈值 | 0.45 | 修正量误差上限 |
| 接缝精修尺度阈值 | 0.03 | 尺度偏移上限 |
| 接缝精修旋转阈值 | 3° | 旋转偏移上限 |
| 接缝精修平移阈值 | 1.5 | 平移偏移上限 |

### 3.5.4 训练参数设置

#### 3.5.4.1 Ada-MVS训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `epochs` | 80 | 训练轮数 |
| `lr` | 0.001 | 初始学习率 |
| `optimizer` | RMSprop | 优化器（alpha=0.9） |
| `batch_size` | 1 | 批大小 |
| `ndepths` | 48,32,8 | 三阶段深度假设数 |
| `depth_intervals_ratio` | 4,2,1 | 三阶段深度间隔比 |
| `dlossw` | 0.5,1.0,2.0 | 三阶段损失权重 |
| `view_num` | 5 | 视角数 |
| `interval_scale` | 1.0 | 深度间隔缩放 |

#### 3.5.4.2 DA3 LoRA微调参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `model_name` | da3-large | 预训练模型 |
| `lora_r` | 8 | LoRA秩 |
| `lora_alpha` | 16.0 | LoRA缩放因子 |
| `lora_dropout` | 0.0 | LoRA dropout |
| `epochs` | 10 | 微调轮数 |
| `lr` | 1e-4 | 学习率 |
| `weight_decay` | 0.01 | 权重衰减 |
| `warmup_steps` | 500 | 学习率预热步数 |
| `max_grad_norm` | 1.0 | 梯度裁剪范数 |
| `depth_loss_weight` | 1.0 | 深度损失权重 |
| `pose_loss_weight` | 0.5 | 位姿损失权重 |
| `num_views` | 2 | 每样本视角数 |
| `image_size` | 504 | 训练图像尺寸 |
| `stride` | 10 | 帧采样步长 |
| `max_depth` | 500.0 | 深度归一化上限 |

---

## 3.6 本章小结

本章详细阐述了无人机视角下前馈式三维重建系统的方案设计与实现。首先，明确了将长序列无人机图像回归至三维几何表示的核心需求，并形式化为Sim3变换估计的优化问题。其次，设计了"分块推理—局部拼接—全局对齐—回环优化—评测输出"的五阶段流水线架构，通过适配器模式实现了四种前馈模型的统一推理接口，通过分而治之策略解决了长序列推理的显存瓶颈。

在关键模块方面，本章详细设计了：统一多模型推理适配层（6键字典+双向补全）、帧间SE3拼接（尺度归一化+Procrustes分析+异常检测）、Chunk间Sim3全局对齐（加权点云配准+异常剔除+接缝精修）、回环检测与Sim3图优化（SALAD视觉位置识别+PyPose图优化）、DSM级评测指标体系（PAG/MAE/RMSE）。

在数据处理与训练方面，本章描述了WHU-OMVS和MatrixCity两个数据集的处理流程，以及Ada-MVS的级联多阶段监督训练和DA3的LoRA参数高效微调方案，给出了完整的损失函数公式和参数设置。整体系统在保证重建精度的前提下，实现了从1560张连续图像帧到完整三维点云的端到端自动重建。
