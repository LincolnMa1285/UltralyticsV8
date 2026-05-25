# 无人机动态扩展目标检测项目规划文档

基于 Ultralytics YOLOv8 8.48 源码，支持**类别逐步扩充、数据集动态增加、增量训练**的完整工程化方案

---

## 一、项目背景与设计目标

### 1.1 背景

- 基于 GitHub 下载的 Ultralytics YOLOv8 8.48 源码进行二次开发
- 训练数据集 **逐步扩充** ，检测目标**动态增加**
- 需支持：新增类别、新增数据、复用历史模型、避免重复训练、工程化可维护

### 1.2 核心设计原则

1. **类别与代码解耦** ：新增类别只改配置，不动源码
2. **数据集版本化** ：按批次 / 版本管理数据，支持合并
3. **增量训练** ：基于旧模型微调，不从头训练
4. **可追溯可复用** ：模型、配置、数据统一版本管理
5. **无人机场景适配** ：小目标、运动模糊、多尺度、视角变化

---

## 二、完整项目目录结构

plaintext

```
drone-yolov8-detection/
├── configs/                   # 所有配置（核心扩展）
│   ├── categories/            # 类别版本配置
│   │   ├── v1.0.yaml
│   │   ├── v2.0.yaml
│   │   └── all_categories.yaml
│   ├── dataset_versions/       # 数据集版本配置
│   │   ├── drone_v1.yaml
│   │   ├── drone_v2.yaml
│   │   └── drone_full.yaml
│   ├── train_configs/          # 训练/增量训练配置
│   │   ├── train_v1.yaml
│   │   ├── train_v2_incremental.yaml
│   │   └── train_full.yaml
│   └── hyp_drone_base.yaml     # 无人机通用超参
├── data/
│   ├── raw/                    # 原始数据（按版本）
│   │   ├── v1.0/
│   │   └── v2.0/
│   ├── processed/              # 预处理后数据
│   │   ├── v1.0/
│   │   ├── v2.0/
│   │   └── full/
│   ├── augmentations/          # 增强脚本
│   └── metadata/               # 类别映射、版本日志
├── ultralytics/                # YOLOv8 8.48 源码（不动）
├── scripts/                    # 自动化脚本
│   ├── data_preprocess.py      # 数据预处理
│   ├── data_merge.py           # 多版本数据合并
│   ├── category_manage.py      # 类别配置管理
│   ├── train_drone.py          # 常规训练
│   ├── incremental_train.py    # 增量训练（核心）
│   ├── eval_drone.py           # 评估（含小目标AP）
│   ├── export_model.py         # 模型导出
│   └── visualize_results.py     # 训练曲线/检测结果可视化
├── runs/                       # 训练输出（按版本）
│   ├── train_v1/
│   ├── train_v2_incremental/
│   └── train_full/
├── weights/                    # 模型权重（版本化）
│   ├── pretrained/
│   ├── v1.0/
│   ├── v2.0/
│   └── full/
├── utils/                      # 工具库
├── requirements.txt
├── VERSION.md                  # 版本迭代日志
└── README.md
```

---

## 三、配置体系设计（支持动态扩展）

### 3.1 类别配置（categories/）

每个版本独立维护， **ID 全局唯一不冲突** 。

#### v1.0.yaml（示例）

yaml

```
nc: 2
names:
  0: pedestrian
  1: car
id_mapping:
  pedestrian: 0
  car: 1
```

#### v2.0.yaml（新增类别示例）

yaml

```
nc: 3
names:
  0: pedestrian
  1: car
  2: drone
id_mapping:
  pedestrian: 0
  car: 1
  drone: 2
version_trace:
  pedestrian: v1.0
  car: v1.0
  drone: v2.0
```

#### all_categories.yaml（自动合并全类别）

统一总类别数、名称、ID 映射、溯源信息。

---

### 3.2 数据集版本配置（dataset_versions/）

#### drone_v2.yaml（增量数据集）

yaml

```
path: ./data/processed/v2.0
train: images/train
val: images/val
test: images/test
category_config: ./configs/categories/v2.0.yaml
incremental: True
depends_on: v1.0
```

#### drone_full.yaml（全量合并数据集）

yaml

```
path: ./data/processed/full
train: images/train
val: images/val
test: images/test
category_config: ./configs/categories/all_categories.yaml
incremental: False
```

---

### 3.3 增量训练配置（train_configs/）

#### train_v2_incremental.yaml

yaml

```
history_model: ./weights/v1.0/best.pt
category_config: ./configs/categories/v2.0.yaml
data: ./configs/dataset_versions/drone_v2.yaml
epochs: 50
batch: 16
imgsz: 640
lr0: 0.01
lrf: 0.01
patience: 10
device: 0
project: ./runs
name: train_v2_incremental
version: v2.0
freeze: [0,1,2]  # 冻结主干，防止旧类别遗忘
```

---

## 四、核心脚本功能说明

### 4.1 类别管理脚本

`scripts/category_manage.py`

- 加载指定版本类别配置
- 自动合并所有版本 → 生成 `all_categories.yaml`
- 维护全局唯一 ID 映射表
- 支持新增类别后一键更新

### 4.2 数据合并脚本

`scripts/data_merge.py`

- 按版本合并图片与标注
- 自动重命名避免文件名冲突
- 生成全量数据集配置

### 4.3 增量训练脚本（核心）

`scripts/incremental_train.py`

- 加载历史模型
- 自动适配新类别数
- 降低学习率、冻结主干网络
- 防止灾难性遗忘
- 输出新版本权重

### 4.4 其他标准脚本

- `data_preprocess.py`：格式转 YOLO、划分数据集、无人机专属增强（小目标、运动模糊）
- `train_drone.py`：全量从头训练
- `eval_drone.py`：mAP、Precision、Recall、**小目标 AP**
- `export_model.py`：导出 ONNX / TensorRT
- `visualize_results.py`：训练曲线、检测效果图

---

## 五、数据集与类别扩充标准流程

当你需要**新增一类 / 一批数据**时，严格按以下步骤：

1. **新建类别版本配置**

    在 `configs/categories/` 新建 `vX.X.yaml`，继承旧 ID，只追加新类别。

2. **自动合并全类别**
   bash

    运行

    ```
    python scripts/category_manage.py
    ```

3. **放入新原始数据**

    放入 `data/raw/vX.X/`

4. **预处理新数据**
   bash

    运行

    ```
    python scripts/data_preprocess.py
    ```

5.（可选）合并到全量数据集

bash

运行

```
python scripts/data_merge.py
```

6. **执行增量训练**

    编写 `train_vX.X_incremental.yaml`
    bash

    运行

    ```
    python scripts/incremental_train.py
    ```

7. **评估 & 导出 & 可视化**
8. **更新 VERSION.md**

---

## 六、版本日志规范（VERSION.md）

plaintext

```
# 无人机目标检测模型版本日志

## v1.0 2026-xx-xx
- 类别：pedestrian, car
- 数据量：xxx 张
- 训练方式：从头训练 yolov8n
- mAP@0.5：xx.xx%

## v2.0 2026-xx-xx
- 新增类别：drone
- 新增数据：xxx 张
- 训练方式：增量训练（基于 v1.0）
- 总类别：3
- mAP@0.5：xx.xx%
```

---

## 七、无人机场景专项优化（内置支持）

1. **小目标优化**
    - 放大增强
    - 调整 box loss 增益
    - 专用小目标 AP 评估
2. **运动模糊优化**
    - 模拟无人机抖动模糊增强
3. **视角变化优化**
    - 旋转、透视、缩放增强
4. **部署轻量化**
    - 支持 FP16 / INT8 量化
    - 导出 ONNX / TensorRT

---

## 八、使用环境（固定不变）

- Python 3.10
- PyTorch 2.0+
- CUDA 11.8+
- Ultralytics 8.48（源码 editable 安装）

---

## 九、项目优势总结

1. **完全适配动态扩充数据集 / 类别**
2. **增量训练，不从头训，节省大量算力**
3. **配置与代码完全分离，便于多人协作**
4. **版本可追溯、模型可复用**
5. **无人机场景深度定制**
6. **工业级工程结构，可直接上线落地**
