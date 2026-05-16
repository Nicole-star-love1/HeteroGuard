# HeteroGuard

**Defending Heterogeneous Graph Neural Networks against Backdoor Attacks**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

## 📝 论文信息

**标题**: HeteroGuard: HeteroGuard: Trigger-Aware Backdoor Unlearning for Heterogeneous Graph Neural Networks  
**会议/期刊**: (待补充)  
**作者**: (待补充)  
**arXiv**: (待补充)  

## 📋 项目简介

HeteroGuard 是一个用于防御异构图神经网络（Heterogeneous Graph Neural Networks, HGNNs）后门攻击的框架。该项目提供了：

- **多种后门攻击方法**：特征攻击、SBA攻击、UBA攻击、关系攻击、干净标签攻击、CBA攻击、梯度攻击
- **防御机制**：HeteroGuard 检测器与防御训练
- **标准化实验流程**：集成式实验运行脚本，支持可复现研究
- **多数据集支持**：DBLP、ACM、IMDB、Freebase 等异构图数据集
- **多模型支持**：HAN、HGT、RGCN、HeteroSAGE 等异构GNN模型

### 主要特性

✅ **统一的攻击接口**：支持多种后门攻击方法的公平比较  
✅ **内存高效的实验设计**：避免模型检查点频繁保存/加载，支持大型图数据集  
✅ **可复现的实验**：支持随机种子设置，确保实验结果可复现  
✅ **全面的评估指标**：干净准确率、攻击成功率(ASR)、检测精度、召回率、F1分数等  
✅ **防御效果验证**：提供完整的防御前后性能对比  

## 🚀 快速开始

### 环境要求

- Python >= 3.8
- PyTorch >= 1.12.0
- PyTorch Geometric >= 2.0.0
- CUDA (可选，用于GPU加速)

### 安装步骤

1. **克隆仓库**
```bash
git clone https://github.com/your-username/HeteroGuard.git
cd HeteroGuard
```

2. **创建虚拟环境** (推荐)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **安装 PyTorch Geometric** (如果上面的命令没有自动安装)
```bash
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```
注意：将 `${TORCH}` 和 `${CUDA}` 替换为你的 PyTorch 和 CUDA 版本，具体请参考 [PyG 官方安装指南](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

### 数据准备

项目支持以下异构图数据集：
- **DBLP**：学术论文网络（作者、论文、会议）
- **ACM**：学术网络（论文、作者、主题）
- **IMDB**：电影网络（电影、演员、导演）
- **Freebase**：知识图谱子集

数据集将自动下载到 `./data` 目录。你也可以手动下载并放置到该目录。

```bash
# 数据会自动下载，无需手动操作
# 如果需要手动下载，请参考各数据集的官方来源：
# DBLP: https://www.dm.dblp.org/
# ACM: https://dl.acm.org/
# IMDB: https://www.imdb.com/
```

## 🔬 实验使用

### 基本使用

#### 1. 快速测试（烟雾测试）

```bash
# 使用 HAN 模型在 ACM 数据集上测试关系攻击
bash experiments/smoke_han_cross_dataset.sh

# 测试防御套件
bash experiments/smoke_ablation_acm_han.sh
```

#### 2. 完整实验运行

##### 攻击实验
```bash
# 在 ACM 数据集上使用 HAN 模型运行关系攻击
python -m experiments.run_integrated \
  --dataset ACM \
  --model HAN \
  --attack relation \
  --poison_rate 0.2 \
  --trigger_size 10 \
  --epochs 200 \
  --debug \
  --save_results
```

##### 防御实验
```bash
# 运行完整的防御套件
python -m experiments.run_integrated \
  --dataset ACM \
  --model HAN \
  --attack relation \
  --poison_rate 0.2 \
  --trigger_size 10 \
  --epochs 200 \
  --run_defense \
  --save_results
```

#### 3. 批量实验脚本

项目提供了多个批量实验脚本：

```bash
# ACM + HAN 主实验（5个随机种子）
bash experiments/launch_acm_han_5seeds.sh

# 防御套件实验（5个随机种子）
bash experiments/launch_acm_han_defense_suite_5seeds.sh

# 跨数据集实验
bash experiments/launch_han_cross_dataset_5seeds.sh

# 模型泛化实验
bash experiments/launch_model_generalization_5seeds.sh

# 鲁棒性实验
bash experiments/launch_robustness_acm_han_5seeds.sh

# 可扩展性实验（OGBMAG，3个随机种子）
bash experiments/launch_scalability_ogbn_mag_3seeds.sh

# 敏感性分析
bash experiments/launch_sensitivity_acm_han_5seeds.sh

# 消融实验
bash experiments/launch_ablation_acm_han_5seeds.sh
```

### 高级配置

#### 攻击类型

| 攻击方法 | 参数值 | 说明 |
|---------|--------|------|
| 特征攻击 | `feature` | 修改节点特征 |
| SBA攻击 | `sba` | 基于结构的后门攻击 |
| UBA攻击 | `uba` | 非目标后门攻击 |
| 关系攻击 | `relation` | 通过修改图结构进行攻击 |
| 干净标签攻击 | `clean_label` | 保持原始标签的攻击 |
| CBA攻击 | `cba` | 基于聚类的后门攻击 |
| 梯度攻击 | `grad` | 基于梯度优化的攻击 |

#### 模型选择

| 模型 | 参数值 | 说明 |
|-----|--------|------|
| HAN | `HAN` | Heterogeneous Graph Attention Network |
| HGT | `HGT` | Heterogeneous Graph Transformer |
| RGCN | `RGCN` | Relational Graph Convolutional Network |
| HeteroSAGE | `HeteroSAGE` | Heterogeneous GraphSAGE |

#### 主要参数说明

```bash
# 基本参数
--dataset          数据集名称 [DBLP, ACM, IMDB, Freebase]
--model            模型名称 [HAN, HGT, RGCN, HeteroSAGE]
--attack           攻击方法 [feature, sba, uba, relation, clean_label, cba, grad]
--seed             随机种子 (默认: 42)

# 攻击参数
--poison_rate      中毒率 (默认: 0.2)
--trigger_size     触发器大小 (默认: 10)
--target_class     目标类别 (默认: 0)
--trigger_strength 触发器强度 (默认: 3.0)

# 训练参数
--epochs           训练轮数 (默认: 200)
--lr               学习率 (默认: 0.01)
--weight_decay     权重衰减 (默认: 5e-4)
--patience         早停耐心值 (默认: 50)

# 防御参数
--run_defense      启用防御
--pretrain_epochs  预训练轮数 (默认: 50)
--defense_epochs   防御训练轮数 (默认: 100)
--detection_ratio  检测比例 (默认: 与poison_rate相同)

# 其他
--device           设备 [cuda, cpu] (默认: cuda)
--debug            调试模式
--verbose          详细输出
--save_results     保存结果到JSON
--output_dir       输出目录 (默认: ./results)
```

### 实验结果

实验结果将保存到 `./results` 目录，包括：

- **JSON 文件**：详细的实验指标
- **日志文件**：训练过程中的日志
- **CSV 文件**：汇总实验结果（用于LaTeX表格）

示例结果文件命名：
```
integrated_ACM_HAN_relation_r0.20_seed42_20260101_120000.json
```

## 📁 项目结构

```
HeteroGuard/
├── attack/                 # 攻击方法实现
│   ├── __init__.py
│   ├── base.py            # 攻击基类
│   ├── feature_attack.py  # 特征攻击
│   ├── sba_attack.py      # SBA攻击
│   ├── uba_attack.py      # UBA攻击
│   ├── relation_attack.py # 关系攻击
│   ├── clean_label_attack.py
│   ├── cba_attack.py
│   ├── grad_attack.py
│   └── hetero_attack.py  # 异构攻击接口
│
├── defense/               # 防御方法实现
│   ├── __init__.py
│   ├── hetero_guard.py   # HeteroGuard 主防御类
│   ├── detector.py       # 检测器
│   ├── embedding_detector.py
│   ├── feature_detector.py
│   ├── structural_detector.py
│   ├── trainer.py         # 防御训练器
│   └── utils.py
│
├── models/                # GNN 模型实现
│   ├── __init__.py
│   ├── han.py            # HAN 模型
│   ├── hgt.py            # HGT 模型
│   ├── rgcn.py           # RGCN 模型
│   ├── hetero_gnn.py     # 异构GNN工厂
│   └── heterosage.py     # HeteroSAGE 模型
│
├── data/                  # 数据处理
│   └── hetero_dataset.py # 异构图数据集加载
│
├── experiments/           # 实验脚本
│   ├── run_integrated.py # 集成实验运行器
│   ├── run_defense_suite.py
│   ├── run_ablation_suite.py
│   ├── run_robustness_suite.py
│   ├── run_scalability_suite.py
│   ├── utils.py          # 实验工具函数
│   └── launch_*.sh       # 批量实验启动脚本
│
├── results/               # 实验结果（不提交到Git）
│
├── data/                  # 数据集（不提交到Git）
│
├── requirements.txt       # Python 依赖
├── .gitignore           # Git 忽略文件
├── LICENSE              # MIT 许可证
└── README.md           # 本文件
```

## 📊 复现论文结果

### 主要实验结果

要复现论文中的主要实验结果，请运行：

```bash
# 1. 主实验表格（表1）
bash experiments/launch_acm_han_5seeds.sh
bash experiments/launch_han_cross_dataset_5seeds.sh

# 2. 防御效果（表2）
bash experiments/launch_acm_han_defense_suite_5seeds.sh

# 3. 模型泛化（表3）
bash experiments/launch_model_generalization_5seeds.sh

# 4. 鲁棒性分析（表4）
bash experiments/launch_robustness_acm_han_5seeds.sh

# 5. 敏感性分析（表5）
bash experiments/launch_sensitivity_acm_han_5seeds.sh

# 6. 消融实验（表6）
bash experiments/launch_ablation_acm_han_5seeds.sh

# 7. 可扩展性实验（表7）
bash experiments/launch_scalability_ogbn_mag_3seeds.sh
```

所有实验将使用多个随机种子运行，确保统计显著性。

### 结果汇总

实验完成后，可以使用以下命令汇总结果：

```bash
# 汇总所有实验结果
python -m experiments.utils summarize_results --results_dir ./results --output paper_tables
```

## 🔍 核心算法说明

### HeteroGuard 防御流程

1. **预训练参考模型**：在干净图上预训练参考模型
2. **异常检测**：通过比较干净模型和中毒模型的嵌入，检测可疑节点
3. **图净化**：移除或降低可疑边的权重
4. **防御训练**：在净化后的图上重新训练模型
5. **触发器反学习**（可选）：通过反学习进一步消除触发器影响

### 攻击方法简介

- **Feature Attack**：修改目标节点的特征，使其与触发器特征相似
- **SBA Attack**：在图中注入特定的子图结构作为触发器
- **UBA Attack**：非目标攻击，使模型在触发器存在时预测任意错误类别
- **Relation Attack**：通过修改节点间的关系类型实现攻击
- **Clean Label Attack**：保持原始标签，但在特征空间中嵌入触发器

## ⚠️ 常见问题

### 1. CUDA 内存不足
**解决方案**：
- 减小 `batch_size`（如果支持）
- 使用更小的模型（`hidden_dim=64` 而不是 `128`）
- 在 CPU 上运行：`--device cpu`

### 2. 数据集下载失败
**解决方案**：
- 检查网络连接
- 手动下载数据集并放置到 `./data` 目录
- 使用代理下载

### 3. 实验结果不一致
**解决方案**：
- 确保使用了相同的随机种子 `--seed`
- 设置 PyTorch 确定性标志（代码中已包含）
- 多次运行取平均值

### 4. 依赖冲突
**解决方案**：
```bash
# 创建全新的虚拟环境
python -m venv clean_env
source clean_env/bin/activate
pip install -r requirements.txt
```

## 📚 引用

如果您在研究中使用了本项目，请引用我们的论文：

```bibtex
@article{heteroguard2026,
  title={HeteroGuard: Defending Heterogeneous Graph Neural Networks against Backdoor Attacks},
  author={},
  journal={},
  year={2026},
  publisher={}
}
```

## 🤝 贡献

欢迎贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开一个 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](./LICENSE) 文件了解详情。

## 📧 联系方式

如有问题或建议，请通过以下方式联系我们：

- 提交 Issue：https://github.com/your-username/HeteroGuard/issues
- 电子邮件：(待补充)

---

**致谢**：
- PyTorch Geometric 团队
- 各数据集的提供者
- (其他致谢)

---

**更新日志**：
- **2026-05-16**：初始版本发布
