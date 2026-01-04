# GraphSAGE-Bloodline: AML vs ALL Classification Using Graph Neural Networks

[![License:  MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c. svg)](https://pytorch.org/)

A comprehensive study evaluating Graph Neural Network (GNN) architectures for binary leukemia classification (Acute Myeloid Leukemia vs Acute Lymphoblastic Leukemia) using cell-level graph representations from peripheral blood smear microscopy images.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Findings](#key-findings)
- [Dataset](#dataset)
- [Graph Construction](#graph-construction)
- [Model Architectures](#model-architectures)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Future Work](#future-work)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## 🔬 Overview

This project introduces a **cell-level graph representation** for leukemia classification that combines morphological, categorical, and spatial information from microscopy images. Unlike traditional CNN-based approaches that treat images as regular grids, our graph-based method explicitly models **cell-to-cell relationships** and **spatial organization patterns** within blood smear images.

### Why Graph Neural Networks for Leukemia Classification?

- **Relational Modeling**: GNNs can capture interactions between different cell types in a blood smear
- **Spatial Awareness**: Graph edges encode spatial proximity between cells
- **Interpretability**: Attention-based models highlight which cells contribute most to classification decisions
- **Scalability**:  Inductive learning enables processing of varying numbers of cells per image

## 🎯 Key Findings

| Finding | Description |
|---------|-------------|
| **Best Model** | GAT (3-layer baseline) achieves **95.0% test accuracy** and **99.6% sensitivity** |
| **Attention Matters** | Attention-based message passing outperforms fixed aggregation methods |
| **Hierarchical Features** | Jumping Knowledge connections improve generalization across all architectures |
| **Virtual Nodes** | Explicit global context via virtual nodes does not improve performance |
| **Depth Trade-off** | Shallower networks (2-layer) often generalize better than deeper ones |

## 📊 Dataset

We use the **Blood Cancer Dataset (Leukemia)** from the Intelligent Machines Lab at Information Technology University (ITU), associated with MICCAI 2024. 

### Dataset Specifications

| Specification | Details |
|---------------|---------|
| **Subset Used** | H_100X_C2 |
| **Magnification** | 100× |
| **Camera Type** | High-quality microscope camera |
| **Classes** | AML (Acute Myeloid Leukemia), ALL (Acute Lymphoblastic Leukemia) |
| **Task** | Binary Classification |

### Naming Convention
Images follow the format: `H/L_100X/40X/10X_C1/C2`
- `H/L`: High-quality / Low-quality camera
- `100X/40X/10X`: Magnification level
- `C1/C2`: Mobile phone camera / Microscope camera

### Cell-Level Annotations
Each image includes JSON annotation files with:
- **Cell centroid coordinates** (x, y) in image space
- **Cell category label** based on expert labeling
- **Morphological attributes**:  cell area, perimeter, shape descriptors

## 🔗 Graph Construction

Each microscopy image is represented as an undirected graph **G = (V, E)**:

### Node Features
Each node (cell) has features combining:
- **Categorical features**: One-hot encoded cell type labels
- **Morphological features**: Area, perimeter, shape descriptors
- **Spatial features**:  Centroid coordinates

### Edge Construction
- Edges connect each cell to its **k-nearest neighbors** (Euclidean distance)
- **Edge weights** = inverse of pairwise Euclidean distance
- Captures both local cellular interactions and broader tissue organization

## 🧠 Model Architectures

We evaluate **15 GNN model variants** across three main architecture families:

### 1. Graph Isomorphism Networks (GIN)
- Maximally expressive for distinguishing graph structures
- Variants:  Baseline (2/3-layer), +Virtual Node, +Jumping Knowledge, +Attention Pooling

### 2. GraphSAGE
- Inductive neighborhood aggregation mechanism
- Scalable for biomedical graph learning
- Variants:  Baseline (2/3-layer), +Virtual Node, +Jumping Knowledge, +Attention Pooling

### 3. Graph Attention Networks (GAT)
- Learnable attention coefficients for neighborhood aggregation
- Dynamic weighting of neighbor importance
- Variants:  Baseline (2/3-layer), +Virtual Node, +Jumping Knowledge, +Attention Pooling

### Architectural Enhancements Studied

| Enhancement | Description |
|-------------|-------------|
| **Jumping Knowledge** | Aggregates representations from all layers for multi-scale features |
| **Virtual Nodes** | Global node connected to all others for context propagation |
| **Attention Pooling** | Learned attention weights for graph-level aggregation |

## 📈 Results

### Performance Comparison

| Model Family | Variant | Train Acc.  | Best Val.  Acc. | Test Acc. |
|--------------|---------|------------|----------------|-----------|
| **GAT** | 3-layer baseline | 0.9126 | 0.9587 | **0.9500** |
| GIN | + Jumping Knowledge | 0.9744 | 0.9413 | 0.9413 |
| GraphSAGE | + Jumping Knowledge | 0.9744 | 0.9326 | 0.9326 |
| GAT | 2-layer | 0.9552 | 0.9652 | 0.9304 |
| GIN | 2-layer | 0.9552 | 0.9652 | 0.9304 |
| GraphSAGE | + Attention Pooling | 0.9765 | 0.9435 | 0.9217 |
| GraphSAGE | 2-layer | 0.9680 | 0.9478 | 0.9174 |
| GIN | 3-layer baseline | 0.9659 | 0.9609 | 0.9152 |
| GraphSAGE | 3-layer baseline | 0.9531 | 0.9391 | 0.9130 |

### Key Observations

1. **GAT Baseline** achieves the best test accuracy (95.0%) with high sensitivity (99.6%)
2. **Jumping Knowledge** connections consistently improve performance across architectures
3. **Virtual nodes** decrease performance — local graph structure is sufficient
4. **Shallower networks** often generalize better than deeper counterparts

### Training Curves

![Model Accuracies](model_accuracies. png)
![Test Accuracies](model_test_accuracies. png)

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/Obsi19/graphsage-bloodline.git
cd graphsage-bloodline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements
```
torch>=2.0.0
torch-geometric>=2.3.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
networkx>=2.6.0
```

## 🚀 Usage

```python
# Example:  Load and preprocess data
from data_loader import load_blood_smear_graphs
from models import GAT

# Load graphs from annotated images
train_graphs, val_graphs, test_graphs = load_blood_smear_graphs(
    data_dir='data/H_100X_C2/',
    k_neighbors=5
)

# Initialize model
model = GAT(
    in_channels=num_features,
    hidden_channels=64,
    out_channels=2,
    num_layers=3,
    heads=4
)

# Train model
trainer = Trainer(model, train_graphs, val_graphs)
trainer.train(epochs=100, lr=0.001)

# Evaluate
test_accuracy = trainer.evaluate(test_graphs)
print(f"Test Accuracy: {test_accuracy:.4f}")
```

## 📁 Project Structure

```
graphsage-bloodline/
├── README.md                    # This file
├── main.tex                     # LaTeX source for the paper
├── ref. bib                      # Bibliography references
├── SNA_FINAL_PROJECT (1).pdf    # Full research paper
├── model_accuracies. png         # Training accuracy comparison
├── model_test_accuracies.png    # Test accuracy comparison
├── sna_fig2.png                 # Framework overview figure
├── GAT*. png                     # GAT model training curves
├── GIN*.png                     # GIN model training curves
├── GS*.png                      # GraphSAGE model training curves
├── data/                        # Dataset directory (not included)
│   └── H_100X_C2/               # High-quality 100x microscope images
├── models/                      # Model implementations
│   ├── gin.py
│   ├── gat.py
│   └── graphsage.py
├── utils/                       # Utility functions
│   ├── graph_construction.py
│   └── preprocessing.py
└── notebooks/                   # Jupyter notebooks for experiments
```

## 🔮 Future Work

- [ ] **Healthy Cell Data Collection**: Gathering data on healthy blood cells for improved baseline comparison
- [ ] **Multi-Center Data**: Integration of data from multiple centers for improved generalization
- [ ] **Multi-Modal Imaging**: Incorporating fluorescent or 3D imaging modalities
- [ ] **Interpretability Enhancements**: Advanced explanation methods for clinical trust
- [ ] **Real-World Deployment**: Integration into semi-automated diagnostic systems
- [ ] **Ensemble Methods**: Combining multiple architectures for improved robustness

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{ahmad2024aml,
  title={AML vs ALL Classification Using Machine Learning and Graph Neural Networks},
  author={Ahmad, Abrar and Ali, Mehar and Aziz, Wahaj and Farooq, Hamza},
  booktitle={Social Network Analysis Project},
  year={2024},
  institution={Information Technology University}
}
```

## 🙏 Acknowledgments

- **Intelligent Machines Lab (ITU)** for providing the Blood Cancer Dataset
- **MICCAI 2024** for the dataset association
- Project supervisors for their guidance and support

## 📧 Contact

For questions or collaborations, please open an issue or contact the authors through the Information Technology University, Department of Computer Science. 

---

**Note**: This project is part of ongoing research.  Results and implementations may be updated as the study progresses. 
