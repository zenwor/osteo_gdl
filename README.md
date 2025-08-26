# 🦴 Osteo GDL
📐🕸️ Geometry in Histopathology: Comparative Analysis on Graph Neural Networks and Riemannian Manifold Embeddings in Osteosarcoma Classification

## ℹ️ Overview
This repository contains the code, experiments, and analyses for exploring **geometric deep learning (GDL)** approaches in osteosarcoma classification from histopathology images. We evaluate three novel architectures that leverage different geometric principles:

- **OsteoGNN** – Graph-based representation learning using KNN graphs and GNN layers.  
- **OSPNet** – Manifold-based feature embeddings using symmetric positive definite (SPD) matrices.  
- **OEHNet** – Hyperbolic embeddings for hierarchical feature encoding.  

We study their effectiveness on patch-level embeddings, hyperparameter sensitivity, and imbalanced dataset handling strategies.

## 🧩 Features
- Graph construction and GNN-based modeling for histopathology patches.
- Riemannian and hyperbolic manifold embeddings for complex feature representations.
- Support for both **ResNet** and **Vision Transformer (ViT)** backbones for patch embeddings.
- Weighted loss and weighted sampling for addressing class imbalance.
- Hyperparameter studies for K in KNN graphs and curvature C in hyperbolic embeddings.

## 📊 Results
The architectures were evaluated on a 3-class osteosarcoma classification task. Key findings include:

- **OsteoGNN** performs strongly on small to medium graph sizes but saturates with very large embeddings.  
- **OEHNet** benefits from moderate curvature and embedding size (C = 1.2–1.5, embedding 128–256).  
- **OSPNet** provides stable performance across manifold dimensions, showing robustness to variations in SPD encoding.  

> For detailed results, refer to the tables in the paper.

## ⬇️ Installation
To download and set up the repository, run:
```bash
git clone https://github.com/zenwor/osteo_gdl.git
cd osteo_gdl
source ./setup.sh
```
Then, make sure to the official dataset and insert its location in setup script.

# 📝 Citation
If you use this code or refer to the results in this work, please cite the following:
```bibtex
@unpublished{nedimovic2025osteosarcomagdl,
  title={Geometry in Histopathology: Comparative Analysis on Graph Neural Networks and Riemannian Manifold Embeddings in Osteosarcoma Classification},
  author={Luka Nedimović},
  note={Manuscript in preparation},
  year={2025}
}
```