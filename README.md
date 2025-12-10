# Deep Learning on 3D Geometrical Objects

![Python](https://img.shields.io/badge/python-3.7%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

This repository provides a comprehensive toolkit for deep learning on 3D point cloud data, featuring state-of-the-art architectures for classification and segmentation tasks. The project includes both deep learning models and geometry processing utilities for end-to-end 3D data processing.

## 🌟 Key Features

- **Multiple Architectures**: Implements PointNet, PointNet++, GAPNet, and HPE models
- **Dual Task Support**: Both classification and segmentation tasks
- **3D Geometry Processing**: Tools for CAD to point cloud conversion and processing
- **End-to-End Pipeline**: From raw 3D data to trained models
- **High Performance**: Optimized C++ core with Python bindings

## 🗂 Project Structure

```
Deep-learning-on-geometrical-objects/
├── Geometry/           # C++ utilities for 3D geometry processing
├── Models/             # Deep learning model implementations
│   ├── pointnet.py     # PointNet implementation
│   ├── pointnet_PP.py  # PointNet++ implementation
│   ├── gapnet_*.py     # GAPNet variants
│   ├── hpe.py          # HPE implementation
│   └── train.ipynb     # Training notebook
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- C++17 compatible compiler (for Geometry utilities)
- CUDA (recommended for GPU acceleration)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Deep-learning-on-geometrical-objects.git
cd Deep-learning-on-geometrical-objects
```

2. Set up the environment:

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# (Optional) Build Geometry utilities
cd Geometry
mkdir -p build && cd build
cmake ..
make
```

## 🧠 Models

### Available Models

| Model      | Task                        | Description                                |
| ---------- | --------------------------- | ------------------------------------------ |
| PointNet   | Classification/Segmentation | Pioneering work for point cloud processing |
| PointNet++ | Classification/Segmentation | Hierarchical feature learning              |
| GAPNet     | Classification/Segmentation | Graph attention networks for point clouds  |
| HPE        | Classification              | Hierarchical point-based edge convolution  |

## 🔧 Geometry Processing

The Geometry module provides high-performance C++ utilities for 3D geometry processing and point cloud generation from various CAD and mesh file formats.

### ✨ Key Features

- **Multiple Format Support**: Convert between STEP, OBJ, PLY, INP, and KEY formats
- **High Performance**: Optimized C++ implementation for large 3D models
- **Python Integration**: Seamless integration with Python workflows
- **Point Cloud Generation**: Generate point clouds from CAD and mesh files
- **Geometry Analysis**: Utilities for point cloud manipulation and processing

### 📁 Supported File Formats

| Format | Extension   | Description                       |
| ------ | ----------- | --------------------------------- |
| STEP   | .step, .stp | Standard 3D model exchange format |
| OBJ    | .obj        | Wavefront 3D model format         |
| PLY    | .ply        | Polygon File Format               |
| INP    | .inp        | Abaqus input format               |
| KEY    | .key        | KeyCreator CAD format             |

### 🛠 Build Instructions

```bash
# Navigate to Geometry directory
cd Geometry

# Create build directory
mkdir -p build && cd build

# Configure and build
cmake ..
make
```

### 📚 Core Components

- **Point Class**: 3D point representation with geometric operations
- **File Readers**: Convert between various 3D file formats
- **Geometry Utilities**: Point cloud processing and analysis tools

## 🛠 Usage

### 1. Data Preparation

Convert CAD files to point clouds:

```c++
./convert_to_pointcloud "input_dir/", "output_dir/"
```

### 2. Training a Model

```python
from Models.pointnet import PointNet
from utils.trainer import Trainer

# Initialize model
model = PointNet(num_classes=10).cuda()

# Set up trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    lr=0.001,
    epochs=100
)

# Start training
trainer.train()
```

### 3. Inference

```python
import torch
from Models.pointnet import PointNet

# Load pretrained model
model = PointNet(num_classes=10).cuda()
model.load_state_dict(torch.load('checkpoints/pointnet_best.pth'))
model.eval()

# Make prediction
with torch.no_grad():
    outputs = model(point_cloud)
    _, predicted = torch.max(outputs.data, 1)
```

## 📊 Visualization

Visualize point clouds and model predictions:

```python
import open3d as o3d
from utils.visualization import visualize_pointcloud

# Load point cloud
pcd = o3d.io.read_point_cloud("example.ply")

# Visualize
visualize_pointcloud(
    pcd,
    window_name="3D Point Cloud",
    point_size=2.0,
    background_color=[0, 0, 0]
)
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

- [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593)
- [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413)
- [GAPNet: Graph Attention based Point Neural Network](https://arxiv.org/abs/1905.08705)
- [HPE: Hierarchical Point-based Edge Convolution](https://arxiv.org/abs/1906.01140)
