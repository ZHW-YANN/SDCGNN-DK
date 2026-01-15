# Structure Divide-and-Conquer: Dual Graph Representation for Accurate Ionic Transport Barrier Prediction of Inorganic Compounds

## Introduction

SDCGNN-DK (Spatial Dual Crystal Graph Neural Network with Domain Knowledge) is a novel deep learning framework designed for predicting materials properties by leveraging both crystal structure and interstice network representations. The model combines geometric graph neural networks with domain knowledge from materials science to provide accurate predictions of material characteristics.

## Features

- **Dual Graph Representation**: Combines crystal structure graphs and interstice network graphs for comprehensive material modeling
- **Multiple GNN Architectures**: Supports various graph neural network implementations, including CGCNN, MEGNet, GATGNN, and GeoCGNN
- **Domain Knowledge Integration**: Incorporates physics-based features from crystallographic analysis tools
- **Flexible Architecture**: Modular design allowing easy customization and extension
- **Comprehensive Data Pipeline**: Complete workflow from CIF file processing to model prediction

## Architecture

### Core Components

- **[GeoCGNN_data_utils.py]**: Handles geometric feature extraction, including radial basis functions and spherical Bessel functions
- **[CIFData]**: Processes crystal structure files with neighbor analysis
- **[SDCGNN]**: Main model combining crystal structure and interstice network convolutions
- **[Merge_Model]**: Implements various fusion strategies for combining multi-graph representations

### Model Variants

- **[CS_Conv]**: Crystal structure convolution layers
- **[IN_Conv]**: Interstice network convolution layers
- **[GATGNN]**: Graph attention mechanism for materials
- **[GeoCGNN]**: Geometric crystal graph neural network

## Knowledge Calculation Component

The project includes a knowledge calculation module that integrates domain-specific analysis:

- **CCNB Integration**: Crystal connectivity and bond valence analysis
- **CAVD Calculations**: Crystal structure analysis via Voronoi decomposition
- **Energy Calculations**: Bond valence site energy computations
- **Transport Network Analysis**: Ion migration pathway identification

## Directory Structure

```
SDCGNN-DK/
├── data_loader/           # Data processing modules
│   ├── GeoCGNN_data_utils.py
│   ├── data_utils.py
│   ├── load_data.py
│   └── process_cif.py
├── model/                 # Model architectures
│   ├── CGCNN_Conv.py
│   ├── CS_Conv.py
│   ├── GATGNN_Conv.py
│   ├── GeoCGNN_Conv.py
│   ├── IN_CGCNN_Conv.py
│   ├── IN_Conv.py
│   ├── MEGNET_Conv.py
│   ├── Merge_Model.py
│   └── SDCGNN.py
├── Knowledge_calculation/ # Domain knowledge integration
│   ├── CCNB-release/
│   └── Cal_Energy/
├── Comparative Models/    # Baseline model implementations
│   ├── BNMCDGNN-main/
│   ├── CGCNN-main/
│   ├── GATGAN-main/
│   ├── MEGNet-main/
│   └── geoCGNN-main/
└── data/                  # Data storage location
```


## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/SDCGNN-DK.git
cd SDCGNN-DK

# Install dependencies
pip install torch torch-geometric numpy pandas pymatgen scipy sklearn ase cavd
```


## Usage

### Data Preparation
Prepare your CIF files and organize them in the data directory:

```
data/
├── cif/              # Crystal structure files
├── txt/              # Interstice network files
└── atom_feature.csv  # Atomic feature definitions
```


### Training Example
```python
from data_loader.load_data import load_data
from model.SDCGNN import SDCGNN
import torch

# Load the dataset
dataset = load_data('./data')

# Initialize the model
model = SDCGNN(args)

# Train the model with your preferred training loop
```

## Comparative Models
The repository includes several baseline models for comparison:

- **CGCNN**: [Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301)
- **MEGNet**: [Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals](https://pubs.acs.org/doi/10.1021/acs.chemmater.9b01294)
- **GATGNN**: [Graph convolutional neural networks with global attention for improved materials property prediction](https://pubs.rsc.org/en/content/articlelanding/2020/cp/d0cp01474e)
- **GeoCGNN**: [A geometric-information-enhanced crystal graph network for predicting properties of materials](https://www.nature.com/articles/s43246-021-00194-3)
- **BNMCDGNN**: [BNM-CDGNN: Batch Normalization Multilayer Perceptron Crystal Distance Graph Neural Network for Excellent-Performance Crystal Property Prediction](https://pubs.acs.org/doi/10.1021/acs.jcim.3c01148)

## Data Pipeline
The framework supports multiple data processing workflows:

1. **Crystal Structure Processing**: Converts CIF files to graph representations
2. **Interstice Network Generation**: Creates void/channel networks from crystal structures
3. **Feature Engineering**: Computes geometric and chemical features
4. **Model Training**: Trains the dual graph neural network

## Results
SDCGNN-DK demonstrates superior performance compared to individual graph representations by leveraging both crystal structure and interstice network information simultaneously.

## Contributing
We welcome contributions to SDCGNN-DK! Please fork the repository and submit pull requests for bug fixes, new features, or improvements.
