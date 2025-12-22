<div align="center">
  <h3 align="center">Supervised Learning for Parameters of Quantum Dots System</h3>

  <p align="center">
   Exploring supervised learning for learning quantum dots system parameters from charge stability diagrams (CSD)
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#verification">Verification</a></li>
      </ul>
    </li>
    <li><a href="#project-structure">Project Structure</a></li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#generating-datasets">Generating Datasets</a></li>
         <ul>
           <li><a href="#configuration-file">Configuration File</a></li>
         </ul>
        <li><a href="#training-models">Training Models</a></li>
      </ul>
    </li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

This project explores supervised learning approaches for learning quantum dots system parameters from charge stability diagrams (CSD). The project uses deep learning models to predict system parameters from simulated charge stability diagrams.

Key features:
- Dataset generation for quantum dot systems
- CNN-based models for parameter prediction
- Support for multiple quantum dot configurations
- Integration with QDarts simulation framework

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

- **Python**: 3.8 or higher (3.9+ recommended)
- **CUDA**: Optional, but recommended for GPU acceleration (CUDA 11.8+ for PyTorch)
- **Git**: For cloning the repository

### Installation

#### Quick Setup (Automated)

**On macOS/Linux:**
```bash
git clone <repository-url>
cd learning_parameters
./setup.sh
```

**On Windows:**
```cmd
git clone <repository-url>
cd learning_parameters
setup.bat
```

The setup scripts will automatically:
- Create a virtual environment
- Install PyTorch (CPU version by default)
- Install all dependencies
- Verify the installation

#### Manual Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd learning_parameters
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On macOS/Linux:
   source venv/bin/activate
   
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install PyTorch** (choose based on your system)
   
   For CUDA 11.8 (recommended for GPU):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
   
   For CPU only:
   ```bash
   pip install torch torchvision torchaudio
   ```

4. **Install remaining dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Note: If you encounter issues with PyTorch installation, install it separately first (step 3), then install the remaining requirements.

5. **Verify installation**
   ```bash
   python -c "import torch; import qdarts; import h5py; print('Installation successful!')"
   ```

### Verification

To verify your setup is working correctly:

1. Check that all imports work:
   ```python
   import torch
   import numpy as np
   import matplotlib.pyplot as plt
   import h5py
   from qdarts.experiment import Experiment
   ```

2. Run a simple test:
   ```bash
   python -c "from qdarts.experiment import Experiment; print('QDarts imported successfully')"
   ```

## Project Structure

```
learning_parameters/
├── qdarts/              # QDarts simulation framework
│   ├── experiment.py
│   ├── simulator.py
│   ├── capacitance_model.py
│   └── ...
├── src/                 # Source code
│   ├── models/         # CNN model definitions
│   │   ├── vanilla_CNN.py
│   │   ├── transfer_CNN.py
│   │   └── multihead_CNN.py
│   ├── utilities/      # Utility functions
│   ├── train_evaluate.py
│   └── plot_results.py
├── configs/            # Configuration files
│   ├── data_generation/
│   └── cnn_model/
├── datasets/           # Generated datasets (gitignored)
├── main.py             # Main entry point
├── dataset_generation.py
├── example.ipynb       # Example notebook
└── requirements.txt    # Python dependencies
```

## Usage

### Generating Datasets

Generate a dataset using a configuration file:

```bash
python dataset_generation.py -c configs/data_generation/test.json
```
#### Configuration file
Description coming soon

### Training Models
Description coming soon

<!-- ROADMAP -->
## Roadmap
- [x] Dataset generation pipeline
- [ ] Preprocessing pipeline
- [ ] Generate larger dataset (26K)
- [ ] Adjust the MB-CNN architecture for new datasets
- [ ] Training pipeline
- [ ] Model evaluation and metrics
- [ ] Come up with experiments to run

<!-- CONTACT -->
## Contact

For questions or issues, please open an issue on the repository.
