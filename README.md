# Latent Kinematic Engine (LKE)

The Latent Kinematic Engine (LKE) is a robust, query-addressed associative memory architecture designed for precise temporal grounding and moment retrieval in untrimmed video content.

This repository contains the training, evaluation, and deployment code for the LKE architecture. By transitioning from simple action-centric classification to an episodic retrieval framework, this codebase demonstrates how query-conditioned temporal attention significantly improves moment localisation over standard temporal transformers.

## Features

- **Kinematic Grounding:** Advanced 1D and spatial-temporal LKE architectures for precise event localisation.
- **Associative Memory:** Episodic memory retrieval module featuring cross-event contextualisation and orthogonal latent disentanglement.
- **Zero-Shot Transfer:** Built-in pipelines for zero-shot evaluation on datasets like ActivityNet Captions and MSR-VTT.
- **Interactive UI:** A Flask-based web application (`app.py`) for real-time inference and visualisation of temporal grounding predictions.
- **Centralised Configuration:** A clean, YAML-based configuration system for managing datasets, checkpoints, and extracted features without hardcoded paths.

## Repository Structure

```
.
├── config.yaml          # Centralized configuration for data, features, and model paths
├── src/                 # Core architecture modules
│   ├── model_lke.py     # Base Latent Kinematic Engine model
│   ├── model_lke_assoc.py # LKE with Associative Gating and Episodic Memory
│   ├── model_sta.py     # Spatial-Temporal Attention models
│   └── data_loader_*.py # Dataset loaders (Charades-STA, ActivityNet, Spatial)
├── scripts/             # Training and evaluation scripts
│   ├── training/        # Scripts for training on Charades and ActivityNet
│   └── evaluation/      # Scripts for ablation studies and zero-shot testing
├── UI/                  # Web interface for interactive grounding
│   ├── app.py           # Flask server
│   ├── templates/       # HTML templates
│   └── static/          # CSS/JS assets
├── data/                # Placeholder for dataset annotations (.csv, .txt, .json)
├── features/            # Placeholder for extracted video/text features (CLIP, VideoMAE)
└── checkpoints/         # Directory for saved PyTorch model weights (.pt)
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Rijulsin/LatentKinematicEngine.git
cd LatentKinematicEngine
```

2. Install the required dependencies:
```bash
pip install torch torchvision transformers flask pyyaml
```

3. Download the necessary spaCy language model for kinematic disentanglement:
```bash
python -m spacy download en_core_web_sm
```

## Configuration

All paths to datasets, features, and model checkpoints are managed through `config.yaml`. Before running training or evaluation scripts, ensure you have placed your data in the corresponding directories or updated `config.yaml` to point to your custom paths.

Example `config.yaml` configuration:
```yaml
paths:
  train_txt: "./data/charades_sta/charades_sta_train.txt"
  test_txt: "./data/charades_sta/charades_sta_test.txt"
  feat_dir_32f: "./features/clip_features_32f_db"
  model_lke_assoc: "./checkpoints/best_charades_sta_lke_assoc_model.pt"
  save_dir: "./checkpoints"
```

## Usage

### Training

To train the associative LKE model on the Charades-STA dataset:
```bash
python scripts/training/train_charades_lke_assoc.py
```

### Evaluation

To evaluate the trained model or run zero-shot benchmarks (e.g., MSR-VTT):
```bash
python scripts/evaluation/test_charades_lke_assoc.py
python scripts/evaluation/test_msrvtt_lke_assoc.py
```

### Interactive Web UI

You can interactively test the model's temporal grounding capabilities using the included web interface. Ensure your trained `.pt` model is placed in the configured checkpoint directory, then run:

```bash
python UI/app.py
```
The server will start at `http://localhost:5000`.

## Datasets and Feature Extraction

This repository expects pre-extracted features (e.g., OpenAI CLIP ViT-B/32) saved as PyTorch tensors (`.pt`). Supported benchmark datasets include:
- Charades-STA
- ActivityNet Captions
- MSR-VTT (Zero-shot evaluation)

Please place your raw annotation files in the `data/` directory and your extracted feature tensors in the `features/` directory according to the paths defined in your `config.yaml`.

### Download Pre-Processed Data and Weights
You can download the pre-extracted 1D Charades-STA features (32-frame), dataset annotations (CSV/TXT), and pre-trained LKE weights directly from this **[Google Drive Folder](https://drive.google.com/drive/folders/1lIg4uge4qLwtqk_KQTKftN_5S0Kj2omE?usp=sharing)**. 

Unzip and place these files into their respective `data/`, `features/`, and `checkpoints/` directories before running the evaluation scripts or the web UI.
