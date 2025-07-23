# Self-supervised Audio Deepfake Detection with WavLM and AASIST

[![Project Status: Active Development](https://img.shields.io/badge/status-active_development-yellow)](https://github.com/yourusername/audio-deepfake-detection)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Status
🚧 **Under Active Development** 🚧  
This project is currently in development (February - October 2025 timeline). Implementation and evaluation phases are ongoing.

## 🔍 Overview
A robust audio deepfake detection system combining:
- **WavLM**: Self-supervised speech representation model
- **AASIST**: Spectro-temporal graph attention network
- **FCNN Classifier**: Fully connected neural network

## 🎯 Key Features
| Feature | Benefit |
|---------|---------|
| Self-supervised learning | Reduces labeled data dependency |
| Hybrid architecture | Combines WavLM + AASIST strengths |
| Multi-dataset evaluation | ASVspoof, In-The-Wild, ADD datasets |
| Comprehensive metrics | EER, AUC-ROC, F1-score, latency |

<!-- 
## 📂 Repository Structure
audio-deepfake-detection/
├── data/ # Dataset preprocessing
├── models/ # WavLM, AASIST, FCNN
├── training/ # Training scripts
├── evaluation/ # Metrics and testing
├── utils/ # Helper functions
├── docs/ # Documentation
├── LICENSE
└── README.md
-->
## 🛠️ Installation
```bash
git clone https://github.com/yourusername/audio-deepfake-detection.git
cd audio-deepfake-detection
pip install -r requirements.txt
```

## 📅 Project Timeline

```mermaid
gantt
    title Audio Deepfake Detection Project Timeline
    dateFormat  YYYY-MM-DD
    axisFormat %b %d
    
    section Research
    Problem Definition       :done, des1, 2025-02-13, 2025-02-27
    Literature Review        :done, des2, 2025-03-11, 2025-03-20
    Final Proposal           :done, des3, 2025-04-18, 2025-04-28
    
    section Development
    Dataset Collection       :active, des4, 2025-04-21, 2025-05-11
    Feature Extraction       :         des5, 2025-05-12, 2025-06-01
    FCNN Implementation      :         des6, 2025-06-02, 2025-06-15
    
    section Evaluation
    Model Training           :         des7, 2025-07-14, 2025-07-27
    Cross-Validation         :         des8, 2025-08-11, 2025-08-24
    
    section Reporting
    Research Paper           :         des9, 2025-09-22, 2025-10-12
    Final Presentation       :         des10, 2025-10-24, 2025-10-24
```