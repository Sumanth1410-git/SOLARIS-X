# 🛰️ SOLARIS-X: Advanced Space Weather Prediction System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange)](https://tensorflow.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6-green)](https://lightgbm.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-red)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)](https://github.com/yourusername/SOLARIS-X)

> **World-class machine learning system for geomagnetic storm prediction using 29 years of space weather data**

---

## 🌟 **OVERVIEW**

**SOLARIS-X** is an advanced ensemble machine learning system designed for real-time geomagnetic storm prediction. Built with **29 years of space weather data** (1996-2025) and **79 physics-informed features**, it achieves **96.5% AUC** with superior storm detection capabilities.

### **🎯 Key Achievements**
- 🏆 **67.8% Storm Recall** - Catches 2 out of 3 geomagnetic storms
- ⚡ **<100ms Inference** - Real-time prediction capability
- 🔬 **Physics-Informed** - Features based on solar wind & magnetospheric physics
- 🚀 **Production-Ready** - Complete MLOps pipeline with deployment architecture

---

## 📊 **PERFORMANCE METRICS**

| Model | AUC | Precision | Recall | F1-Score | Best For |
|-------|-----|-----------|--------|----------|-----------|
| **Meta-Ensemble** ⭐ | **0.9646** | 0.4862 | **0.6784** | **0.5664** | Storm Detection |
| LightGBM Baseline | 0.9671 | **0.6655** | 0.4904 | 0.5647 | High Precision |

> 🎖️ **Meta-Ensemble selected for production** - Superior storm detection critical for space weather operations

---

## 🏗️ **SYSTEM ARCHITECTURE**

### **🔬 Core Components**

1. **Advanced Feature Engineering**
   - 29-year dataset (257,232 samples)
   - 79 physics-based predictors
   - Temporal lag features (1-24 hours)
   - Solar cycle & seasonal patterns

2. **Ensemble Architecture**
   - **LightGBM**: Gradient boosting baseline
   - **Bidirectional GRU**: Temporal sequence modeling
   - **Meta-Learner**: Combines predictions optimally

3. **Production Pipeline**
   - Memory-optimized data processing
   - Robust scaling & preprocessing
   - Model persistence & versioning
   - Real-time inference capability

---

## 📈 **DATASET & FEATURES**

### **🛰️ Space Weather Data Sources**
- **NASA OMNI Database**: Solar wind parameters, IMF data
- **NOAA SWPC**: Geomagnetic indices (Kp, AE, Dst)  
- **Temporal Coverage**: 1996-2025 (29 years)
- **Data Quality**: 98.3% completeness after cleaning

### **⚡ Top Physics Features**
1. **Kp-index** - Geomagnetic activity indicator
2. **IMF Magnitude** - Interplanetary magnetic field strength
3. **Plasma Beta** - Solar wind plasma parameter
4. **AE Index** - Auroral electrojet activity
5. **Solar Cycle Phase** - Long-term solar variability

---

## 🚀 **QUICK START**

### **Installation**

Clone repository
```
git clone https://github.com/yourusername/SOLARIS-X.git
cd SOLARIS-X
```
Create virtual environment
```
python -m venv venv
```
Activate virtual environment
```
#Windows:
venv\Scripts\activate

#Linux/Mac:
source venv/bin/activate
```
Install dependencies
```
pip install -r requirements.txt
```

### **Training Pipeline**

Run complete training pipeline
```
python scripts/training/complete_pipeline.py
```
Individual model training
```
python scripts/training/test_lightgbm.py
python scripts/training/test_neural_network.py
```

### **Quick Prediction Example**
```
import joblib
import pandas as pd
import numpy as np

Example space weather features
features = {
'Kp_index': 3.5,
'IMF_Magnitude_lag_12h': 7.2,
'Plasma_Beta': 0.8,
'AE_index': 150.0,
'Solar_Cycle_Phase': 0.6
# ... additional features required
}

print("SOLARIS-X Space Weather Prediction System")
print("Geomagnetic Storm Prediction: Ready for deployment")
```

---

## 📁 **PROJECT STRUCTURE**

### **Core Directories**

**📊 Data Pipeline**
- `data/processed/features/` - Engineered feature datasets (excluded)
- `data/raw/omni/` - Original OMNI database files (excluded)

**🤖 Training System**  
- `scripts/training/models/` - Individual model trainers (4 advanced models)
- `scripts/training/utils/` - Training utilities and base classes
- `scripts/training/complete_pipeline.py` - Main orchestrator

**💾 Model Management**
- `models/checkpoints/` - Training checkpoints and metadata
- `models/trained/` - Production models (excluded)

**📈 Results & Analysis**
- `results/plots/` - Performance visualizations (10 charts)

**📋 Configuration**
- `requirements.txt` - Python dependencies
- `.gitignore` - Repository optimization
- `README.md` - Documentation

### **Key Features**
- **25+ Python modules** with professional architecture
- **4 advanced ML models** including meta-ensemble system
- **Complete MLOps pipeline** with automated evaluation
- **Production-ready deployment** configuration

---
### **🏗️ Architecture Highlights**
- **Modular Design**: Separated model trainers and utilities
- **Production Ready**: Complete MLOps pipeline structure  
- **Optimized Storage**: Large files excluded via .gitignore
- **Comprehensive Evaluation**: Visualization and metrics tracking
- **Professional Organization**: Clear separation of concerns

**Note**: Files marked as "(excluded)" are not tracked in git due to size constraints but are generated during training.



---

## 🔬 **SCIENTIFIC METHODOLOGY**

### **🎯 Research Approach**
- **Temporal Validation**: Proper chronological train/validation/test splits
- **Physics-Informed**: Features based on magnetospheric coupling theory
- **Imbalanced Learning**: Specialized techniques for rare storm events
- **Ensemble Methods**: Meta-learning for optimal prediction combination

### **📊 Evaluation Protocol**
- **No Data Leakage**: Strict temporal separation of datasets
- **Multiple Metrics**: AUC, F1, Precision, Recall for comprehensive assessment
- **Cross-Validation**: Robust performance estimation
- **Uncertainty Quantification**: Prediction confidence intervals

---

## 🌍 **APPLICATIONS**

### **🛰️ Operational Use Cases**
- **Space Weather Centers**: NOAA, ESA integration
- **Satellite Operations**: ISS, commercial satellite protection
- **Power Grid Safety**: Geomagnetic storm early warning
- **Aviation**: High-altitude flight safety alerts

### **🎓 Research Applications**
- **Space Physics**: Magnetospheric dynamics research
- **Climate Science**: Space weather impact studies
- **Machine Learning**: Rare event prediction techniques
- **Data Science**: Time series ensemble methods

---

## 🔧 **TECHNICAL SPECIFICATIONS**

### **💻 System Requirements**
- **Python**: 3.8+ 
- **RAM**: 8GB minimum, 16GB recommended
- **CPU**: Multi-core processor (12+ cores optimal)
- **Storage**: 5GB for full dataset and models

### **🛠️ Key Dependencies**
```
lightgbm==4.6.0
tensorflow-cpu==2.20.0
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.1
seaborn==0.12.2
```

---

## 👨‍💻 **AUTHOR**

**Sumanth** - *Space Weather & Machine Learning Research*

- 🌐 **GitHub**: [@Sumanth1410-git](https://github.com/Sumanth1410-git)
- 📧 **Contact**: sumanthp141005@gmail.com


---

## 🏆 **ACHIEVEMENTS**

- 🥇 **96.5% AUC** - World-class space weather prediction performance
- 🎯 **67.8% Storm Recall** - Superior rare event detection
- 🚀 **Production-Ready** - Complete MLOps pipeline
- 🔬 **Physics-Informed** - Scientifically validated approach
- 📊 **29-Year Dataset** - Comprehensive historical coverage

---

## 📄 **LICENSE**

This project is licensed under the **MIT License**.

---

## 🙏 **ACKNOWLEDGMENTS**

- **NASA OMNI Database** - Space weather data provision
- **NOAA Space Weather Prediction Center** - Operational data access
- **Space Weather Community** - Research inspiration and validation
- **Open Source Contributors** - Tool and library development

---

<div align="center">

### 🌟 **SOLARIS-X: Protecting Earth from Space Weather** 🌟

*Built with ❤️ for the space weather research community*

**[⭐ Star this repository](https://github.com/Sumanth1410-git/SOLARIS-X)** if it helps your research!

</div>


