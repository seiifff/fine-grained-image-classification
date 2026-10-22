# Fine-Grained Image Classification with Attention (SE-ResNet50 + Grad-CAM)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 🚀 Overview
A PyTorch implementation of **fine-grained visual classification** using **SE-ResNet50** (Squeeze-and-Excitation attention) with strong augmentations, transfer learning, and **Grad-CAM interpretability**.

Designed for datasets with subtle inter-class differences:
- 🍽️ **Food-101**
- 🚗 **Stanford Cars**
- 🐦 **CUB-200-2011** (manual setup)

## 📦 Features
- ✅ **Strong Albumentations pipeline** (RandomResizedCrop, ColorJitter, etc.)
- ✅ **Two-stage training**: backbone freeze → full fine-tuning
- ✅ **Label smoothing**, cosine LR scheduling, mixed-precision
- ✅ **Grad-CAM visualizations** to inspect model attention
- ✅ **Per-class metrics** and confusion matrix

