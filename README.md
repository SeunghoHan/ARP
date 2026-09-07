# Attribute-Guided Relevance Propagation (ARP)

Official implementation of our research on **Attribute-Guided Relevance Propagation (ARP)**, a post-hoc explainable AI method for interpreting deep image classifiers.

## Overview

ARP aims to improve the interpretability of deep neural networks by connecting model predictions with human-understandable semantic attributes.

Unlike conventional attribution methods that mainly visualize pixel-level importance, ARP analyzes internal feature representations and associates them with semantic attributes to provide more structured explanations of model decisions.

The method was developed for CNN-based image classifiers and later extended to Transformer-based architectures.

## Key Idea

The overall procedure consists of:

1. Extracting internal feature representations from a pretrained classifier
2. Learning semantic attribute representations from training samples
3. Measuring the relationship between model features and learned attributes
4. Propagating attribute-guided relevance toward the input
5. Retrieving representative training examples associated with important attributes

This enables the explanation to provide both:

- spatial relevance indicating **where** the model focuses
- semantic evidence indicating **what kind of feature** contributes to the prediction

## Repository Contents

This repository includes code for:

- Attribute representation learning
- Attribute-guided relevance propagation
- Relevance map generation
- Attribute-based example retrieval
- Quantitative evaluation of explanations

## Environment

Main dependencies:

- Python
- PyTorch
- NumPy
- OpenCV
- scikit-learn

Please refer to the repository configuration files for detailed package versions.

## Usage

The repository provides implementations for generating ARP explanations from pretrained image classifiers.

Detailed execution commands and model/data paths may depend on the experimental setup used in the paper.

## Publication

**Attribute-Guided Relevance Propagation for Interpreting Image Classifier Based on Deep Neural Networks**
*Computer Vision and Image Understanding*, 2025.

## Citation

If you find this work useful, please cite the corresponding paper.
