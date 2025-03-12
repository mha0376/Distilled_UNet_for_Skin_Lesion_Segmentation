![banner](https://github.com/user-attachments/assets/077119ea-7d46-4cd1-adb9-0bb35cb27bbc)

# Distilled UNet for Skin Lesion Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1y0njH__4VfU38eHda263URARnl1plMnR)

## Overview

This project demonstrates knowledge distillation for skin lesion segmentation, specifically designed to run seamlessly on Google Colaboratory (Colab).  It utilizes a powerful teacher model (DeepLabV3+) to guide the training of a lightweight student model (UNet), aiming to achieve high segmentation accuracy with a smaller, more efficient network. This approach is ideal for deploying accurate medical image analysis models, even in resource-constrained environments.

Leveraging the ISIC 2016 skin lesion dataset, this project provides a ready-to-use Colab notebook for training, evaluation, and visualization.

**This project is designed to be run on Google Colaboratory. You can open and run the provided `.ipynb` notebook directly on Colab.**

## Table of Contents

- [Overview](#overview)
- [Table of Contents](#table-of-contents)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation (Colab)](#installation-colab)
- [Usage](#usage)
  - [Running the Colab Notebook](#running-the-colab-notebook)
  - [Training the Student Model](#training-the-student-model)
  - [Evaluating the Model](#evaluating-the-model)
  - [Visualizing Predictions](#visualizing-predictions)
  - [Monitoring with TensorBoard](#monitoring-with-tensorboard)
- [Model Architecture](#model-architecture)
  - [Teacher Model](#teacher-model)
  - [Student Model](#student-model)
- [Dataset](#dataset)
- [Training Details](#training-details)
  - [Loss Functions](#loss-functions)
  - [Optimizer and Scheduler](#optimizer-and-scheduler)
  - [Distillation Strategy](#distillation-strategy)
- [Evaluation Metrics](#evaluation-metrics)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

- **Google Colab Ready:** Project designed and tested to run directly on Google Colab, leveraging free GPU resources.
- **Knowledge Distillation:** Employs knowledge distillation to transfer knowledge from a DeepLabV3+ teacher to a UNet student model.
- **Skin Lesion Segmentation:** Focused on accurate segmentation of skin lesions in dermoscopic images.
- **ISIC 2016 Dataset:** Utilizes the widely recognized ISIC 2016 dataset for training and validation. Dataset download and preparation are automated within the Colab notebook.
- **PyTorch Implementation:** Built entirely using the PyTorch deep learning framework.
- **UNet Student Model:**  Employs the efficient UNet architecture for the student model.
- **DeepLabV3+ Teacher Model:**  Leverages a pre-trained DeepLabV3+ ResNet101 model as the teacher for knowledge transfer.
- **Dice Loss and Distillation Loss:** Combines Dice loss for segmentation accuracy and KL Divergence for effective knowledge distillation.
- **TensorBoard Integration:** Training progress, loss, and Dice score are automatically tracked and visualized using TensorBoard directly within Colab.
- **Inline Prediction Visualization:**  Includes functions to visualize segmentation predictions directly within the Colab notebook for quick qualitative assessment.
- **Best Model Saving:** Automatically saves the best performing student model based on validation Dice score.

## Getting Started

### Prerequisites

- **Google Account:** Required to access Google Colaboratory.
- **Web Browser:**  To open and run the Colab notebook.

### Installation (Colab)

**No local installation is required to run this project!** Everything is designed to work directly within Google Colaboratory.

1.  **Open in Colab:** Click on the "Open in Colab" badge at the top of this README, or upload the provided `.ipynb` notebook to your Google Drive and open it in Colab.
    [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1y0njH__4VfU38eHda263URARnl1plMnR)

2.  **Runtime Type:** In Colab, go to `Runtime` > `Change runtime type` and select `GPU` as the Hardware accelerator for faster training.

## Usage

### Running the Colab Notebook

Simply open the provided `.ipynb` notebook in Google Colab and execute the cells sequentially. The notebook is self-contained and includes all steps for:

- **Dataset Download and Preparation:**  Automatically downloads and organizes the ISIC 2016 dataset directly within the Colab environment.
- **Model Definition and Initialization:** Defines and initializes both teacher and student models.
- **Training Process:** Executes the knowledge distillation training loop.
- **Validation and Evaluation:**  Evaluates the student model during training.
- **Visualization:** Displays segmentation predictions and TensorBoard dashboards within the notebook.
- **Model Saving:** Saves the best student model to Colab's storage.

### Training the Student Model

To start training, simply run all cells in the Colab notebook. The training parameters (epochs, learning rate, etc.) are pre-configured in the notebook but can be easily adjusted. The notebook outputs training progress, validation metrics, and saves the best student model automatically.

### Evaluating the Model

Validation is performed during training, and the validation Dice score and loss are displayed after each epoch.  The best student model, based on the highest validation Dice score, is saved as `best_student_model.pth` in the Colab environment.

### Visualizing Predictions

After training, the notebook automatically visualizes segmentation predictions on a few validation examples. Inline plots are generated, showing the input image, ground truth mask, and the student model's predicted segmentation mask. This provides a quick visual assessment of the model's performance.

![image](https://github.com/user-attachments/assets/5e232dfd-733b-47f5-a880-e8ed7a422269)
![image](https://github.com/user-attachments/assets/b80f563b-9bff-44de-8cbd-8faf8e4ccf09)

### Monitoring with TensorBoard

Training progress is monitored using TensorBoard directly within the Colab notebook. After running the training cells, TensorBoard allowing you to visualize:

- **Training and Validation Loss curves**
- **Validation Dice Score**
- **Other relevant metrics**

This enables you to monitor the learning process, identify potential issues, and assess the effectiveness of knowledge distillation.

## Model Architecture

### Teacher Model

- **Architecture:** DeepLabV3+ with a ResNet101 backbone. Leverages pre-trained weights from `torchvision.models.segmentation.deeplabv3_resnet101`.
- **Purpose:** Serves as the knowledgeable "teacher" to guide the training of the student model. Provides soft probability targets for distillation.

### Student Model

- **Architecture:** UNet. A custom UNet implementation is provided in the notebook, designed for efficient segmentation.
- **Purpose:**  The "student" model, trained to mimic the teacher's predictions and learn from ground truth, aiming for a smaller and faster network with good performance.

## Dataset

- **Dataset:** ISIC 2016 Skin Lesion Segmentation Challenge Dataset (Part 3B).
- **Source:**  Publicly available dataset for skin lesion segmentation from the ISIC challenge.
- **Preparation:** The Colab notebook automatically downloads and prepares the dataset. It creates directories (`train_image_data`, `train_mask_data`, etc.) and splits the training data into training and validation sets.

## Training Details

### Loss Functions

- **Dice Loss:** Primary segmentation loss function, calculated using `dice_loss()` in the notebook.
- **Distillation Loss:**  Combines Dice Loss and Kullback-Leibler Divergence (KL Divergence) using `distillation_loss()`.
    - **KL Divergence:** Measures the difference between the teacher's and student's prediction distributions, guiding the student to learn the "style" of the teacher's predictions.

### Optimizer and Scheduler

- **Optimizer:** Adam (`torch.optim.Adam`) is used to optimize the student model.
- **Scheduler:** ReduceLROnPlateau (`torch.optim.lr_scheduler.ReduceLROnPlateau`) adjusts the learning rate dynamically during training based on validation loss plateaus.

### Distillation Strategy

- **Online Knowledge Distillation:**  The teacher model processes inputs in each training iteration to provide real-time guidance to the student model. The student learns from both ground truth and teacher predictions simultaneously.

## Evaluation Metrics

- **Validation Loss:** Average distillation loss on the validation set.
- **Validation Dice Score:** Average Dice score on the validation set. The primary metric for evaluating segmentation performance.

The best student model is saved based on the highest Validation Dice Score.

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please:

1.  **Fork the repository.**
2.  **Create a new branch** for your feature or bug fix.
3.  **Implement your changes** and ensure they are well-documented and tested within the Colab environment.
4.  **Submit a pull request** with a clear description of your contribution.

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- **ISIC 2016 Dataset:**  We acknowledge the International Skin Imaging Collaboration (ISIC) for providing the ISIC 2016 dataset.
- **PyTorch:** This project is built using the PyTorch deep learning framework.
- **DeepLabV3+ and UNet Architectures:** We acknowledge the researchers who developed the DeepLabV3+ and UNet architectures, which are fundamental to this work.
- **Google Colaboratory:** For providing free and accessible cloud-based GPU resources that make this project easily runnable for anyone.
