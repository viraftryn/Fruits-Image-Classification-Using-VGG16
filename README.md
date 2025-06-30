# Fruits Image Classification Using VGG16

A comprehensive deep learning project for multiclass fruit image classification using Convolutional Neural Networks (CNN) with VGG16 architecture and custom modifications.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Data Exploration & Preprocessing](#data-exploration--preprocessing)
- [Challenges & Difficulties](#challenges--difficulties)
- [Model Architecture](#model-architecture)
- [Results](#results)

## 🎯 Project Overview

This project implements a fruit image classification system using deep learning techniques. The main objectives include:

- Building a robust CNN architecture based on VGG16 for multiclass fruit classification
- Exploring and preprocessing image data with appropriate augmentation techniques
- Identifying and addressing challenges in image quality and variability
- Implementing both baseline VGG16 and modified architectures for optimal performance
- Comprehensive evaluation and performance analysis

## 📊 Dataset

The dataset consists of fruit images across multiple classes with varying characteristics:
- **Image Classification**: banana, betel_nut, bitter_gourd, black_berry
- **Image Resolution**: Original images resized to 224×224 pixels
- **Data Split**: 
  - Training: 80%
  - Validation: 10%
  - Testing: 10%

### Data Preprocessing Steps
1. **Image Resizing**: All images standardized to 224×224 resolution
2. **Color Histogram Analysis**: Exploration of RGB distribution patterns
3. **Data Augmentation**: Applied to increase dataset diversity and improve model generalization
4. **Normalization**: Pixel values normalized for optimal training performance

## 🔍 Data Exploration & Preprocessing

### Color Histogram Analysis
- Analyzed RGB channel distributions across different fruit categories
- Identified color patterns and variations within and between classes
- Used insights to inform preprocessing and augmentation strategies

### Data Augmentation Techniques
Applied various augmentation methods to enhance dataset diversity:
- Rescale
- Rotation
- Flipping

## 🚧 Challenges & Difficulties

### Image Quality Issues
1. **Resolution Variations**
   - Original images with inconsistent resolutions
   - Solution: Standardized resizing to 224×224 pixels

2. **Noise Artifacts**
   - Some images contain compression artifacts or blur
   - Impact: Reduced feature extraction quality

### Image Variability Challenges
1. **Lighting Conditions**
   - Inconsistent illumination across images
   - Examples: Some fruits photographed under natural light vs. artificial lighting
   - Impact: Color representation variations

2. **Perspective and Angle**
   - Fruits captured from different viewpoints
   - Examples: Top-down, side view, angled perspectives
   - Challenge: Maintaining consistent feature recognition

3. **Occlusions**
   - Partial fruit visibility due to overlapping or shadows
   - Impact: Incomplete feature extraction

4. **Background Variability**
   - Different background contexts (tables, natural settings, white backgrounds)
   - Challenge: Model focus on fruit features rather than background

5. **Scale Variations**
   - Fruits appearing at different sizes within images
   - Impact: Feature scale inconsistency

## 🏗️ Model Architecture

### Baseline VGG16 Architecture

#### Pre-trained Model Details
- **Base Model**: VGG16 architecture from scratch using Keras
- **Input Shape**: (224, 224, 3)
- **Feature Extraction**: Convolutional layers frozen initially
- **Classification Head**: Custom dense layers for fruit classification

#### Model Architecture
```
Model: "baseline_vgg16"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)             (None, 224, 224, 64)       1792     
conv2d_1 (Conv2D)           (None, 224, 224, 64)       26928
max_pooling2d (MaxPooling2  (None, 112, 112, 64)       0
D)
conv2d_2 (Conv2D)           (None, 112, 112, 128)       73856
conv2d_3 (Conv2D)           (None, 112, 112, 128)       147584
max_pooling2d_1 (MaxPoolin  (None, 56, 56, 128)         0
g2D)
conv2d_4 (Conv2D)           (None, 56, 56, 256)         295168
conv2d_5 (Conv2D)           (None, 56, 56, 256)         590080
conv2d_6 (Conv2D)           (None, 56, 56, 256)         590080
max_pooling2d_2 (MaxPoolin  (None, 28, 28, 256)         0
g2D)
conv2d_7 (Conv2D)           (None, 28, 28, 512)         1180160
conv2d_8 (Conv2D)           (None, 28, 28, 512)         2359808
conv2d_9 (Conv2D)           (None, 28, 28, 512)         2359808
max_pooling2d_3 (MaxPoolin  (None, 14, 14, 512)         0
g2D)
conv2d_10 (Conv2D)          (None, 14, 14, 512)         2359808
conv2d_11 (Conv2D)          (None, 14, 14, 512)         2359808
conv2d_12 (Conv2D)          (None, 14, 14, 512)         2359808
max_pooling2d_4 (MaxPoolin  (None, 7, 7, 512)           0
g2D)
Flatten (Flatten)           (None, 25088)               0
dense (Dense)               (None, 4096)                102764544
dense_1 (Dense)             (None, 4096)                16781312
dense_2 (Dense)             (None, 4)                   16388
=================================================================
Total params: 134276931
Trainable params: 134276932
Non-trainable params: 0
```

### Modified VGG16 Architecture

#### Key Modifications
1. **Batch Normalization**
   - Implemented after convolutional layers
   - Improved training stability and convergence speed

3. **Additional Early Stopping**
   - Early stopping with 10 patience monitoring

## 📈 Results

### Model Performance Metrics

#### Baseline VGG16
- **Training Accuracy**: 25.78%
- **Validation Accuracy**: 19.37%
- **Test Accuracy**: 24.38%

#### Modified VGG16
- **Training Accuracy**: 79.37%
- **Validation Accuracy**: 76.25%
- **Test Accuracy**: 82.50%

### Performance Analysis
- **Loss Curves**: Training and validation loss progression
- **Accuracy Curves**: Training and validation accuracy trends

### Samples Ground Truth Modified VGG16
```
Modified Architecture:
5/5 [==============================] - 1s 121ms/step
Sample 1: Ground Truth = 3, Predicted = 0
Sample 2: Ground Truth = 3, Predicted = 0
Sample 3: Ground Truth = 2, Predicted = 0
Sample 4: Ground Truth = 2, Predicted = 0
Sample 5: Ground Truth = 1, Predicted = 0
Sample 6: Ground Truth = 3, Predicted = 3
Sample 7: Ground Truth = 1, Predicted = 0
Sample 8: Ground Truth = 0, Predicted = 0
Sample 9: Ground Truth = 0, Predicted = 0
Sample 10: Ground Truth = 1, Predicted = 3
```

#### Analysis Model Performance
Judging from several prediction samples above, there is a significant difference between the ground truth value and the prediction results. Even with quite good accuracy performance, this modified VGG16 architecture model still has an overfitting problem in making predictions.


