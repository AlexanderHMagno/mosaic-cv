---
title: 'Mosaic Analysis'
emoji: '🚀' # Use an emoji that represents your app
colorFrom: 'blue' # Starting gradient color
colorTo: 'purple' # Ending gradient color
sdk: 'gradio' # Replace with the SDK you're using (e.g., 'gradio', 'streamlit', etc.)
sdk_version: '4.44.1' # Replace with the appropriate version
app_file: 'src/app.py' # Replace with the main file for your app
pinned: false # Set to true if you want to pin the Space
---

# Mosaic Image Processing Application

## Introduction

This project is a Python-based application that generates mosaic images from an input image. The process involves four primary components:

1. **Image Preprocessing**: Standardizing the input image for consistent processing.
2. **Grid Division and Analysis**: Segmenting the image into grids and analyzing each grid cell.
3. **Tile Mapping with Blending**: Overlaying tiles onto the image and blending them appropriately.
4. **Performance Evaluation**: Measuring the efficiency and quality of the mosaic.

## Methodology

### 1. Image Preprocessing

- The input image is converted to **RGB format** for consistency.
- Dimensions are resized to be multiples of the grid size, ensuring seamless grid division.
- Area interpolation is used to maintain visual quality during resizing.

### 2. Grid Division and Analysis

- The preprocessed image is divided into grid cells of the specified size.
- Each cell's average color is calculated and used to annotate the image for visualization.

### 3. Tile Mapping with Blending

- A set of tile images is selected randomly from a directory.
- Each tile is resized to match grid cell dimensions.
- The tile is blended with the grid cell’s average color using weighted addition and mapped to its respective position in the mosaic.

### 4. Performance Evaluation

The generated mosaic is evaluated using the following metrics:

- **Mean Squared Error (MSE)**: Measures pixel-wise differences; lower values indicate greater similarity.
- **Peak Signal-to-Noise Ratio (PSNR)**: Higher values indicate better image quality.
- **Histogram Metrics**: Includes correlation, Chi-Square, intersection, and Bhattacharyya distance for evaluating color distribution similarity.

## Performance Metrics

The following results were obtained during the evaluation:

| Metric                                | Value    | Description                                                          |
| ------------------------------------- | -------- | -------------------------------------------------------------------- |
| **Mean Squared Error (MSE)**          | 99.50    | Lower values indicate greater similarity.                            |
| **Peak Signal-to-Noise Ratio (PSNR)** | 16.98 dB | Higher values indicate better quality (30–50 dB is typical).         |
| **Histogram Correlation**             | 0.1416   | Values range from -1 to 1; closer to 1 indicates strong correlation. |
| **Histogram Chi-Square**              | 201.4498 | Lower values indicate better matches.                                |
| **Histogram Intersection**            | 27.6817  | Higher values indicate better overlap.                               |
| **Bhattacharyya Distance**            | 0.9011   | Lower values indicate greater similarity.                            |

## Results

- **Execution Time**: ~3 seconds for a 1920x1920 image.
- **Output Quality**: Metrics indicate lower similarity between the mosaic and the original image, with opportunities to enhance blending and tile matching.

## Additional Improvements

This application demonstrates a functional and efficient pipeline for generating mosaics. The blending and tile mapping produce visually coherent results, but further improvements are possible:

- **Tile Variety**: Increasing the diversity of tiles can enhance color matching.
- **Blending Optimization**: Refining blending techniques can improve the mosaic’s fidelity to the original.
- **Histogram Similarity**: Optimizing histogram metrics can achieve better color distribution similarity.

## Future updates will focus on:

- Enhancing image similarity by optimizing blending techniques.
- Increasing tile diversity for better results.
- Improving histogram-based similarity metrics.

These enhancements will improve the visual quality and computational performance of the application.
