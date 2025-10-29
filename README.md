# K-Means Background Removal for Image Segmentation

A machine learning project implementing K-means clustering for automated background removal as a preprocessing step for image classification models. This project demonstrates unsupervised learning techniques applied to computer vision tasks using the DUTS saliency detection dataset.

## Project Overview

This project explores the use of K-means clustering to segment images and remove backgrounds, potentially improving the performance of downstream image classification models. The approach identifies background regions by analyzing color clusters that predominantly appear at image borders.

### Installation

To install the required packages, you can use pip:

```bash
pip install -r requirements.txt
```

### Key Features

- **Automated Background Removal**: Uses K-means clustering in LAB color space to identify and remove background regions
- **Border-based Background Detection**: Intelligently identifies background clusters by analyzing border pixels
- **Performance Evaluation**: Compares results against ground truth masks using Intersection over Union (IoU) metrics
- **Batch Processing**: Processes entire datasets efficiently with progress tracking

## Dataset

This project uses the [DUTS (Densely Annotated UTD Salient Object) Dataset](https://kaggle.com/datasets/balraj98/duts-saliency-detection-dataset), which contains:

- High-quality images with various objects
- Corresponding ground truth segmentation masks
- Diverse scenes and object types for robust evaluation

## Methodology

1. **Image Preprocessing**: Images are rescaled and converted to LAB color space for better perceptual uniformity
2. **K-means Clustering**: Pixels are clustered into 3 groups based on color similarity
3. **Background Identification**: Border analysis determines which clusters represent background
4. **Mask Generation**: Non-background pixels are preserved while background is removed
5. **Evaluation**: IoU scores compare results with ground truth masks

## Implementation Details

### Core Algorithm

- **Color Space**: LAB color space for perceptually uniform clustering
- **Cluster Count**: 3 clusters (typically: foreground + 2 background variations)
- **Distance Metric**: Euclidean distance in LAB space
- **Convergence**: 30 iterations of cluster center refinement

### Key Functions

- `Kmeans()`: Core clustering implementation
- `remove_background()`: Complete background removal pipeline
- `cluster_assignments()`: Assigns pixels to nearest cluster centers

## Results

The algorithm achieves varying IoU scores depending on image complexity:

- **Simple backgrounds**: Higher IoU scores (>0.7)
- **Complex scenes**: Lower IoU scores but still meaningful segmentation
- **Average performance**: See statistics in the notebook for detailed metrics

## Requirements

```
numpy
pandas
matplotlib
scikit-image
scikit-learn
kagglehub
tqdm
pathlib
```

## Usage

1. **Setup Environment**: Install required dependencies
2. **Download Data**: The notebook automatically downloads the DUTS dataset via kagglehub
3. **Run Analysis**: Execute the Jupyter notebook `lab.ipynb` to reproduce results
4. **View Results**: Processed images and evaluation metrics are generated automatically

## File Structure

```
.
├── lab.ipynb           # Main analysis notebook
├── readme              # Project documentation
├── .gitignore         # Git ignore file
```

## Authors

- Peter Dall-Hansen
- Gustav Jensen
- Anes K
- Peter Dall-Hansen

## Course Information

This project was developed as part of the **Introduction to Intelligent Systems** (Introduction to Intelligent Systems) course at DTU (Technical University of Denmark) - Lab Assignment 3.

## License

This project is for educational purposes as part of academic coursework.
