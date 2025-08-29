# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an emotion recognition research project focused on analyzing the EXPW (Expression in the Wild) dataset. The repository contains analysis tools for evaluating multiple vision-language models' performance on facial emotion recognition tasks.

## Key Files and Structure

### Core Analysis Scripts
- `analyze_emotion_data.py` - Full-featured emotion dataset analyzer with visualization dependencies (matplotlib, seaborn)
- `analyze_emotion_data_simple.py` - Simplified analyzer using only standard library + pandas/numpy
- `expw_results_all_methods.csv` - Main dataset file containing emotion recognition results from 9 different models

### Dataset Structure
The CSV dataset contains 1,500 samples with 27 features including:
- **Ground truth labels**: 7 emotion categories (angry, disgust, fear, happy, sad, surprise, neutral)
- **Image metadata**: Paths, dimensions (640×605 avg), face bounding boxes in multiple formats
- **Model predictions**: Results from 9 different vision-language models
- **Response data**: Structured reasoning traces with `<think>` and `<answer>` tags

### Image Data
- `dfew_128/New folder/` - Contains 128 emotion test images in JPG format
- Images are numbered (1.jpg through 200.jpg) with gaps in sequence
- Standard resolution around 640×605 pixels

## Development Commands

### Environment Setup
```bash
# Create conda environment for analysis
conda create -n emotion_analysis python=3.9 -y

# Install required dependencies
pip install pandas numpy matplotlib seaborn
```

### Running Analysis
```bash
# Full analysis with visualizations
python analyze_emotion_data.py

# Simplified analysis (standard library only)
python analyze_emotion_data_simple.py
```

## Data Architecture

### Emotion Mapping
The dataset uses consistent emotion mapping:
- 0: angry, 1: disgust, 2: fear, 3: happy, 4: sad, 5: surprise, 6: neutral

### Model Categories
1. **EXPW Models**: Fine-tuned variants (7B/3B, with/without training)
2. **Vision-Language Models**: VideoLLaMA2, LLaVANext, VILA, MiniCPM-V, VideoCCAM

### Bounding Box Formats
- `original_face_bboxes` - Original image coordinates
- `face_bboxes_resized` - Resized image coordinates  
- `face_bboxes_retina` - Retina-scaled coordinates
- `face_bboxes` - Standard format coordinates
- `face_bboxes_resized_relative_area_sqrt` - Relative face area measurements

### Response Structure
Model responses follow a structured format:
- `full_response` - Complete reasoning trace with `<think>` and `<answer>` tags
- `response_prediction` - Extracted prediction text
- Individual model prediction columns with emotion labels

## Working with the Dataset

### Loading Data
```python
import pandas as pd
df = pd.read_csv('expw_results_all_methods.csv')
```

### Key Columns to Focus On
- `ground_truth` - Manual emotion annotations (string format)
- `expression_label` - Numeric emotion labels (0-6)
- Model prediction columns ending with `_label` or `_prediction`
- Bounding box columns for face detection analysis

### Common Analysis Patterns
- Model performance comparison against ground truth
- Inter-model agreement analysis
- Face detection accuracy evaluation
- Response quality assessment

The codebase is designed for emotion recognition research and benchmarking, with comprehensive tools for analyzing multi-modal model performance on facial emotion classification tasks.