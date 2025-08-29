#!/usr/bin/env python3
"""
Emotion Recognition Dataset Analysis Script

This script analyzes the EXPW (Expression in the Wild) dataset structure and provides
comprehensive insights about the emotion recognition data, model predictions, and
dataset characteristics.

Author: Generated for emotion recognition analysis
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json
import ast
from pathlib import Path

class EmotionDatasetAnalyzer:
    """
    Comprehensive analyzer for emotion recognition dataset
    """

    def __init__(self, csv_path):
        """
        Initialize the analyzer with CSV file path

        Args:
            csv_path (str): Path to the CSV file containing emotion data
        """
        self.csv_path = csv_path
        self.df = None
        self.emotion_mapping = {
            0: 'angry',
            1: 'disgust',
            2: 'fear',
            3: 'happy',
            4: 'sad',
            5: 'surprise',
            6: 'neutral'
        }

        # Model prediction columns
        self.prediction_columns = [
            'expw_label_prediction_7B',
            'expw_label_prediction_without_train_7B',
            'expw_label_prediction_3B',
            'expw_label_prediction_without_train_3B',
            'videollama2_result_label',
            'llavanext_result_label',
            'vila_result_label',
            'minicpmv_result_label',
            'videoccam_result_label'
        ]

        self.load_data()

    def load_data(self):
        """Load and preprocess the CSV data"""
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"✓ Successfully loaded {len(self.df)} rows and {len(self.df.columns)} columns")
        except Exception as e:
            print(f"✗ Error loading CSV file: {e}")
            return False
        return True

    def analyze_basic_structure(self):
        """Analyze basic dataset structure"""
        print("\n" + "="*60)
        print("BASIC DATASET STRUCTURE ANALYSIS")
        print("="*60)

        print(f"\n📊 Dataset Overview:")
        print(f"   • Total samples: {len(self.df):,}")
        print(f"   • Total features: {len(self.df.columns)}")
        print(".2f")

        print(f"\n🏷️  Data Types Distribution:")
        dtype_counts = self.df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"   • {dtype}: {count} columns")

        print(f"\n📋 Column Categories:")

        # Categorize columns
        image_cols = [col for col in self.df.columns if 'image' in col.lower() or 'path' in col.lower()]
        bbox_cols = [col for col in self.df.columns if 'bbox' in col.lower()]
        emotion_cols = [col for col in self.df.columns if 'expression' in col.lower() or 'ground' in col.lower()]
        prediction_cols = [col for col in self.df.columns if 'prediction' in col.lower() or 'result' in col.lower()]
        response_cols = [col for col in self.df.columns if 'response' in col.lower()]

        print(f"   • Image/Video info: {len(image_cols)} columns")
        print(f"   • Face bounding boxes: {len(bbox_cols)} columns")
        print(f"   • Emotion labels: {len(emotion_cols)} columns")
        print(f"   • Model predictions: {len(prediction_cols)} columns")
        print(f"   • Response data: {len(response_cols)} columns")

    def analyze_emotion_distribution(self):
        """Analyze emotion label distribution"""
        print("\n" + "="*60)
        print("EMOTION DISTRIBUTION ANALYSIS")
        print("="*60)

        # Ground truth distribution
        gt_counts = self.df['ground_truth'].value_counts()
        print(f"\n🎯 Ground Truth Emotion Distribution:")
        for emotion, count in gt_counts.items():
            percentage = (count / len(self.df)) * 100
            print("8s")

        # Expression label distribution (numeric)
        expr_counts = self.df['expression_label'].value_counts().sort_index()
        print(f"\n🔢 Expression Label (Numeric) Distribution:")
        for label_num, count in expr_counts.items():
            emotion_name = self.emotion_mapping.get(label_num, 'unknown')
            percentage = (count / len(self.df)) * 100
            print("2d")

        # Check consistency between expression_label and ground_truth
        inconsistencies = 0
        for idx, row in self.df.iterrows():
            expected_emotion = self.emotion_mapping.get(row['expression_label'], 'unknown')
            if expected_emotion != row['ground_truth']:
                inconsistencies += 1

        if inconsistencies > 0:
            print(f"\n⚠️  Found {inconsistencies} inconsistencies between expression_label and ground_truth")
        else:
            print(f"\n✅ All expression labels are consistent with ground truth")

    def analyze_bounding_boxes(self):
        """Analyze face bounding box data"""
        print("\n" + "="*60)
        print("FACE BOUNDING BOX ANALYSIS")
        print("="*60)

        bbox_columns = [col for col in self.df.columns if 'bbox' in col.lower()]

        print(f"\n📦 Bounding Box Columns: {len(bbox_columns)}")
        for col in bbox_columns:
            print(f"   • {col}")

        # Analyze bbox data types and content
        print(f"\n🔍 Bounding Box Data Analysis:")

        # Check if bbox data is properly formatted
        for col in bbox_columns:
            try:
                # Try to parse first few bbox entries
                sample_bbox = self.df[col].dropna().iloc[0] if not self.df[col].empty else None
                if sample_bbox:
                    # Try to parse as JSON/list
                    try:
                        parsed = ast.literal_eval(sample_bbox)
                        print(f"   • {col}: Properly formatted (type: {type(parsed).__name__})")
                        if isinstance(parsed, list) and len(parsed) > 0:
                            if isinstance(parsed[0], list):
                                print(f"     └─ Contains {len(parsed)} face(s), each with {len(parsed[0])} coordinates")
                            else:
                                print(f"     └─ Single face with {len(parsed)} coordinates")
                    except:
                        print(f"   • {col}: String format (needs parsing)")
                else:
                    print(f"   • {col}: Empty or null values")
            except Exception as e:
                print(f"   • {col}: Error analyzing - {str(e)}")

        # Analyze image dimensions
        print(f"\n📏 Image Dimensions Statistics:")
        print(f"   • Width - Mean: {self.df['width'].mean():.1f}, Min: {self.df['width'].min()}, Max: {self.df['width'].max()}")
        print(f"   • Height - Mean: {self.df['height'].mean():.1f}, Min: {self.df['height'].min()}, Max: {self.df['height'].max()}")

        # Calculate bbox relative area if available
        if 'face_bboxes_resized_relative_area_sqrt' in self.df.columns:
            area_col = 'face_bboxes_resized_relative_area_sqrt'
            valid_areas = pd.to_numeric(self.df[area_col], errors='coerce').dropna()
            if len(valid_areas) > 0:
                print(f"\n📐 Face Size Analysis (relative area sqrt):")
                print(f"   • Mean: {valid_areas.mean():.4f}")
                print(f"   • Median: {valid_areas.median():.4f}")
                print(f"   • Std: {valid_areas.std():.4f}")
                print(f"   • Min: {valid_areas.min():.4f}")
                print(f"   • Max: {valid_areas.max():.4f}")

    def analyze_model_predictions(self):
        """Analyze model prediction performance and distribution"""
        print("\n" + "="*60)
        print("MODEL PREDICTIONS ANALYSIS")
        print("="*60)

        print(f"\n🤖 Available Prediction Models: {len(self.prediction_columns)}")
        for i, model in enumerate(self.prediction_columns, 1):
            print(f"   {i}. {model}")

        # Analyze prediction distributions for each model
        print(f"\n📊 Prediction Distribution by Model:")

        model_stats = {}
        for model_col in self.prediction_columns:
            if model_col in self.df.columns:
                predictions = self.df[model_col].dropna()
                unique_preds = predictions.unique()

                # Count valid predictions (non-null, non-empty)
                valid_count = len(predictions)
                total_count = len(self.df)
                coverage = (valid_count / total_count) * 100

                model_stats[model_col] = {
                    'total_predictions': total_count,
                    'valid_predictions': valid_count,
                    'coverage': coverage,
                    'unique_values': len(unique_preds),
                    'unique_predictions': unique_preds[:10]  # Show first 10
                }

                print(f"\n🧠 {model_col}:")
                print(f"   • Coverage: {coverage:.1f}% ({valid_count}/{total_count})")
                print(f"   • Unique predictions: {len(unique_preds)}")
                if len(unique_preds) <= 10:
                    print(f"   • Values: {sorted(unique_preds)}")
                else:
                    print(f"   • Sample values: {sorted(unique_preds)[:5]}...{sorted(unique_preds)[-5:]}")

        # Compare model agreement
        self.analyze_model_agreement()

    def analyze_model_agreement(self):
        """Analyze agreement between different models"""
        print(f"\n🤝 Model Agreement Analysis:")

        # Get valid predictions for all models
        valid_predictions = {}
        for model_col in self.prediction_columns:
            if model_col in self.df.columns:
                preds = self.df[model_col].dropna()
                if len(preds) > 0:
                    valid_predictions[model_col] = preds

        if len(valid_predictions) >= 2:
            # Calculate pairwise agreement
            model_names = list(valid_predictions.keys())

            print(f"\n📈 Pairwise Model Agreement:")
            for i in range(len(model_names)):
                for j in range(i+1, len(model_names)):
                    model1, model2 = model_names[i], model_names[j]

                    # Find common indices
                    common_indices = set(valid_predictions[model1].index) & set(valid_predictions[model2].index)

                    if len(common_indices) > 0:
                        preds1 = valid_predictions[model1].loc[list(common_indices)]
                        preds2 = valid_predictions[model2].loc[list(common_indices)]

                        agreement = (preds1 == preds2).sum() / len(common_indices) * 100
                        print("30s")

    def analyze_response_data(self):
        """Analyze the full response data structure"""
        print("\n" + "="*60)
        print("RESPONSE DATA ANALYSIS")
        print("="*60)

        if 'full_response' in self.df.columns:
            print(f"\n💬 Full Response Analysis:")

            # Sample some responses
            sample_responses = self.df['full_response'].dropna().head(3)

            for i, response in enumerate(sample_responses, 1):
                print(f"\n📝 Sample Response {i}:")
                print(f"   Length: {len(response)} characters")

                # Check if it contains think/answer tags
                if '<think>' in response and '<answer>' in response:
                    print(f"   ✓ Contains structured think/answer format")
                elif '<think>' in response:
                    print(f"   ✓ Contains think tags")
                elif '<answer>' in response:
                    print(f"   ✓ Contains answer tags")
                else:
                    print(f"   - Plain text response")

                # Show first 200 characters
                preview = response[:200].replace('\n', ' ')
                print(f"   Preview: {preview}...")

        # Analyze response_prediction structure
        if 'response_prediction' in self.df.columns:
            print(f"\n🔮 Response Prediction Analysis:")

            sample_preds = self.df['response_prediction'].dropna().head(3)
            for i, pred in enumerate(sample_preds, 1):
                print(f"\n🎯 Sample Prediction {i}:")
                print(f"   Raw: {pred}")

                # Try to parse as JSON
                try:
                    parsed = ast.literal_eval(pred)
                    print(f"   ✓ Parsed as: {type(parsed).__name__}")
                    if isinstance(parsed, list) and len(parsed) > 0:
                        print(f"   ✓ Contains {len(parsed)} prediction(s)")
                        if isinstance(parsed[0], dict):
                            keys = list(parsed[0].keys())
                            print(f"   ✓ Keys: {keys}")
                except:
                    print(f"   ✗ Could not parse as structured data")

    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("\n" + "="*80)
        print("COMPREHENSIVE DATASET SUMMARY REPORT")
        print("="*80)

        print(f"\n📊 OVERVIEW:")
        print(f"   Dataset: EXPW (Expression in the Wild)")
        print(f"   Purpose: Emotion recognition from facial images")
        print(f"   Total Samples: {len(self.df):,}")
        print(f"   Features: {len(self.df.columns)}")

        print(f"\n🎯 EMOTIONS:")
        gt_counts = self.df['ground_truth'].value_counts()
        for emotion, count in gt_counts.items():
            percentage = (count / len(self.df)) * 100
            print("8s")

        print(f"\n🤖 MODELS ANALYZED:")
        for i, model in enumerate(self.prediction_columns, 1):
            if model in self.df.columns:
                valid_preds = self.df[model].dropna()
                coverage = (len(valid_preds) / len(self.df)) * 100
                print("25s")

        print(f"\n📦 DATA CHARACTERISTICS:")
        print(f"   • Image dimensions: {self.df['width'].mean():.0f} x {self.df['height'].mean():.0f} (average)")
        print(f"   • Face detection: Multiple bbox formats available")
        print(f"   • Model responses: Structured with reasoning traces")
        print(f"   • Ground truth: Manual emotion annotations")

        print(f"\n🔍 KEY INSIGHTS:")
        print(f"   • Multi-modal dataset with image, text, and structured predictions")
        print(f"   • Rich metadata including face detection coordinates")
        print(f"   • Multiple model architectures represented")
        print(f"   • Suitable for emotion recognition benchmarking")

        print(f"\n💡 USE CASES:")
        print(f"   • Model performance comparison")
        print(f"   • Emotion recognition research")
        print(f"   • Multi-modal learning evaluation")
        print(f"   • Face detection and analysis")

    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("🚀 Starting Emotion Dataset Analysis...")
        print(f"📁 Analyzing file: {self.csv_path}")

        if not self.load_data():
            return

        # Run all analysis components
        self.analyze_basic_structure()
        self.analyze_emotion_distribution()
        self.analyze_bounding_boxes()
        self.analyze_model_predictions()
        self.analyze_response_data()
        self.generate_summary_report()

        print(f"\n✅ Analysis Complete!")
        print(f"📋 Dataset understanding script finished successfully.")


def main():
    """Main function to run the analysis"""

    # Define the CSV file path
    csv_file = "expw_results_all_methods.csv"

    # Check if file exists
    if not Path(csv_file).exists():
        print(f"❌ Error: File '{csv_file}' not found in current directory")
        print(f"   Current directory: {Path.cwd()}")
        return

    # Create analyzer and run analysis
    analyzer = EmotionDatasetAnalyzer(csv_file)
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()
