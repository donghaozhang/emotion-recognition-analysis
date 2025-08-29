#!/usr/bin/env python3
"""
GLM-4.5V Performance Evaluation Script

This script compares GLM-4.5V emotion predictions with ground truth labels
from the EXPW dataset and generates comprehensive evaluation metrics and
visualizations in HTML format.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# For Excel file support
try:
    import openpyxl
except ImportError:
    print("Warning: openpyxl not installed. Excel file support may be limited.")
    print("Install with: pip install openpyxl")

class GLMPerformanceEvaluator:
    """Evaluate GLM-4.5V predictions against ground truth"""
    
    def __init__(self):
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.color_map = {
            'angry': '#e74c3c',
            'disgust': '#8e44ad',
            'fear': '#34495e',
            'happy': '#f1c40f',
            'sad': '#3498db',
            'surprise': '#e67e22',
            'neutral': '#95a5a6'
        }
        
    def load_data(self):
        """Load GLM predictions and ground truth data"""
        print("📊 Loading data...")
        
        # Load GLM predictions
        glm_results_path = Path('glm_emotion_results.json')
        if not glm_results_path.exists():
            print("❌ GLM results file not found. Please run the analysis first.")
            return None, None
            
        with open(glm_results_path, 'r', encoding='utf-8') as f:
            glm_results = json.load(f)
            
        # Convert to DataFrame
        glm_df = pd.DataFrame(glm_results)
        
        # Load ground truth from Excel file
        excel_path = Path('valid_set_1_test_for_inference_add_width_height.xlsx')
        if excel_path.exists():
            ground_truth_df = pd.read_excel(excel_path)
            print(f"✅ Loaded ground truth from Excel file")
        else:
            # Fallback to CSV if Excel not found
            expw_path = Path('expw_results_all_methods.csv')
            if not expw_path.exists():
                print("❌ Ground truth file not found.")
                return glm_df, None
            ground_truth_df = pd.read_csv(expw_path)
            print(f"✅ Loaded ground truth from CSV file")
        
        print(f"✅ Loaded {len(glm_df)} GLM predictions")
        print(f"✅ Loaded {len(ground_truth_df)} ground truth samples")
        
        return glm_df, ground_truth_df
    
    def match_predictions_with_ground_truth(self, glm_df, ground_truth_df):
        """Match GLM predictions with ground truth based on image names"""
        print("\n🔄 Matching predictions with ground truth...")
        
        matched_data = []
        unmatched_images = []
        
        for _, glm_row in glm_df.iterrows():
            # Extract image number from filename (e.g., '1.jpg' -> 1)
            image_name = glm_row['image_name']
            try:
                image_num = int(Path(image_name).stem)
            except ValueError:
                print(f"⚠️  Could not parse image number from: {image_name}")
                continue
            
            # Find corresponding ground truth by video_name (which contains the image number)
            gt_match = ground_truth_df[ground_truth_df['video_name'] == image_num]
            
            if not gt_match.empty:
                # Check if using Excel format with label_grounding_truth column
                if 'label_grounding_truth' in ground_truth_df.columns:
                    ground_truth = gt_match.iloc[0]['label_grounding_truth']
                else:
                    # Fallback to ground_truth column for CSV format
                    ground_truth = gt_match.iloc[0]['ground_truth']
                
                matched_data.append({
                    'image_name': image_name,
                    'image_number': image_num,
                    'predicted': glm_row['predicted_emotion'],
                    'ground_truth': ground_truth,
                    'tokens_used': glm_row.get('tokens_used', 0),
                    'raw_response': glm_row.get('raw_response', '')
                })
            else:
                unmatched_images.append(image_name)
        
        matched_df = pd.DataFrame(matched_data)
        
        print(f"✅ Matched {len(matched_df)} predictions with ground truth")
        if unmatched_images:
            print(f"⚠️  Could not find ground truth for {len(unmatched_images)} images")
            print(f"   Unmatched: {', '.join(unmatched_images[:5])}{'...' if len(unmatched_images) > 5 else ''}")
        
        return matched_df
    
    def calculate_metrics(self, matched_df):
        """Calculate evaluation metrics"""
        print("\n📈 Calculating metrics...")
        
        y_true = matched_df['ground_truth'].values
        y_pred = matched_df['predicted'].values
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Get unique labels present in the data
        unique_labels = sorted(list(set(y_true) | set(y_pred)))
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
        
        # Calculate per-class metrics
        report = classification_report(y_true, y_pred, labels=unique_labels, 
                                      target_names=unique_labels, output_dict=True)
        
        # Calculate macro and weighted F1 scores
        macro_f1 = f1_score(y_true, y_pred, labels=unique_labels, average='macro')
        weighted_f1 = f1_score(y_true, y_pred, labels=unique_labels, average='weighted')
        
        metrics = {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'confusion_matrix': cm,
            'labels': unique_labels,
            'classification_report': report,
            'total_samples': len(matched_df),
            'correct_predictions': sum(y_true == y_pred)
        }
        
        print(f"✅ Overall Accuracy: {accuracy:.2%}")
        print(f"✅ Macro F1 Score: {macro_f1:.3f}")
        print(f"✅ Weighted F1 Score: {weighted_f1:.3f}")
        
        return metrics
    
    def generate_html_report(self, matched_df, metrics):
        """Generate comprehensive HTML report with visualizations"""
        print("\n🎨 Generating HTML report...")
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GLM-4.5V Emotion Recognition Performance Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            background: rgba(255, 255, 255, 0.98);
            border-radius: 20px;
            padding: 40px;
            margin-bottom: 30px;
            text-align: center;
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
        }}
        
        .header h1 {{
            color: #2c3e50;
            font-size: 2.8em;
            margin-bottom: 15px;
            font-weight: 700;
        }}
        
        .subtitle {{
            color: #7f8c8d;
            font-size: 1.3em;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .metric-card {{
            background: rgba(255, 255, 255, 0.98);
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.15);
        }}
        
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: block;
            margin-bottom: 10px;
        }}
        
        .metric-label {{
            color: #7f8c8d;
            font-size: 1.1em;
            font-weight: 500;
        }}
        
        .section {{
            background: rgba(255, 255, 255, 0.98);
            border-radius: 20px;
            padding: 35px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }}
        
        .section-title {{
            font-size: 2em;
            color: #2c3e50;
            margin-bottom: 25px;
            padding-bottom: 15px;
            border-bottom: 3px solid linear-gradient(135deg, #667eea, #764ba2);
            font-weight: 600;
        }}
        
        .confusion-matrix {{
            overflow-x: auto;
            margin: 20px 0;
        }}
        
        .confusion-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 14px;
        }}
        
        .confusion-table th, .confusion-table td {{
            padding: 12px;
            text-align: center;
            border: 1px solid #e0e0e0;
        }}
        
        .confusion-table th {{
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            font-weight: 600;
        }}
        
        .confusion-table td {{
            background: white;
            font-weight: 500;
        }}
        
        .diagonal {{
            background: rgba(46, 204, 113, 0.2) !important;
            font-weight: bold;
            color: #27ae60;
        }}
        
        .chart-container {{
            position: relative;
            height: 400px;
            margin: 30px 0;
        }}
        
        .performance-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        
        .performance-table th {{
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        
        .performance-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #e0e0e0;
        }}
        
        .performance-table tr:hover {{
            background: rgba(102, 126, 234, 0.05);
        }}
        
        .accuracy-ring {{
            width: 200px;
            height: 200px;
            margin: 0 auto;
        }}
        
        .sample-predictions {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        
        .prediction-card {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            border-left: 4px solid #667eea;
        }}
        
        .correct {{
            border-left-color: #27ae60;
            background: rgba(46, 204, 113, 0.05);
        }}
        
        .incorrect {{
            border-left-color: #e74c3c;
            background: rgba(231, 76, 60, 0.05);
        }}
        
        .prediction-header {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            font-weight: 600;
        }}
        
        .prediction-details {{
            color: #555;
            font-size: 0.95em;
        }}
        
        .heatmap-container {{
            margin: 30px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
        }}
        
        @media (max-width: 768px) {{
            .metrics-grid {{
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            }}
            
            .header h1 {{
                font-size: 2em;
            }}
            
            .confusion-table {{
                font-size: 12px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header class="header">
            <h1>🤖 GLM-4.5V Performance Report</h1>
            <p class="subtitle">Emotion Recognition Evaluation on EXPW Dataset</p>
        </header>

        <!-- Key Metrics -->
        <section class="metrics-grid">
            <div class="metric-card">
                <span class="metric-value">{metrics['accuracy']:.1%}</span>
                <div class="metric-label">Overall Accuracy</div>
            </div>
            <div class="metric-card">
                <span class="metric-value">{metrics['correct_predictions']}/{metrics['total_samples']}</span>
                <div class="metric-label">Correct Predictions</div>
            </div>
            <div class="metric-card">
                <span class="metric-value">{metrics['macro_f1']:.3f}</span>
                <div class="metric-label">Macro F1 Score</div>
            </div>
            <div class="metric-card">
                <span class="metric-value">{metrics['weighted_f1']:.3f}</span>
                <div class="metric-label">Weighted F1 Score</div>
            </div>
            <div class="metric-card">
                <span class="metric-value">{len(metrics['labels'])}</span>
                <div class="metric-label">Emotion Classes</div>
            </div>
            <div class="metric-card">
                <span class="metric-value">{matched_df['tokens_used'].sum():,}</span>
                <div class="metric-label">Total Tokens Used</div>
            </div>
        </section>

        <!-- Confusion Matrix -->
        <section class="section">
            <h2 class="section-title">📊 Confusion Matrix</h2>
            <div class="confusion-matrix">
                <table class="confusion-table">
                    <thead>
                        <tr>
                            <th>True \\ Predicted</th>
                            {"".join([f'<th>{label.capitalize()}</th>' for label in metrics['labels']])}
                        </tr>
                    </thead>
                    <tbody>
"""
        
        # Add confusion matrix rows
        for i, true_label in enumerate(metrics['labels']):
            html_content += f"                        <tr>\n"
            html_content += f"                            <th>{true_label.capitalize()}</th>\n"
            for j, pred_label in enumerate(metrics['labels']):
                value = metrics['confusion_matrix'][i, j]
                cell_class = 'diagonal' if i == j else ''
                html_content += f"                            <td class='{cell_class}'>{value}</td>\n"
            html_content += f"                        </tr>\n"
        
        html_content += """
                    </tbody>
                </table>
            </div>
            
            <!-- Confusion Matrix Heatmap -->
            <div id="heatmap" class="heatmap-container"></div>
        </section>

        <!-- Per-Class Performance -->
        <section class="section">
            <h2 class="section-title">📈 Per-Class Performance</h2>
            <table class="performance-table">
                <thead>
                    <tr>
                        <th>Emotion</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1-Score</th>
                        <th>Support</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        # Add per-class metrics
        for label in metrics['labels']:
            if label in metrics['classification_report']:
                report = metrics['classification_report'][label]
                html_content += f"""
                    <tr>
                        <td><strong>{label.capitalize()}</strong></td>
                        <td>{report['precision']:.3f}</td>
                        <td>{report['recall']:.3f}</td>
                        <td>{report['f1-score']:.3f}</td>
                        <td>{int(report['support'])}</td>
                    </tr>
"""
        
        html_content += """
                </tbody>
            </table>
            
            <!-- Bar Chart -->
            <div class="chart-container">
                <canvas id="performanceChart"></canvas>
            </div>
        </section>

        <!-- Distribution Comparison -->
        <section class="section">
            <h2 class="section-title">🎯 Prediction vs Ground Truth Distribution</h2>
            <div class="chart-container">
                <canvas id="distributionChart"></canvas>
            </div>
        </section>

        <!-- Sample Predictions -->
        <section class="section">
            <h2 class="section-title">📝 Sample Predictions</h2>
            <div class="sample-predictions">
"""
        
        # Add sample predictions
        samples = matched_df.head(12)
        for _, row in samples.iterrows():
            is_correct = row['predicted'] == row['ground_truth']
            card_class = 'correct' if is_correct else 'incorrect'
            status = '✅' if is_correct else '❌'
            
            html_content += f"""
                <div class="prediction-card {card_class}">
                    <div class="prediction-header">
                        <span>{row['image_name']}</span>
                        <span>{status}</span>
                    </div>
                    <div class="prediction-details">
                        <div>🎯 Ground Truth: <strong>{row['ground_truth']}</strong></div>
                        <div>🤖 Predicted: <strong>{row['predicted']}</strong></div>
                        <div>🪙 Tokens: {row['tokens_used']}</div>
                    </div>
                </div>
"""
        
        html_content += """
            </div>
        </section>

    </div>

    <script>
        // Confusion Matrix Heatmap
        const confusionMatrix = """ + str(metrics['confusion_matrix'].tolist()) + """;
        const labels = """ + str(metrics['labels']) + """;
        
        const heatmapData = [{
            z: confusionMatrix,
            x: labels.map(l => l.charAt(0).toUpperCase() + l.slice(1)),
            y: labels.map(l => l.charAt(0).toUpperCase() + l.slice(1)),
            type: 'heatmap',
            colorscale: 'Blues',
            showscale: true,
            text: confusionMatrix.map(row => row.map(val => val.toString())),
            texttemplate: '%{text}',
            textfont: {
                size: 12
            }
        }];
        
        const heatmapLayout = {
            title: 'Confusion Matrix Heatmap',
            xaxis: {
                title: 'Predicted',
                side: 'bottom'
            },
            yaxis: {
                title: 'True',
                autorange: 'reversed'
            },
            width: 600,
            height: 500
        };
        
        Plotly.newPlot('heatmap', heatmapData, heatmapLayout);
        
        // Per-Class Performance Chart
        const performanceCtx = document.getElementById('performanceChart').getContext('2d');
        const classReport = """ + json.dumps(metrics['classification_report']) + """;
        
        const performanceData = {
            labels: labels.map(l => l.charAt(0).toUpperCase() + l.slice(1)),
            datasets: [
                {
                    label: 'Precision',
                    data: labels.map(l => classReport[l] ? classReport[l].precision : 0),
                    backgroundColor: 'rgba(52, 152, 219, 0.8)',
                    borderColor: 'rgba(41, 128, 185, 1)',
                    borderWidth: 2
                },
                {
                    label: 'Recall',
                    data: labels.map(l => classReport[l] ? classReport[l].recall : 0),
                    backgroundColor: 'rgba(155, 89, 182, 0.8)',
                    borderColor: 'rgba(142, 68, 173, 1)',
                    borderWidth: 2
                },
                {
                    label: 'F1-Score',
                    data: labels.map(l => classReport[l] ? classReport[l]['f1-score'] : 0),
                    backgroundColor: 'rgba(46, 204, 113, 0.8)',
                    borderColor: 'rgba(39, 174, 96, 1)',
                    borderWidth: 2
                }
            ]
        };
        
        new Chart(performanceCtx, {
            type: 'bar',
            data: performanceData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    title: {
                        display: true,
                        text: 'Per-Class Metrics Comparison'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1,
                        title: {
                            display: true,
                            text: 'Score'
                        }
                    }
                }
            }
        });
        
        // Distribution Comparison Chart
        const distributionCtx = document.getElementById('distributionChart').getContext('2d');
        """ 
        
        # Calculate distributions
        true_dist = matched_df['ground_truth'].value_counts()
        pred_dist = matched_df['predicted'].value_counts()
        
        html_content += f"""
        const trueDistribution = {true_dist.to_dict()};
        const predDistribution = {pred_dist.to_dict()};
        
        const distributionData = {{
            labels: labels.map(l => l.charAt(0).toUpperCase() + l.slice(1)),
            datasets: [
                {{
                    label: 'Ground Truth',
                    data: labels.map(l => trueDistribution[l] || 0),
                    backgroundColor: 'rgba(52, 152, 219, 0.6)',
                    borderColor: 'rgba(41, 128, 185, 1)',
                    borderWidth: 2
                }},
                {{
                    label: 'GLM-4.5V Predictions',
                    data: labels.map(l => predDistribution[l] || 0),
                    backgroundColor: 'rgba(231, 76, 60, 0.6)',
                    borderColor: 'rgba(192, 57, 43, 1)',
                    borderWidth: 2
                }}
            ]
        }};
        
        new Chart(distributionCtx, {{
            type: 'bar',
            data: distributionData,
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        position: 'top',
                    }},
                    title: {{
                        display: true,
                        text: 'Distribution Comparison: Ground Truth vs Predictions'
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Count'
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
        
        # Save HTML report
        output_path = Path('glm_performance_report.html')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ HTML report saved to: {output_path.absolute()}")
        
        return output_path
    
    def run_evaluation(self):
        """Run complete evaluation pipeline"""
        print("🚀 Starting GLM-4.5V Performance Evaluation")
        print("="*50)
        
        # Load data
        glm_df, ground_truth_df = self.load_data()
        if glm_df is None:
            return
        
        # Match predictions with ground truth
        if ground_truth_df is not None:
            matched_df = self.match_predictions_with_ground_truth(glm_df, ground_truth_df)
            
            if matched_df.empty:
                print("❌ No matching samples found between predictions and ground truth")
                return
            
            # Calculate metrics
            metrics = self.calculate_metrics(matched_df)
            
            # Generate HTML report
            report_path = self.generate_html_report(matched_df, metrics)
            
            print("\n" + "="*50)
            print("📊 EVALUATION SUMMARY")
            print("="*50)
            print(f"✅ Total Samples Evaluated: {metrics['total_samples']}")
            print(f"✅ Overall Accuracy: {metrics['accuracy']:.2%}")
            print(f"✅ Macro F1 Score: {metrics['macro_f1']:.3f}")
            print(f"✅ Weighted F1 Score: {metrics['weighted_f1']:.3f}")
            print(f"✅ Report saved to: {report_path}")
            print("\n🎉 Evaluation complete! Open the HTML report to view detailed results.")
        else:
            print("⚠️  Could not load ground truth data for evaluation")


def main():
    """Main function"""
    evaluator = GLMPerformanceEvaluator()
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()