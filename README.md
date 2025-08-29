# 🎭 Emotion Recognition Analysis Framework

A comprehensive multi-model emotion recognition evaluation system using state-of-the-art vision-language models on the EXPW (Expression in the Wild) dataset.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Models](https://img.shields.io/badge/Models-2-orange.svg)

## 🔗 Live Demo

**[📊 View Interactive Report](https://donghaozhang.github.io/emotion-recognition-analysis/)** - Multi-model comparison dashboard

## 🏆 Performance Results

| Model | Accuracy | Macro F1 | Speed | Cost Efficiency |
|-------|----------|----------|-------|-----------------|
| **🥇 Gemini 2.5 Flash** | **67.2%** | **0.569** | **2.9s/img** | High tokens (1,392/img) |
| **🥈 GLM-4.5V** | 30.5% | 0.303 | 4.5s/img | **Low tokens (379/img)** |

## 📁 Project Structure

```
emotion-recognition-analysis/
├── 📄 README.md                   # This file
├── ⚙️ requirements.txt            # Python dependencies
├── 🔧 .env                       # API configuration (not committed)
├── 📋 CLAUDE.md                  # Development guidelines
│
├── 📂 src/                       # Source code
│   ├── 📂 analyzers/            # Model implementations
│   │   ├── 🤖 glm_emotion_analyzer.py      # GLM-4.5V analyzer
│   │   └── 🤖 gemini_emotion_analyzer.py   # Gemini 2.5 Flash analyzer
│   │
│   ├── 🚀 run_glm_analysis.py    # GLM analysis runner
│   ├── 🚀 run_gemini_analysis.py # Gemini analysis runner
│   │
│   └── 📂 evaluation/           # Evaluation scripts
│       ├── 📊 evaluate_glm_performance.py        # Single model evaluation
│       └── 📊 evaluate_multi_model_performance.py # Multi-model comparison
│
├── 📂 data/                      # Dataset files
│   ├── 📊 valid_set_1_test_for_inference_add_width_height.xlsx
│   └── 📊 expw_results_all_methods.csv
│
├── 📂 results/                   # Analysis results
│   ├── 📈 glm_emotion_results.json      # GLM predictions
│   ├── 📈 glm_emotion_results.csv       # GLM summary
│   ├── 📈 gemini_emotion_results.json   # Gemini predictions  
│   └── 📈 gemini_emotion_results.csv    # Gemini summary
│
├── 📂 reports/                   # HTML reports & dashboards
│   ├── 🎨 multi_model_comparison_report.html    # Performance comparison
│   ├── 🎨 multi_model_emotion_report.html       # Interactive dashboard
│   ├── 🎨 glm_performance_report.html           # GLM detailed report
│   └── 🎨 emotion_analysis_dashboard.html       # Dataset overview
│
├── 📂 dfew_128/                  # Test images (128 EXPW samples)
└── 📂 archive/                   # Historical data & backups
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/donghaozhang/emotion-recognition-analysis.git
cd emotion-recognition-analysis

# Create conda environment
conda create -n emotion_analysis python=3.9 -y
conda activate emotion_analysis

# Install dependencies
pip install -r requirements.txt
```

### 2. API Configuration

Get your OpenRouter API key from [openrouter.ai/keys](https://openrouter.ai/keys) and create `.env` file:

```bash
# .env file
OPENROUTER_API_KEY=your_api_key_here
```

### 3. Run Analysis

```bash
# Run GLM-4.5V analysis
cd src
python run_glm_analysis.py

# Run Gemini 2.5 Flash analysis  
python run_gemini_analysis.py

# Compare both models
cd evaluation
python evaluate_multi_model_performance.py
```

## 📊 Models Comparison

### GLM-4.5V
- **Accuracy**: 30.5%
- **Speed**: 4.5s per image
- **Cost**: 379 tokens per image (73% cheaper)
- **Best for**: Cost-sensitive applications

### Gemini 2.5 Flash Image Preview  
- **Accuracy**: 67.2% (2.2x better)
- **Speed**: 2.9s per image (55% faster)
- **Cost**: 1,392 tokens per image
- **Best for**: High-accuracy requirements

## 🎯 Emotion Categories

The system recognizes 7 emotion categories:
- **Angry** 😠
- **Disgust** 🤢  
- **Fear** 😨
- **Happy** 😊
- **Neutral** 😐
- **Sad** 😢
- **Surprise** 😲

## 📈 Dataset Information

- **Source**: EXPW (Expression in the Wild) dataset
- **Test Set**: 128 carefully selected facial expression images
- **Format**: JPG images, ~640x605 resolution
- **Ground Truth**: Human-annotated emotion labels
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

## 🔧 Development

### Adding New Models

1. Create analyzer in `src/analyzers/new_model_analyzer.py`
2. Implement required interface:
   ```python
   def analyze_emotion(self, image_path: Path) -> Dict
   def analyze_batch(self, image_folder: Path, num_images: int) -> List[Dict]
   def save_results(self, results: List[Dict], output_file: str)
   ```
3. Add to evaluation framework in `evaluate_multi_model_performance.py`
4. Create runner script in `src/run_new_model_analysis.py`

### Key Features

- **🔄 Real-time Progress Monitoring**: Live updates during batch processing
- **📊 Comprehensive Metrics**: Accuracy, F1-scores, confusion matrices
- **🎨 Interactive Reports**: HTML dashboards with charts and visualizations  
- **⚡ Parallel Processing**: Efficient batch analysis with rate limiting
- **🔒 Error Handling**: Robust error recovery and logging
- **📁 Organized Structure**: Clean, maintainable codebase

## 📋 Requirements

```txt
openai>=1.3.0        # OpenRouter API access
pillow>=9.0.0        # Image processing
python-dotenv>=0.19.0 # Environment variables
pandas>=1.5.0        # Data manipulation
numpy>=1.21.0        # Numerical operations
matplotlib>=3.5.0    # Plotting
seaborn>=0.11.0      # Statistical visualization
scikit-learn>=1.1.0  # Machine learning metrics
openpyxl>=3.0.0      # Excel file handling
```

## 🎨 Reports & Visualizations

The framework generates several types of reports:

1. **📊 Multi-Model Comparison**: Side-by-side performance analysis
2. **🎯 Confusion Matrices**: Per-model accuracy breakdown by emotion
3. **📈 Performance Metrics**: Precision, recall, F1-scores
4. **⏱️ Speed & Cost Analysis**: Processing time and token usage
5. **📋 Interactive Dashboards**: Switch between models dynamically

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenRouter**: API access to cutting-edge language models
- **EXPW Dataset**: High-quality emotion recognition data
- **GLM-4.5V**: Advanced reasoning capabilities from Zhipu AI
- **Gemini 2.5 Flash**: Google's fast vision-language model

## 📞 Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/donghaozhang/emotion-recognition-analysis/issues)
- 📧 **Contact**: Create an issue for questions or suggestions
- 📖 **Documentation**: Check `CLAUDE.md` for development guidelines

---

**🤖 Generated with Claude Code** | **⭐ Star this repo if you find it useful!**