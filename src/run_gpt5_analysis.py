#!/usr/bin/env python3
"""
Demo script to run GPT-5 emotion analysis

Before running this script:
1. Get your API key from: https://openrouter.ai/keys
2. Add it to the .env file (already configured)
3. Install dependencies: pip install -r requirements.txt

The script will automatically load the API key from the .env file.
"""

import os
from pathlib import Path
from analyzers.gpt5_emotion_analyzer import GPT5EmotionAnalyzer

def main():
    """Run GPT-5 emotion analysis demo"""
    
    # Check for API key
    if not os.getenv('OPENROUTER_API_KEY'):
        print("❌ Please set OPENROUTER_API_KEY environment variable")
        print("Get your API key from: https://openrouter.ai/keys")
        print("\nWindows: set OPENROUTER_API_KEY=your_api_key_here")
        print("Linux/Mac: export OPENROUTER_API_KEY=your_api_key_here")
        return
    
    # Configuration
    image_folder = Path(r"../dfew_128/New folder")
    num_images = None  # None = analyze ALL images (128 total)
    
    try:
        # Initialize analyzer
        print("🔧 Initializing GPT-5 analyzer...")
        analyzer = GPT5EmotionAnalyzer()
        
        # Run analysis
        results = analyzer.analyze_batch(image_folder, num_images)
        
        # Save results
        if results:
            analyzer.save_results(results, "../results/gpt5_emotion_results.json")
            print("\n🎉 Demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()