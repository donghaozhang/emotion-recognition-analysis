#!/usr/bin/env python3
"""
Demo script to run GLM-4.5V emotion analysis

Before running this script:
1. Get your API key from: https://openrouter.ai/keys
2. Add it to the .env file (already configured)
3. Install dependencies: pip install -r requirements.txt

The script will automatically load the API key from the .env file.
"""

import os
from pathlib import Path
from glm_emotion_analyzer import GLMEmotionAnalyzer

def main():
    """Run emotion analysis demo"""
    
    # Check for API key
    if not os.getenv('OPENROUTER_API_KEY'):
        print("❌ Please set OPENROUTER_API_KEY environment variable")
        print("Get your API key from: https://openrouter.ai/keys")
        print("\nWindows: set OPENROUTER_API_KEY=your_api_key_here")
        print("Linux/Mac: export OPENROUTER_API_KEY=your_api_key_here")
        return
    
    # Configuration
    image_folder = Path(r"C:\Users\zdhpe\Desktop\emotion\dfew_128\New folder")
    num_images = None  # None = analyze ALL images (128 total)
    
    try:
        # Initialize analyzer
        print("🔧 Initializing GLM-4.5V analyzer...")
        analyzer = GLMEmotionAnalyzer()
        
        # Run analysis
        results = analyzer.analyze_batch(image_folder, num_images)
        
        # Save results
        if results:
            analyzer.save_results(results, "glm_emotion_results.json")
            print("\n🎉 Demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()