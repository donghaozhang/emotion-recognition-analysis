#!/usr/bin/env python3
"""
GLM-4.5V Emotion Recognition Script

This script uses the GLM-4.5V model via OpenRouter API to analyze emotions
in facial images from the EXPW dataset.

Requirements:
- openai library for API calls
- PIL for image processing
- base64 for image encoding
- Environment variable OPENROUTER_API_KEY
"""

import os
import base64
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
import random

try:
    from openai import OpenAI
    from PIL import Image
    from dotenv import load_dotenv
except ImportError:
    print("Missing required packages. Install with:")
    print("pip install openai pillow python-dotenv")
    exit(1)

# Load environment variables from .env file
load_dotenv()


class GLMEmotionAnalyzer:
    """Emotion analyzer using GLM-4.5V model via OpenRouter"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the analyzer with OpenRouter API key"""
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OpenRouter API key required. Set OPENROUTER_API_KEY environment variable or pass api_key parameter")
        
        # Initialize OpenAI client with OpenRouter endpoint
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        
        self.model_name = "z-ai/glm-4.5v"
        self.emotion_categories = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
        
    def encode_image_to_base64(self, image_path: Path) -> str:
        """Convert image to base64 encoding for API"""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize if too large (optional optimization)
                max_size = (1024, 1024)
                if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # Save to bytes and encode
                import io
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=90)
                image_bytes = buffer.getvalue()
                
                return base64.b64encode(image_bytes).decode('utf-8')
                
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None
    
    def analyze_emotion(self, image_path: Path) -> Dict:
        """Analyze emotion in a single image using GLM-4.5V"""
        print(f"🔍 Analyzing: {image_path.name}")
        
        # Encode image
        base64_image = self.encode_image_to_base64(image_path)
        if not base64_image:
            return {"error": "Failed to encode image", "image_path": str(image_path)}
        
        try:
            # Prepare the prompt for emotion detection
            prompt = """Look at the person's face in this image and identify their emotion.

Choose exactly ONE emotion from this list:
- angry
- disgust  
- fear
- happy
- sad
- surprise
- neutral

Answer with only the emotion word (lowercase). Do not include any other text or explanation."""

            # Make API call
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=50,
                temperature=0.3
            )
            
            # Extract prediction
            raw_response = response.choices[0].message.content or ""
            
            # Handle GLM-4.5V specific response format (may contain special tokens)
            prediction = raw_response.strip()
            
            # Remove GLM-specific formatting tokens
            prediction = prediction.replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "")
            prediction = prediction.replace("\\n", "").replace("\n", "").strip().lower()
            
            # Clean up prediction - remove extra text and punctuation
            prediction = prediction.replace(".", "").replace(",", "").replace("!", "").strip()
            
            # Validate prediction is in expected categories
            if prediction not in self.emotion_categories:
                # Try to extract valid emotion from response
                found_emotion = None
                for emotion in self.emotion_categories:
                    if emotion in prediction.lower():
                        found_emotion = emotion
                        break
                
                if found_emotion:
                    prediction = found_emotion
                else:
                    prediction = "unknown"
                    if raw_response:  # Only show warning if there was actually a response
                        print(f"   ⚠️  Unexpected response: '{raw_response}' -> classified as unknown")
            
            result = {
                "image_path": str(image_path),
                "image_name": image_path.name,
                "predicted_emotion": prediction,
                "raw_response": raw_response,
                "tokens_used": response.usage.total_tokens if response.usage else 0,
                "model": self.model_name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            print(f"   ✓ Emotion detected: {prediction}")
            return result
            
        except Exception as e:
            error_result = {
                "error": str(e),
                "image_path": str(image_path),
                "image_name": image_path.name,
                "model": self.model_name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            print(f"   ✗ Error: {e}")
            return error_result
    
    def analyze_batch(self, image_folder: Path, num_images: int = 20) -> List[Dict]:
        """Analyze emotions in a batch of images"""
        print(f"🚀 Starting GLM-4.5V emotion analysis")
        print(f"📁 Image folder: {image_folder}")
        print(f"🎯 Target images: {num_images}")
        print("-" * 60)
        
        # Get list of image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        image_files = [
            f for f in image_folder.iterdir() 
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            print(f"❌ No image files found in {image_folder}")
            return []
        
        print(f"📷 Found {len(image_files)} image files")
        
        # Randomly select images if more than requested
        if len(image_files) > num_images:
            selected_images = random.sample(image_files, num_images)
            print(f"🎲 Randomly selected {num_images} images for analysis")
        else:
            selected_images = image_files[:num_images]
            print(f"📝 Analyzing all {len(selected_images)} available images")
        
        # Sort for consistent ordering
        selected_images.sort(key=lambda x: x.name)
        
        results = []
        total_tokens = 0
        
        for i, image_path in enumerate(selected_images, 1):
            print(f"\n[{i}/{len(selected_images)}] ", end="")
            
            result = self.analyze_emotion(image_path)
            results.append(result)
            
            if 'tokens_used' in result:
                total_tokens += result['tokens_used']
            
            # Rate limiting - small delay between requests
            if i < len(selected_images):
                time.sleep(1)
        
        print(f"\n" + "="*60)
        print(f"✅ Analysis Complete!")
        print(f"📊 Results: {len(results)} images processed")
        print(f"🪙 Total tokens used: {total_tokens:,}")
        
        # Summary statistics
        successful_predictions = [r for r in results if 'predicted_emotion' in r]
        error_count = len(results) - len(successful_predictions)
        
        if successful_predictions:
            emotion_counts = {}
            for result in successful_predictions:
                emotion = result['predicted_emotion']
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            print(f"📈 Emotion distribution:")
            for emotion, count in sorted(emotion_counts.items()):
                percentage = (count / len(successful_predictions)) * 100
                print(f"   • {emotion.capitalize()}: {count} ({percentage:.1f}%)")
        
        if error_count > 0:
            print(f"⚠️  Errors: {error_count} images failed to process")
        
        return results
    
    def save_results(self, results: List[Dict], output_file: str = "glm_emotion_results.json"):
        """Save analysis results to JSON file"""
        output_path = Path(output_file)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Results saved to: {output_path.absolute()}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            return False


def main():
    """Main function to run GLM-4.5V emotion analysis"""
    
    # Configuration
    IMAGE_FOLDER = Path(r"C:\Users\zdhpe\Desktop\emotion\dfew_128\New folder")
    NUM_IMAGES = 20
    OUTPUT_FILE = "glm_emotion_results.json"
    
    print("🎭 GLM-4.5V Emotion Recognition Analyzer")
    print("="*60)
    
    # Check if image folder exists
    if not IMAGE_FOLDER.exists():
        print(f"❌ Error: Image folder not found: {IMAGE_FOLDER}")
        print("Please check the path and try again.")
        return
    
    # Check for API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("❌ Error: OPENROUTER_API_KEY environment variable not set")
        print("\nTo set the API key:")
        print("Windows: set OPENROUTER_API_KEY=your_api_key_here")
        print("Linux/Mac: export OPENROUTER_API_KEY=your_api_key_here")
        print("\nOr get your API key from: https://openrouter.ai/keys")
        return
    
    try:
        # Initialize analyzer
        analyzer = GLMEmotionAnalyzer(api_key)
        
        # Run batch analysis
        results = analyzer.analyze_batch(IMAGE_FOLDER, NUM_IMAGES)
        
        if results:
            # Save results
            analyzer.save_results(results, OUTPUT_FILE)
            
            print(f"\n🎉 Analysis completed successfully!")
            print(f"📄 View results in: {OUTPUT_FILE}")
        else:
            print(f"\n❌ No results to save")
            
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        print("Please check your API key and internet connection.")


if __name__ == "__main__":
    main()