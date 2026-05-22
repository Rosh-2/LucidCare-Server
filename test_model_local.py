import os
import sys
import numpy as np
import cv2
import torch

# Add parent directory to path to allow importing from models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model import MedicalDiagnosticSystem

def test_system():
    print("Testing MedicalDiagnosticSystem...")
    print(f"Torch version: {torch.__version__}")
    
    # 1. Initialization
    print("\n[1] Testing Initialization...")
    system = MedicalDiagnosticSystem(groq_api_key="dummy_key")
    if system.model is None:
        print("CRITICAL: Model failed to load.")
        return
    else:
        print("Success: Model loaded.")
        try:
             print(f"Model classifier: {system.model.classifier}")
        except:
             print("Could not print classifier structure")

    # 2. Mock Analysis
    print("\n[2] Testing Analysis with Dummy Image...")
    dummy_img_path = "temp_test_xray.png"
    dummy_img = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.rectangle(dummy_img, (100, 100), (300, 300), (255, 255, 255), -1) 
    cv2.imwrite(dummy_img_path, dummy_img)
    
    try:
        result = system.analyze_image(dummy_img_path)
        print("Analysis Result Keys:", result.keys())
        if "error" in result:
             print(f"Analysis failed with error: {result['error']}")
        else:
             print("Analysis successful.")
    except Exception as e:
        print(f"Analysis raised exception: {e}")
    finally:
        if os.path.exists(dummy_img_path):
            os.remove(dummy_img_path)

    print("\nTest Complete.")

if __name__ == "__main__":
    test_system()
