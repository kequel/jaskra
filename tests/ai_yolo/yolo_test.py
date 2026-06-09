import pytest
from ultralytics import YOLO
import numpy as np

model = YOLO('../../ai/yolo/weights/last.pt')

def test_model_inference():
    """
    STABILITY TEST (Smoke Test):
    Verifies if the model correctly handles the inference process 
    on a blank image and returns the expected data format.
    """
    
    # Create a 'dummy' image: a black square of 512x512 pixels
    # np.zeros creates a matrix filled with zeros (representing black color)
    dummy_img = np.zeros((512, 512, 3), dtype=np.uint8)
    
    results = model(dummy_img)
    
    # ASSERTION: Check if the results list is not empty
    assert len(results) > 0, "Error: The model failed to return any result object"
    
    # ASSERTION: Verify if the first result contains the 'boxes' attribute
    # This ensures the model is functioning as an object detector.
    assert hasattr(results[0], 'boxes'), "Error: Result object does not contain detection boxes"

def test_specific_detection():
    """
    ACCURACY TEST (Functional Test):
    Verifies if the model correctly detects an object on a specific, 
    known test image where the target is guaranteed to be present.
    """
    
    # Run detection on a real sample image
    # save=True allows for visual inspection of the results if the test fails
    results = model.predict('1.png', save=True)
    
    # Retrieve the number of detected bounding boxes
    detected_count = len(results[0].boxes)
    
    # ASSERTION: Ensure the model found at least one object
    # If the object is present in '1.png' but not detected, the test will fail.
    assert detected_count > 0, f"Failure: Model failed to detect any objects on 1.png!"

if __name__ == "__main__":
    test_model_inference()
    test_specific_detection()
    print("All tests passed successfully!")