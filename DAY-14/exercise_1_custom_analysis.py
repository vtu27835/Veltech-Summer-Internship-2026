# exercise_1_custom_analysis.py
import cv2
import numpy as np
# Ensure day14_mini_project.py is in the same directory
from day14_mini_project import ObjectAnalysisPipeline

class CustomObjectAnalysis(ObjectAnalysisPipeline):
    """Extended pipeline with custom geometric analysis metrics"""

    def analyze_contours(self):
        """Override to add custom shape descriptors safely"""
        print("\n[Custom] Analyzing contours with additional metrics...")
        
        # 1. Run parent analysis first to populate results and clean contours
        super().analyze_contours()
        
        # 2. Iterate directly over the synchronized data structures
        # We pair the saved contour arrays with their corresponding result dictionaries
        for result, contour in zip(self.results['contours'], self.contours):
            area = result['area']
            perimeter = result['perimeter']
            
            # Calculate Circularity (values close to 1.0 indicate a perfect circle)
            if perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter ** 2)
            else:
                circularity = 0
                
            # Calculate Extent (Object Area / Bounding Box Area)
            # e.g., a perfect square has an extent of 1.0; a circle is ~0.785
            x, y, w, h = result['bbox']
            bbox_area = w * h
            extent = area / bbox_area if bbox_area > 0 else 0
            
            # Inject metrics safely back into the existing dictionary
            result['circularity'] = circularity
            result['extent'] = extent
            
            print(f" Contour {result['id']} ({result['shape']}): "
                  f"circularity={circularity:.3f}, extent={extent:.3f}")
                  
        return self

# -----------------------------------------------------------------
# Execution Block
# -----------------------------------------------------------------
if __name__ == "__main__":
    # Replace with a valid path to an image file for testing
    IMAGE_PATH = "test_image.png" 
    pipeline = CustomObjectAnalysis(IMAGE_PATH, "custom_output")
    pipeline.run()