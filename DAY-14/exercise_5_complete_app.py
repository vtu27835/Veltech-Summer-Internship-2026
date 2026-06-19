# exercise_5_complete_app.py
import cv2
import numpy as np
import os
from datetime import datetime

class CompleteImageAnalysisApp:
    """Complete application with dynamic grid preview matrices"""
    
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image from target path: {image_path}")
            
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.original = self.image.copy()
        self.steps = []
        
    def add_step(self, name, image):
        """Add a step to the pipeline"""
        self.steps.append((name, image))
        return self
        
    def process(self):
        """Run the complete processing pipeline"""
        print("Processing Image Pipeline Stages...")
        
        # Step 1: Original
        self.add_step("Original", self.image.copy())
        
        # Step 2: Grayscale
        self.add_step("Grayscale", cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR))
        
        # Step 3: Blur
        blurred = cv2.GaussianBlur(self.gray, (5, 5), 0)
        self.add_step("Blurred", cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR))
        
        # Step 4: Histogram Equalization
        equalized = cv2.equalizeHist(blurred)
        self.add_step("Equalized", cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR))
        
        # Step 5: Threshold
        # Using INVERTED Otsu thresholding to guarantee foreground masks match contour rules (white)
        _, binary = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        self.add_step("Binary", cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))
        
        # Step 6: Edges
        edges = cv2.Canny(equalized, 50, 150)
        self.add_step("Edges", cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
        
        # Step 7: Contours
        # FIX: Cross-version support for modern OpenCV unpacking structures
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_vis = self.image.copy()
        cv2.drawContours(contour_vis, contours, -1, (0, 255, 0), 2)
        self.add_step("Contours", contour_vis)
        
        # Step 8: Analysis
        analysis_vis = self.image.copy()
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) < 50:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(analysis_vis, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(analysis_vis, f"#{i+1}", (x+5, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        self.add_step("Analysis", analysis_vis)
        
        return self
        
    def show_grid(self):
        """Display all tracked timeline steps in a perfectly padded grid layout"""
        n = len(self.steps)
        cols = 4
        rows = (n + cols - 1) // cols
        
        thumb_w, thumb_h = 200, 150
        pad = 10
        
        # FIX: Explicit matrix dimension sizing corresponding with inner loop arithmetic steps
        grid_h = rows * thumb_h + (rows + 1) * pad
        grid_w = cols * thumb_w + (cols + 1) * pad
        grid = np.zeros((grid_h, grid_w, 3), dtype="uint8")
        
        for i, (name, img) in enumerate(self.steps):
            row = i // cols
            col = i % cols
            
            # Stride math mapping coordinates safely inside canvas frame bounds
            x = pad + col * (thumb_w + pad)
            y = pad + row * (thumb_h + pad)
            
            img_resized = cv2.resize(img, (thumb_w, thumb_h))
            grid[y:y+thumb_h, x:x+thumb_w] = img_resized
            
            # Label overlay rendering
            # Dark backing shadow behind tracking labels to maximize contrast legibility
            cv2.putText(grid, name, (x+7, y+22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
            cv2.putText(grid, name, (x+5, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
        return grid
        
    def generate_summary(self):
        """Generate formatted final summary execution report"""
        report = f"""================================================================================
IMAGE ANALYSIS SUMMARY
================================================================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Image Dimensions: {self.image.shape[1]} x {self.image.shape[0]}

================================================================================
PROCESSING TIMELINE LOGS
================================================================================
"""
        for i, (name, _) in enumerate(self.steps):
            report += f"  Stage {i+1:<2}: {name}\n"
            
        report += f"""================================================================================
STATUS: COMPLETED | Total Logged Pipeline Blocks: {len(self.steps)}
================================================================================
"""
        return report

# -----------------------------------------------------------------
# RUN APP LAYER
# -----------------------------------------------------------------
if __name__ == "__main__":
    # Ensure you replace this with a valid filename for testing
    IMAGE_PATH = "test_image.png"
    
    if not os.path.exists(IMAGE_PATH):
        # Create a dummy image if test_image.png doesn't exist so it runs out-of-the-box
        dummy_img = np.zeros((400, 400, 3), dtype="uint8")
        cv2.circle(dummy_img, (100, 100), 40, (0, 0, 255), -1)
        cv2.rectangle(dummy_img, (220, 220), (320, 320), (0, 255, 0), -1)
        cv2.imwrite(IMAGE_PATH, dummy_img)
        print(f"Generated a mockup playground asset at: '{IMAGE_PATH}'")

    app = CompleteImageAnalysisApp(IMAGE_PATH)
    app.process()
    
    # Extract structural layout grid map matrix
    grid_preview = app.show_grid()
    
    # Terminal readout logs
    print(app.generate_summary())
    
    # Save output visualization
    cv2.imwrite("complete_pipeline.png", grid_preview)
    print(" → Processed panel map successfully written to: 'complete_pipeline.png'")
    
    # Window Render loop
    cv2.imshow("Complete Analysis Pipeline Matrix Layout", grid_preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()