# day14_mini_project.py
# Week 2 Review - Complete Object Analysis Pipeline
from __future__ import print_function
import numpy as np
import argparse
import cv2
import os
from datetime import datetime
import matplotlib.pyplot as plt

print("=" * 70)
print("DAY 14: WEEK 2 REVIEW & MINI PROJECT")
print("=" * 70)

# -----------------------------------------------------------------
# CLASS: ObjectAnalysisPipeline
# -----------------------------------------------------------------
class ObjectAnalysisPipeline:
    """
    Complete pipeline for object detection and analysis
    """

    def __init__(self, image_path, output_dir="day14_output"):
        self.image_path = image_path
        self.output_dir = output_dir
        self.image = None
        self.original = None
        self.gray = None
        self.binary = None
        self.edges = None
        self.contours = None
        self.results = {}

        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

    def load_image(self):
        """Step 1: Load image from disk"""
        print("\n[Step 1] Loading image...")

        self.original = cv2.imread(self.image_path)

        if self.original is None:
            raise ValueError(f"Could not load image: {self.image_path}")

        self.image = self.original.copy()
        print(f"Loaded: {self.image.shape[1]} x {self.image.shape[0]} pixels")
        return self

    def preprocess(self):
        """Step 2: Convert to grayscale, blur, equalize if needed"""
        print("\n[Step 2] Preprocessing...")

        # Convert to grayscale
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        print(" → Converted to grayscale")

        # Analyze histogram to check contrast
        spread = np.percentile(self.gray, 95) - np.percentile(self.gray, 5)

        if spread < 100:
            print(" → Low contrast detected. Applying histogram equalization...")
            self.gray = cv2.equalizeHist(self.gray)
        else:
            print(" → Contrast is adequate")

        # Apply blur
        self.gray = cv2.GaussianBlur(self.gray, (5, 5), 0)
        print(" → Applied Gaussian blur (5x5)")

        return self

    def threshold(self, method='auto'):
        """Step 3: Apply thresholding"""
        print("\n[Step 3] Applying thresholding...")

        if method == 'simple':
            _, self.binary = cv2.threshold(
                self.gray, 127, 255, cv2.THRESH_BINARY
            )
            print(" → Simple threshold (T=127)")

        elif method == 'adaptive':
            self.binary = cv2.adaptiveThreshold(
                self.gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2
            )
            print(" → Adaptive threshold")

        else:  # auto - use Otsu
            _, self.binary = cv2.threshold(
                self.gray,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            print(" → Otsu's thresholding (auto)")

        # Clean up binary image
        kernel = np.ones((3, 3), np.uint8)

        self.binary = cv2.morphologyEx(
            self.binary,
            cv2.MORPH_CLOSE,
            kernel
        )

        self.binary = cv2.morphologyEx(
            self.binary,
            cv2.MORPH_OPEN,
            kernel
        )

        print(" → Applied morphological cleaning")
        return self

    def detect_edges(self, low=50, high=150):
        """Step 4: Edge detection"""
        print("\n[Step 4] Detecting edges...")

        self.edges = cv2.Canny(self.gray, low, high)
        print(f" → Canny edge detection (low={low}, high={high})")
        return self

    def find_contours(self, min_area=50, retrieval_mode=cv2.RETR_EXTERNAL):
        """Step 5: Find and filter contours"""
        print("\n[Step 5] Finding contours...")

        contours_info = cv2.findContours(
            self.binary,
            retrieval_mode,
            cv2.CHAIN_APPROX_SIMPLE
        )

        self.contours = contours_info[-2]
        print(f" → Found {len(self.contours)} contours")

        # Filter by area
        self.contours = [
            c for c in self.contours
            if cv2.contourArea(c) > min_area
        ]

        print(f" → Filtered to {len(self.contours)} contours (area > {min_area})")
        return self

    def analyze_contours(self):
        """Step 6: Analyze each contour"""
        print("\n[Step 6] Analyzing contours...")

        self.results['contours'] = []

        for i, contour in enumerate(self.contours):
            # Basic properties
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            x, y, w, h = cv2.boundingRect(contour)

            # Center
            M = cv2.moments(contour)

            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2

            # Shape classification
            approx = cv2.approxPolyDP(
                contour,
                0.02 * perimeter,
                True
            )

            vertices = len(approx)

            if vertices == 3:
                shape = "Triangle"
            elif vertices == 4:
                aspect_ratio = w / h if h > 0 else 1
                if 0.9 <= aspect_ratio <= 1.1:
                    shape = "Square"
                else:
                    shape = "Rectangle"
            elif vertices == 5:
                shape = "Pentagon"
            elif vertices == 6:
                shape = "Hexagon"
            elif vertices > 6:
                shape = "Circle"
            else:
                shape = "Unknown"

            result = {
                'id': i + 1,
                'area': area,
                'perimeter': perimeter,
                'bbox': (x, y, w, h),
                'center': (cx, cy),
                'vertices': vertices,
                'shape': shape
            }

            self.results['contours'].append(result)

            print(
                f"Contour {i+1}: {shape}, "
                f"area={area:.0f}, "
                f"bbox={w}x{h}, "
                f"center=({cx},{cy})"
            )

        return self

    def visualize_results(self):
        """Step 7: Draw contours and labels on image"""
        print("\n[Step 7] Visualizing results...")
        for result in self.results['contours']:
            x, y, w, h = result['bbox']
            cx, cy = result['center']
            
            # Draw bounding box and center
            cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(self.image, (cx, cy), 4, (0, 0, 255), -1)
            
            # Label shape
            cv2.putText(self.image, f"#{result['id']}: {result['shape']}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
        cv2.imwrite(f"{self.output_dir}/01_visualized_results.png", self.image)
        print(" → Saved: 01_visualized_results.png")
        return self

    def generate_report(self):
        """Step 8: Generate final written report"""
        print("\n[Step 8] Generating report...")
        report = f"""=========================================================================
OBJECT ANALYSIS REPORT
=========================================================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Image: {self.image_path}
Dimensions: {self.image.shape[1]} x {self.image.shape[0]} pixels
=========================================================================
PIPELINE STEPS
=========================================================================
✓ Load image
✓ Preprocess (grayscale, blur, histogram equalization)
✓ Threshold (Otsu's method)
✓ Edge detection (Canny)
✓ Find contours
✓ Analyze contours
✓ Visualize results
=========================================================================
CONTOUR ANALYSIS
=========================================================================
Total contours found: {len(self.contours)}
"""

        # Add detailed contour info
        for result in self.results['contours']:
            report += f"""
Contour #{result['id']}
 Shape: {result['shape']}
 Area: {result['area']:.0f} pixels
 Perimeter: {result['perimeter']:.1f} pixels
 Bounding Box: {result['bbox'][0]}, {result['bbox'][1]} - {result['bbox'][2]}x{result['bbox'][3]}
 Center: ({result['center'][0]}, {result['center'][1]})
 Vertices: {result['vertices']}
"""

        # Shape statistics
        report += """=========================================================================
SHAPE STATISTICS
=========================================================================
"""
        shape_counts = {}
        for result in self.results['contours']:
            shape = result['shape']
            shape_counts[shape] = shape_counts.get(shape, 0) + 1

        for shape, count in shape_counts.items():
            report += f" {shape}: {count}\n"

        # Area statistics
        if self.results['contours']:
            areas = [r['area'] for r in self.results['contours']]
            report += f"""=========================================================================
AREA STATISTICS
=========================================================================
 Min area: {min(areas):.0f}
 Max area: {max(areas):.0f}
 Average area: {np.mean(areas):.0f}
 Total area: {sum(areas):.0f}
"""

        report += """=========================================================================
END OF REPORT
=========================================================================
"""

        # Save report
        #  NEW WAY (Safe for all platforms)
        with open(f"{self.output_dir}/REPORT.txt", "w", encoding="utf-8") as f:
            f.write(report)

        print(" → Saved: REPORT.txt")
        print(report)

        return self

    def run(self):
        """Execute the complete pipeline"""
        print("\n" + "=" * 70)
        print("RUNNING OBJECT ANALYSIS PIPELINE")
        print("=" * 70)

        try:
            self.load_image()
            self.preprocess()
            self.threshold(method='auto')
            self.detect_edges()
            self.find_contours(min_area=50)
            self.analyze_contours()
            self.visualize_results()
            self.generate_report()

            print("\n" + "=" * 70)
            print("PIPELINE COMPLETE!")
            print("=" * 70)
            print(f"\nAll outputs saved to: {self.output_dir}/")

        except Exception as e:
            print(f"\nERROR: {e}")

        return self

# -----------------------------------------------------------------
# SECTION 1: RUN THE PIPELINE
# -----------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("INITIALIZING OBJECT ANALYSIS PIPELINE")
    print("=" * 70)
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image to analyze")
    ap.add_argument("-o", "--output", default="day14_output", help="Output directory (default: day14_output)")
    args = vars(ap.parse_args())
    
    # Create and run pipeline
    pipeline = ObjectAnalysisPipeline(args["image"], args["output"])
    pipeline.run()
    
    # -----------------------------------------------------------------
    # SECTION 2: ADDITIONAL ANALYSIS WITH MATPLOTLIB
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ADDITIONAL ANALYSIS WITH MATPLOTLIB")
    print("=" * 70)
    
    # Create histogram comparison
    if pipeline.gray is not None and pipeline.binary is not None:
        print("\nGenerating histogram analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original
        axes[0, 0].imshow(cv2.cvtColor(pipeline.original, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        # Grayscale histogram
        axes[0, 1].hist(pipeline.gray.ravel(), bins=256, color='black', alpha=0.7)
        axes[0, 1].set_title("Grayscale Histogram")
        axes[0, 1].set_xlabel("Pixel Intensity")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].grid(True, alpha=0.3)
        
        # Binary
        axes[1, 0].imshow(pipeline.binary, cmap='gray')
        axes[1, 0].set_title("Binary Image (Thresholded)")
        axes[1, 0].axis('off')
        
        # Binary histogram
        axes[1, 1].hist(pipeline.binary.ravel(), bins=2, color='black', alpha=0.7)
        axes[1, 1].set_title("Binary Histogram (0 and 255)")
        axes[1, 1].set_xlabel("Pixel Value")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].set_xticks([0, 255])
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{args['output']}/03_histogram_analysis.png")
        plt.close()
        print(" → Saved: 03_histogram_analysis.png")
        
    # -----------------------------------------------------------------
    # SECTION 3: CONCLUSION
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DAY 14 - WEEK 2 MINI PROJECT COMPLETE!")
    print("=" * 70)
    print("\nSkills demonstrated in this project:")
    print(" ✓ Image loading and preprocessing")
    print(" ✓ Color space conversion (grayscale)")
    print(" ✓ Histogram analysis")
    print(" ✓ Histogram equalization")
    print(" ✓ Gaussian blurring")
    print(" ✓ Thresholding (Otsu's method)")
    print(" ✓ Morphological operations")
    print(" ✓ Edge detection (Canny)")
    print(" ✓ Contour detection and analysis")
    print(" ✓ Shape classification")
    print(" ✓ Object extraction and labeling")
    print(" ✓ Report generation")
    print("\n" + "=" * 70)
    print("WEEK 2 COMPLETE! READY FOR WEEK 3!")
    print("=" * 70)
    cv2.destroyAllWindows()