# face_detector.py
# Face Detector class using Haar cascades
import cv2

class FaceDetector: 
    """ A class for detecting faces in images using Haar cascades """
    
    def __init__(self, faceCascadePath): 
        """ Initialize the face detector with a cascade classifier
        Parameters: 
        - faceCascadePath: path to the XML cascade file 
        """ 
        # Load the cascade classifier 
        self.faceCascade = cv2.CascadeClassifier(faceCascadePath)
        
        # Check if cascade was loaded properly 
        if self.faceCascade.empty(): 
            raise ValueError(f"Could not load cascade from: {faceCascadePath}") 
        else: 
            print("✓ Face cascade loaded successfully")
            
    def detect(self, image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), maxSize=None): 
        """ Detect faces in an image
        Parameters: 
        - image: input image (grayscale or color) 
        - scaleFactor: how much image is reduced at each scale 
        - minNeighbors: minimum neighbors for face confirmation 
        - minSize: minimum window size 
        - maxSize: maximum window size
        Returns: 
        - list of rectangles (x, y, w, h) for each face 
        """ 
        # Convert to grayscale if needed 
        if len(image.shape) == 3: 
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        else: 
            gray = image
            
        # Perform face detection 
        faces = self.faceCascade.detectMultiScale( 
            gray, 
            scaleFactor=scaleFactor, 
            minNeighbors=minNeighbors, 
            minSize=minSize, 
            maxSize=maxSize, 
            flags=cv2.CASCADE_SCALE_IMAGE 
        )
        
        return faces
        
    def draw_boxes(self, image, faces, color=(0, 255, 0), thickness=2): 
        """ Draw bounding boxes around detected faces
        Parameters: 
        - image: image to draw on 
        - faces: list of face rectangles 
        - color: RGB color tuple 
        - thickness: line thickness
        Returns: 
        - image with bounding boxes drawn 
        """ 
        result = image.copy()
        
        for (x, y, w, h) in faces: 
            cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
            
        return result