import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from PIL import Image as PILImage



class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename


    def is_valid_ct_scan(self, img_path):
        """
        CT scans have specific properties:
        - Mostly grayscale (low color saturation)
        - Specific aspect ratios
        """
        try:
            img = PILImage.open(img_path).convert('RGB')
            img_array = np.array(img, dtype=float)

            # Check if image is grayscale-like
            # In CT scans R≈G≈B for most pixels
            r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
            
            # Color deviation — CT scans have very low color variance
            rg_diff = np.mean(np.abs(r - g))
            rb_diff = np.mean(np.abs(r - b))
            gb_diff = np.mean(np.abs(g - b))
            color_deviation = (rg_diff + rb_diff + gb_diff) / 3

            print(f"Color deviation: {color_deviation}")

            # If color deviation > 15, it's likely a colorful natural image
            if color_deviation > 15:
                return False
            return True
        except Exception as e:
            print(f"Validation error: {e}")
            return False


    def predict(self):
        # load model
        model = load_model(os.path.join("model", "model.h5"), compile=False)

        #  Step 1 — Check if image looks like a CT scan
        if not self.is_valid_ct_scan(self.filename):
            return [{"image": "error", 
                     "message": "Please upload a valid Kidney CT Scan image"}]

        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        raw_output = model.predict(test_image)
        confidence = float(np.max(raw_output))
        result = np.argmax(raw_output, axis=1)

        print(f"Confidence: {confidence}, Result: {result}")

        #  Step 2 — Confidence check as second layer
        if confidence < 0.85:
            return [{"image": "error",
                     "message": "Please upload a valid Kidney CT Scan image"}]

        if result[0] == 1:
            prediction = 'Tumor'
        elif result[0] == 0:
            prediction = 'Normal'
        else:
            prediction = 'Unknown'

        return [{"image": prediction, "confidence": round(confidence * 100, 2)}]