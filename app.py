import os
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from flask_cors import CORS
from models.skin_tone.skin_tone_knn import identify_skin_tone
from models.recommender.rec import recs_essentials, makeup_recommendation
from PIL import Image
import tensorflow as tf
import tf_keras as k3
import numpy as np
import base64
from io import BytesIO
import traceback

# Initialize Flask app
app = Flask(__name__)
api = Api(app)

# Configure CORS for Flutter and web
CORS(app, resources={
    r"/api/*": {
        "origins": ["*"],
        "methods": ["PUT", "POST", "GET"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Global variables for models
model1 = None
model2 = None

# Load models once at startup
@app.before_first_request
def load_models():
    global model1, model2
    try:
        model1 = k3.models.load_model('./models/skin_model')
        model2 = k3.models.load_model('./models/acne_model')
        print("Both models loaded successfully")
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise e
    
@app.route('/api/test-connection', methods=['POST'])
def test_connection():
    try:
        data = request.get_json()
        if data and 'message' in data:
            print(f"\n\n[SUCCESS] Received message: {data['message']}\n")
            return jsonify({"status": "success", "received": data['message']})
        return jsonify({"error": "No message received"}), 400
    except Exception as e:
        print(f"\n[ERROR] Connection test failed: {str(e)}\n")
        return jsonify({"error": str(e)}), 500

# Helper functions
def load_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_tensor = tf.keras.preprocessing.image.img_to_array(img)
    return np.expand_dims(img_tensor, axis=0) / 255.0

def predict_model(model, class_names, img_tensor):
    pred = model.predict(img_tensor)
    if len(pred[0]) > 1:
        return class_names[tf.argmax(pred[0])]
    return class_names[int(tf.round(pred[0]))]

# API Endpoints
class SkinMetrics(Resource):
    def post(self):
        try:
            # Validate input
            if 'file' not in request.json:
                print("[ERROR] No image provided in the request")
                return {"error": "No image provided"}, 400
            
            print("[INFO] Image received for analysis")

            # Process image
            file_data = request.json['file']
            header, data = file_data.split(',', 1)
            im = Image.open(BytesIO(base64.b64decode(data)))
            
            # Save temporary image
            os.makedirs('./temp', exist_ok=True)
            img_path = './temp/analysis_image.jpg'
            im.save(img_path)
            print(f"[INFO] Image saved at {img_path}")

            # Make predictions
            img_tensor = load_image(img_path)
            print("[INFO] Image successfully preprocessed")

            skin_pred = predict_model(model1, ['Dry', 'Normal', 'Oily'], img_tensor)
            acne_pred = predict_model(model2, ['Low', 'Moderate', 'Severe'], img_tensor)
            tone_pred = identify_skin_tone(img_path, 'models/skin_tone/skin_tone_dataset.csv')
            print(f"[INFO] Predictions - Skin Type: {skin_pred}, Acne Severity: {acne_pred}, Skin Tone: {tone_pred}")

            # Cleanup
            if os.path.exists(img_path):
                os.remove(img_path)
            
            print("[INFO] Sending response")
            return {
                "skin_type": skin_pred,
                "skin_tone": str(tone_pred),
                "acne_severity": acne_pred
            }, 200
            
        except Exception as e:
            print("[ERROR] Exception occurred during processing:")
            traceback.print_exc()
            return {"error": str(e)}, 500

class Recommendation(Resource):
    def post(self):
        try:
            data = request.json
            required_fields = ['features', 'tone', 'skin_type']
            
            if not all(field in data for field in required_fields):
                return {"error": "Missing required fields"}, 400

            # Convert features to proper format
            feature_vector = [int(data['features'][f]) for f in [
                'normal', 'dry', 'oily', 'combination', 'acne', 'sensitive',
                'fine lines', 'wrinkles', 'redness', 'dull', 'pore',
                'pigmentation', 'blackheads', 'whiteheads', 'blemishes',
                'dark circles', 'eye bags', 'dark spots'
            ]]
   
            # Get recommendations
            general = recs_essentials(feature_vector, None)
            makeup = makeup_recommendation(
                data['tone'], 
                data['skin_type'].lower()
            )
            
            return {
                "general_recommendations": general,
                "makeup_recommendations": makeup
            }, 200
            
        except Exception as e:
            traceback.print_exc()
            return {"error": str(e)}, 500
        


# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify(error=str(e)), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify(error="Internal server error"), 500

# Register API endpoints
api.add_resource(SkinMetrics, "/api/analyze-skin")
api.add_resource(Recommendation, "/api/get-recommendations")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)