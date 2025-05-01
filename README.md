# 🌿 Healing_Bloom_Product_RecommendingEngine

This is the backend system for the **Product Recommender** in the Healing Bloom platform. It uses **machine learning models** and **facial analysis** to detect acne-related conditions and recommend personalized skincare products.

---

## 🔧 Tech Stack

- 🐍 **Python 3.x**
- 🌐 **Flask** - lightweight web framework
- 📦 **Flask-RESTful** - for building REST APIs
- 🧠 **TensorFlow / Keras** - for loading acne detection models (EfficientNet)
- 🗄️ **PostgreSQL (PSQL)** - for storing user data and product information
- 🔌 **SQLAlchemy** - ORM for database interactions
- 🎨 **OpenCV / Pillow** - for face and skin image preprocessing

---

## 🧠 Functionality

### 🔍 Acne Detection
- Uses a pre-trained **EfficientNet** model to analyze face images.
- Detects presence, type, and severity of acne conditions.

### 🛍️ Product Recommendation
- Based on acne type, skin tone, and severity.
- Pulls relevant skincare products from the PostgreSQL database.
- Returns top personalized recommendations via the API.

---

## 📁 Project Structure

\`\`\`
product_recommender/
│
├── app.py                      # Main Flask application
├── model/                      # Contains ML model files (.h5 or .keras)
├── utils/                      # Preprocessing and helper functions
├── database/                   # DB models and queries
├── requirements.txt
└── README.md
\`\`\`

---

## 🚀 How to Run

### 🔧 1. Install Requirements
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 🔁 2. Run Flask Server
\`\`\`bash
python app.py
\`\`\`

---

## 🔗 API Endpoints

- \`POST /predict\`  
  Uploads face image and returns acne prediction + product list.

- \`GET /products/<skin_condition>\`  
  Returns recommended products for a given condition.

---

## 🛠️ Environment Variables

Create a \`.env\` file with:
\`\`\`
DB_URI=postgresql://username:password@localhost:5432/healingbloom
MODEL_PATH=./model/acne_model.keras
\`\`\`

---

## ⚠️ Notes

- Model files are not included due to size limits. Host separately via S3, Google Drive, or use Git LFS.
- Ensure database is set up and seeded with product data before running.
" > README.md
