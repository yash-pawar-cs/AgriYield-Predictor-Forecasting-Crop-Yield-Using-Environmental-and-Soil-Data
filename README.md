🌾 AgriYield Predictor

AI-Powered Crop Yield Prediction System

🚀 Overview

AgriYield Predictor is a machine learning-based web application that predicts crop yield and total production based on real-world agricultural inputs such as rainfall, temperature, humidity, soil type, and nutrient values.
This system is designed to assist farmers, researchers, and agricultural planners in making data-driven decisions.


🎯 Features

- 🌱 Crop Yield Prediction (tons/hectare)
- 📊 Total Production Estimation
- 🤖 AI-Based Insights & Explanation
- 🔄 Auto NPK Estimation (if unknown)
- 📄 Downloadable PDF Report
- 🎨 Modern Glassmorphism UI Design
- ⚡ Real-time Prediction using Flask


🧠 Machine Learning

- Model trained using historical agricultural datasets

- Input features:
  
  - State
  - Crop
  - Rainfall
  - Temperature
  - Humidity
  - Soil Type
  - N, P, K values
  - pH level

- Output:
  
  - Predicted Yield
  - Estimated Production


🛠️ Tech Stack

- Backend: Flask (Python)
- Frontend: HTML, CSS (Glassmorphism UI)
- ML Model: Scikit-learn
- Data Processing: Pandas
- Model Storage: Joblib
- PDF Generation: ReportLab


📂 Project Structure

AgriYield-Predictor/
│── app.py
│── requirements.txt
│── models/
│── data/
│── templates/
│── static/
│── notebooks/


⚙️ Installation & Setup

1. Clone Repository

git clone https://github.com/your-username/AgriYield-Predictor.git
cd AgriYield-Predictor

2. Install Dependencies

pip install -r requirements.txt

3. Run Application

python app.py

4. Open Browser

http://127.0.0.1:5000


🧪 Testing

The system has been tested using real-world agricultural inputs including:

- Rice (High rainfall region)
- Wheat (Moderate climate)
- Maize (Balanced conditions)

Edge cases such as low rainfall and missing NPK values were also tested.


📄 PDF Report

Users can download a detailed report containing:

- Input data
- Predicted yield
- Production estimate
- AI-based insights


