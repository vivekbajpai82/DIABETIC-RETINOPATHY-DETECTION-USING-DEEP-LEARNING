# Diabetic Retinopathy Detection using Deep Learning 👁️

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Click%20Here-success?style=for-the-badge)](https://diabeticretinopathy-detection.netlify.app/)

An advanced full-stack web application designed to detect and classify Diabetic Retinopathy from retinal fundus images using Deep Learning.

## 🌐 Live Demo
Experience the live application here: **[Diabetic Retinopathy Detection App](https://diabeticretinopathy-detection.netlify.app/)**

## 🚀 Project Overview
This project leverages deep learning and computer vision techniques to analyze retinal images and identify the presence and severity of Diabetic Retinopathy. The model has been developed and trained using the comprehensive **EyePACS 2015 dataset** to ensure robust feature extraction and high accuracy. 

The application is structured with a modern API Gateway architecture, separating the client-facing frontend from the heavy machine learning inference backend, ensuring smooth performance even on cloud deployment platforms.

## 🛠️ Tech Stack
* **Deep Learning / ML:** Python, PyTorch, OpenCV
* **Backend API:** FastApi
* **Frontend:** React.js 
* **Deployment:** Netlify (Frontend) & Render (Backend) 

## ✨ Key Features
* **Accurate Detection:** Classifies retinal images into stages of Diabetic Retinopathy.
* **API Gateway Architecture:** Efficiently manages and routes requests between the frontend interface and the ML inference server.
* **Highly Optimized Backend:** Implements RAM optimization with manual garbage collection (`gc`) and on-demand model loading to run efficiently on cloud platforms like Render.
* **Seamless Full-Stack Integration:** Provides an intuitive web interface for users to upload images and receive real-time predictions.

## 📁 Repository Structure
* `/frontend` - Contains the complete user interface code.
* `/backend` - Contains the API gateway and the deep learning model serving logic.
* `netlify.toml` - Configuration file for frontend deployment on Netlify.
* `render.yaml` - Infrastructure-as-Code configuration for backend deployment on Render.

## ⚙️ Local Setup & Installation

### Prerequisites
* Node.js (v14+)
* Python (3.8+)

### Steps to Run Locally
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/vivekbajpai82/DIABETIC-RETINOPATHY-DETECTION-USING-DEEP-LEARNING.git](https://github.com/vivekbajpai82/DIABETIC-RETINOPATHY-DETECTION-USING-DEEP-LEARNING.git)
    cd DIABETIC-RETINOPATHY-DETECTION-USING-DEEP-LEARNING
    ```

2.  **Setup the Backend:**
    ```bash
    cd backend
    # Install Node dependencies
    npm install 
    # (If using a separate Python env for the model, install requirements)
    # pip install -r requirements.txt
    
    # Start the backend server
    npm run dev 
    ```

3.  **Setup the Frontend:**
    ```bash
    cd ../frontend
    npm install
    npm start
    ```

## 📄 License
This project is licensed under the [MIT License](LICENSE).
