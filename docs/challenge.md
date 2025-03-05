# **Software Engineer (ML & LLMs) Challenge - README**

## **General Description**

This repository contains the solution to the **Software Engineer (ML & LLMs) Challenge**, where a complete machine learning pipeline was implemented to predict flight delays at SCL airport.

All challenge requirements were met, including:
- Transcription of the model from the Jupyter Notebook into a Python script.
- Implementation of an API with **FastAPI** to expose the model.
- Deployment of the model in **Google Cloud Platform (GCP)** using **Cloud Functions**.
- Use of **GitHub Actions** for continuous integration and deployment (CI/CD).

Additionally, the **`dev-all-pipeline`** branch explored an alternative approach that expands on the challenge. However, due to time constraints, it was not possible to achieve the desired level of quality in terms of organization, experimentation, and optimization.

---

## **Part 1: Choosing the Best Model**

### **Why is XGBoost better than Logistic Regression?**

#### **1. Better Performance on Non-Linear Data**  
XGBoost uses decision trees in boosting, allowing it to capture complex relationships in the data.  
Logistic Regression assumes a linear relationship between variables, limiting its predictive capacity.

#### **2. Handling Class Imbalance**  
XGBoost allows adjusting the `scale_pos_weight` parameter, improving the detection of delayed flights (class 1).  
Logistic Regression can use `class_weight`, but it does not achieve the same performance on imbalanced data.

#### **3. Better Generalization Capability**  
XGBoost reduces overfitting through techniques such as regularization and tree pruning.  
Logistic Regression can be affected by data imbalance and categorical variables.

#### **4. Feature Importance and Automatic Variable Selection**  
XGBoost allows visualization of variable importance with `plot_importance`.  
Logistic Regression does not have a native method for this and requires additional manual processing.

#### **Conclusion**  
XGBoost provides better precision, recall, and ability to detect delays, especially on imbalanced and non-linear data, making it the definitive model choice.

---

## **Part 2: API Implementation with FastAPI**

A **REST service with FastAPI** was implemented to expose the prediction model. This service allows querying the probability of a flight delay based on its characteristics.

### **Main Endpoints**
- `POST /predict`: Receives flight data and returns the probability of delay.
- `GET /health`: Health check endpoint for API monitoring.

The API was tested and validated with `make api-test`.

---

## **Part 3: Cloud Deployment (GCP)**

The API was deployed on **Google Cloud Platform (GCP)** using **Cloud Functions** for inference. The trained model is stored locally within the **Cloud Function** container and is dynamically loaded when making a prediction.

---

## **Part 4: CI/CD with GitHub Actions**

A continuous integration and deployment (CI/CD) pipeline was configured using **GitHub Actions** to ensure code stability and automatic deployment to the cloud.

The workflow includes:
- **Automated test execution** to validate the model and API.
- **Automatic deployment to GCP** when merging into the `main` branch.

---

## **Additional Exploration: `dev-all-pipeline`**

In addition to meeting the challenge requirements, an alternative approach was explored in the **`dev-all-pipeline`** branch, aiming to structure a more complete and modularized pipeline, including:
- Clear separation of feature engineering, training, and inference processes.
- Greater control over the quality of experiments and model testing.
- Improved organization and scalability of the code.

However, due to time constraints, it was not possible to achieve the expected implementation quality. Nevertheless, this initiative was included as part of the additional work explored in this challenge.

---

## **Conclusion**

This repository contains the complete solution for the **Software Engineer (ML & LLMs) Challenge**, where all challenge requirements were met:
✅ Model implementation in Python.  
✅ API creation with FastAPI.  
✅ Deployment on GCP with Cloud Functions.  
✅ CI/CD with GitHub Actions.  

Additionally, an alternative solution was explored in the `dev-all-pipeline` branch, which could improve project organization and scalability in the future.

Despite time challenges, a functional and scalable solution was delivered. 🚀

