## **Proposed Architecture Based on Cloud Run Jobs and Cloud Run Functions**

This architecture proposal aims to automate the **feature engineering, training, and inference** workflow using **Cloud Run Jobs** and **Cloud Run Functions**, orchestrated through **Pub/Sub**. However, due to time constraints, the implementation is not fully developed but is presented as a scalable and efficient solution for flight delay prediction.

---

## **Pipeline Workflow**

### **1. Feature Engineering (Cloud Run Job)**
- A **Cloud Run Job** runs daily at **00:00**, triggered by a **Cloud Scheduler**.
- Loads raw data from **Cloud Storage**.
- Performs cleaning, transformation, and feature creation (`high_season`, `min_diff`, `period_day`).
- Stores the processed data in **BigQuery**.
- **Publishes a message to Pub/Sub** indicating that the data is ready for training.
- The container for this job is located in **full_pipeline**.

### **2. Model Training (Cloud Run Job)**
- A **Cloud Run Job** for training is automatically triggered when a message is received in **Pub/Sub** confirming that the data has been processed.
- Loads the transformed data from **BigQuery**.
- Trains the model with the best identified parameters.
- Evaluates the model and saves the best version in **Cloud Storage**.
- **Publishes a message to Pub/Sub** indicating that a new model has been trained and is ready for use.
- The container for this job is located in **full_pipeline**.

### **3. Real-Time Inference (Cloud Run Function)**
- A **Cloud Run Function** exposes an HTTP endpoint to receive prediction requests.
- When it receives flight data, it loads the latest model from **Cloud Storage**.
- Processes the input and generates the delay prediction.
- **Publishes a message to Pub/Sub** with the prediction for storage or use in other systems.

---

## **Proposal for Automating Inference**
Currently, inference is performed manually via HTTP requests to the **Cloud Run Function**. To make this process fully automated, the following is proposed:

### **Trigger from BigQuery or Pub/Sub**
- Each time new flights are recorded in **BigQuery**, a **Pub/Sub** event could be automatically triggered.
- This event would send the flight information to the **Cloud Run Function** for inference.

### **Prediction Storage**
- The **Cloud Run Function** would process the request and store the results in **BigQuery** or **Cloud Storage**.
- A monitoring service could alert on recurring delay patterns.

### **Webhook for Integration with Other Systems**
- Predictions could be automatically sent to an external airline API or airport system via a **Webhook**, reducing the need for manual queries.

---

## **Complete Automation with Pub/Sub**

| **Event** | **Activated Service** | **Executed Action** |
|------------|----------------------------|----------------------|
| **00:00 (midnight)** | Cloud Scheduler | Triggers the **Cloud Run Job** for feature engineering. |
| **Processed data in BigQuery** | Pub/Sub | Triggers the **Cloud Run Job** for training. |
| **Model trained and stored in Cloud Storage** | Pub/Sub | Does not directly activate another service, but inference uses the updated model. |
| **New flight recorded in BigQuery** | Pub/Sub | Activates the **Cloud Run Function** to automatically predict whether there will be a delay. |

---

## **Conclusion**
While the implementation is not fully completed, this architecture provides a scalable and efficient solution for managing the lifecycle of a Machine Learning model. With the addition of **automatic triggers for inference**, a **fully automated workflow** would be achieved, where predictions are generated without manual intervention and stored in a structured manner for analysis and use in airport operations. 🚀

