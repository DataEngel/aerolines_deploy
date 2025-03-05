# **Flight Delay Prediction API Project**

## **Project Overview**

Welcome to the **Flight Delay Prediction API Project**. In this project, you will help operationalize a machine learning model designed to predict the probability of flight delays at Santiago International Airport (SCL). This involves working with an existing dataset and model developed by a Data Scientist and deploying it into a fully functional API.

## **Project Context**

The goal is to enable the airport team to access flight delay predictions via an API. The provided dataset contains real and public flight information, and a machine learning model has already been developed to predict flight delays based on this data.

---

## **Dataset Description**

The dataset includes the following columns:

| Column | Description |
|--------|------------|
| `Fecha-I` | Scheduled date and time of the flight. |
| `Vlo-I` | Scheduled flight number. |
| `Ori-I` | Programmed origin city code. |
| `Des-I` | Programmed destination city code. |
| `Emp-I` | Scheduled flight airline code. |
| `Fecha-O` | Date and time of flight operation. |
| `Vlo-O` | Flight operation number of the flight. |
| `Ori-O` | Operation origin city code. |
| `Des-O` | Operation destination city code. |
| `Emp-O` | Airline code of the operated flight. |
| `DIA` | Day of the month of flight operation. |
| `MES` | Number of the month of flight operation. |
| `AÑO` | Year of flight operation. |
| `DIANOM` | Day of the week of flight operation. |
| `TIPOVUELO` | Type of flight, I = International, N = National. |
| `OPERA` | Name of the airline that operates. |
| `SIGLAORI` | Origin city name. |
| `SIGLADES` | Destination city name. |

Additionally, the Data Scientist has created the following derived columns:

| Column | Description |
|--------|------------|
| `high_season` | 1 if `Fecha-I` is between Dec-15 and Mar-3, or Jul-15 and Jul-31, or Sep-11 and Sep-30, 0 otherwise. |
| `min_diff` | Difference in minutes between `Fecha-O` and `Fecha-I`. |
| `period_day` | Categorization of `Fecha-I` into morning (5:00-11:59), afternoon (12:00-18:59), and night (19:00-4:59). |
| `delay` | 1 if `min_diff` > 15, 0 otherwise. |

---

## **Project Tasks**

### **1. Code Migration and Model Selection**
- Convert the provided Jupyter Notebook (`exploration.ipynb`) into a Python script (`model.py`).
- Fix any potential bugs in the original notebook.
- Evaluate the proposed models and select the best one, providing a justification for your choice.
- Maintain good programming practices throughout the implementation.
- Ensure that the model passes the provided tests by running `make model-test`.

> **Notes:**
> - You **cannot** remove or change the name/arguments of **provided** methods.
> - You **can** modify the implementation of the provided methods.
> - You **can** create additional helper methods and classes if necessary.

---

### **2. API Development**
- Deploy the trained model as an API using `FastAPI` in the `api.py` file.
- Ensure the API passes the provided tests by running `make api-test`.

> **Notes:** 
> - You **must** use `FastAPI` (other frameworks are not allowed).

---

### **3. Cloud Deployment**
- Deploy the API on a cloud provider of your choice (GCP is recommended).
- Update the API’s URL in the `Makefile` (line 26).
- Ensure the API passes the stress test by running `make stress-test`.

> **Note:**  
> - The API **must** remain active until reviewed.

---

### **4. Implement CI/CD**
- Create a new `.github` folder and copy the provided `workflows` folder into it.
- Complete the `ci.yml` and `cd.yml` files to automate the integration and deployment process.