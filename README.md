# ❤️ Heart Disease ML Project

## 📝 Project Overview
This project analyzes and predicts heart disease risk using the UCI Heart Disease dataset.  
It includes:

- 🧹 **Data preprocessing & cleaning**
- 📊 **Feature selection & dimensionality reduction (PCA)**
- 🤖 **Supervised learning** (Logistic Regression, Decision Tree, Random Forest, SVM)
- 🧩 **Unsupervised learning** (K-Means, Hierarchical Clustering)
- ⚙️ **Model optimization** (Hyperparameter tuning)
- 🌐 **Streamlit web UI** for real-time predictions
- 🚀 **[Bonus] Deployment via Ngrok**

---

## 📁 Folder Structure
Heart_Disease_Project/
│── data/
│ ├── heart_disease.csv
│ └── heart_disease_selected_features.csv
│── notebooks/
│ ├── 01_data_preprocessing.ipynb
│ ├── 02_pca_analysis.ipynb
│ ├── 03_feature_selection.ipynb
│ ├── 04_supervised_learning.ipynb
│ ├── 05_unsupervised_learning.ipynb
│ ├── 06_hyperparameter_tuning.ipynb
│── models/
│ ├── Random_Forest_model.pkl
│ ├── Logistic_Regression_model.pkl
│ ├── Decision_Tree_model.pkl
│ ├── SVM_model.pkl
│ ├── scaler.pkl
│── UI/
│ └── app.py
│── Deploy/
│ └── ngrok_deploy.py
│── results/
│ └── evaluation_metrics.txt
│── README.md
│── requirements.txt
│── .gitignore

yaml
Copy code

---

## ⚙️ Requirements
```bash
pip install -r requirements.txt
Main libraries used:

pandas, numpy, matplotlib, seaborn

scikit-learn

streamlit

pyngrok

joblib

🚀 How to Run the Project
1️⃣ Jupyter Notebooks
Open each notebook in notebooks/ to see:

Data preprocessing & cleaning

PCA and dimensionality reduction

Feature selection

Training supervised & unsupervised models

Hyperparameter tuning

2️⃣ Streamlit UI
Open terminal in UI/ folder:

bash
Copy code
cd Heart_Disease_Project/UI
streamlit run app.py
Enter patient data in the web interface to get real-time heart disease predictions.

3️⃣ Ngrok Deployment [Bonus]
Verify your Ngrok account and get authtoken:

bash
Copy code
ngrok authtoken <YOUR_AUTHTOKEN>
Open terminal in Deploy/ folder:

bash
Copy code
cd Heart_Disease_Project/Deploy
python ngrok_deploy.py
Copy the Ngrok public URL printed in the console to access the Streamlit app online.

💾 Model Files
Random_Forest_model.pkl – trained Random Forest classifier

Logistic_Regression_model.pkl – trained Logistic Regression

Decision_Tree_model.pkl – trained Decision Tree

SVM_model.pkl – trained SVM classifier

scaler.pkl – StandardScaler used for data preprocessing

Make sure these models are in the models/ folder.

📊 Data Files
heart_disease.csv – original dataset

heart_disease_selected_features.csv – dataset with selected key features

📈 Results
Evaluation metrics for all models are stored in results/evaluation_metrics.txt.
Includes Accuracy, Precision, Recall, F1-score, ROC-AUC.

📝 Notes
Use absolute paths in app.py for models and data to avoid FileNotFound errors.

Always run Streamlit apps with:

bash
Copy code
streamlit run app.py
