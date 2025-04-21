
# Customer Churn Prediction Model

This project focuses on building an Artificial Neural Network (ANN) to predict customer churn for a given business. Customer churn refers to when customers stop doing business with a company. Accurately predicting churn helps businesses identify customers likely to leave and take proactive steps to retain them.

## Features
- **Data Preprocessing**: Handles missing values, encodes categorical variables, and scales numerical features.
- **Artificial Neural Network (ANN)**: The model is designed with an input layer corresponding to features, hidden layers for complex pattern recognition, and an output layer for binary classification (churn or not).
- **Evaluation**: Metrics like accuracy, precision, recall, F1-score, and confusion matrix are used to evaluate model performance.
- **Hyperparameter Tuning**: GridSearch and RandomSearch techniques can be applied to fine-tune the ANN's architecture and training process.

## Dependencies
- Python 3.x
- Pandas
- NumPy
- TensorFlow/Keras
- Scikit-learn
- Matplotlib/Seaborn (for visualization)

## How to Run
1. Clone the repository:  
   ```bash
   git clone <repository-url>
   ```
2. Install the dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:  
   Open `Churn Prediction Model.ipynb` in Jupyter Notebook or Jupyter Lab.

## Dataset
The dataset used in this model consists of various customer attributes like tenure, services subscribed, and payment methods. Each customer is labeled as "churn" or "no churn." The dataset can be loaded from a CSV file.
