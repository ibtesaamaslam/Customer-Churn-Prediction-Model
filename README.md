<img width="1024" height="682" alt="image" src="https://github.com/user-attachments/assets/e86768cb-92cf-4748-83e9-f328a0276159" />
<br>

# 🚀 Customer Churn Prediction Model
### Using Artificial Neural Network (ANN)

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

> **🔥 Predict customer churn before it happens!**  
> A complete end-to-end Artificial Neural Network (ANN) solution built on the Telco Customer Churn dataset — with every code block explained for absolute beginners.

---

## 📖 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [✨ Key Features](#-key-features)
- [📊 Dataset](#-dataset)
- [🛠️ Model Architecture](#️-model-architecture)
- [📘 Step-by-Step Code Explanation](#-step-by-step-code-explanation)
- [🚀 Installation & How to Run](#-installation--how-to-run)
- [📈 Results & Visualizations](#-results--visualizations)
- [🔧 Dependencies](#-dependencies)
- [💡 Business Impact](#-business-impact)
- [📌 Future Improvements](#-future-improvements)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## 🎯 Project Overview

Customer **churn** is when a customer stops using a company's services. It is one of the most expensive problems in subscription businesses — telecom, SaaS, banking, and beyond. Retaining an existing customer is **5–25× cheaper** than acquiring a new one.

This project builds a powerful **Artificial Neural Network (ANN)** to predict whether a customer will churn (`Yes` / `No`). The model leverages 20+ customer features including tenure, monthly charges, contract type, payment method, and service subscriptions.

**Why ANN?**  
Deep neural networks excel at capturing complex non-linear patterns and feature interactions that traditional machine learning models often miss.

**Goal:** Deliver high **precision + recall** so businesses can proactively offer retention campaigns to at-risk customers — before they leave.

---

## ✨ Key Features

- ✅ Complete end-to-end pipeline (raw CSV → trained ANN)
- ✅ Professional data cleaning & preprocessing
- ✅ Handles missing values in `TotalCharges`
- ✅ One-hot encoding of all categorical features
- ✅ Feature scaling for neural network stability
- ✅ Deep ANN with hidden layers, ReLU activation & Dropout regularization
- ✅ Training history visualization (accuracy & loss curves)
- ✅ Ready for hyperparameter tuning
- ✅ Every code block explained for **absolute beginners**

---

## 📊 Dataset

**File:** `Customer-Churn.csv` *(included in repository)*  
**Size:** 7,043 customers × 21 columns  
**Source:** Classic Telco Customer Churn dataset

### Key Columns

| Column | Description | Type |
|---|---|---|
| `customerID` | Unique ID — dropped before training | String |
| `gender` | Male / Female | Categorical |
| `SeniorCitizen` | 0 = No, 1 = Yes | Binary |
| `Partner` | Has a partner? | Yes/No |
| `Dependents` | Has dependents? | Yes/No |
| `tenure` | Months with the company | Numerical |
| `PhoneService` | Has phone service? | Yes/No |
| `MultipleLines` | Multiple lines option | Categorical |
| `InternetService` | DSL / Fiber optic / No | Categorical |
| `OnlineSecurity` | Online security subscription | Categorical |
| `OnlineBackup` | Online backup add-on | Categorical |
| `DeviceProtection` | Device protection plan | Categorical |
| `TechSupport` | Tech support subscription | Categorical |
| `StreamingTV` | Streaming TV add-on | Categorical |
| `StreamingMovies` | Streaming movies add-on | Categorical |
| `Contract` | Month-to-month / One year / Two year | Categorical |
| `PaperlessBilling` | Paperless billing enabled? | Yes/No |
| `PaymentMethod` | Electronic check, Mailed check, etc. | Categorical |
| `MonthlyCharges` | Current monthly bill amount | Numerical |
| `TotalCharges` | Total amount billed — contains blanks for new customers | Numerical |
| `Churn` | **Target variable:** Yes = customer left | Binary |

> **⚠️ Important Note:** `TotalCharges` contains blank strings for customers with `tenure = 0` (brand-new customers). The notebook converts these to numeric and handles them properly using `pd.to_numeric(..., errors='coerce')` followed by `dropna()`.

---

## 🛠️ Model Architecture

```
Sequential ANN
├── Input Layer      →  26 neurons  (after one-hot encoding + scaling)
├── Hidden Layer 1   →  32 neurons  +  ReLU activation  +  Dropout(0.2)
├── Hidden Layer 2   →  16 neurons  +  ReLU activation
└── Output Layer     →   1 neuron   +  Sigmoid  (outputs churn probability 0–1)
```

| Component | Choice | Reason |
|---|---|---|
| **Loss Function** | Binary Crossentropy | Standard for binary classification |
| **Optimizer** | Adam | Adaptive learning rate; fast convergence |
| **Output Activation** | Sigmoid | Produces probability between 0 and 1 |
| **Hidden Activation** | ReLU | Avoids vanishing gradient; trains fast |
| **Regularization** | Dropout (0.2) | Prevents overfitting on training data |
| **Metrics** | Accuracy, Precision, Recall, F1-score | Balanced view of model performance |

---

## 📘 Step-by-Step Code Explanation

Every single block from **`Churn Prediction Model.ipynb`** is explained below so even a complete beginner can follow along.

---

### 1. Title & Introduction *(Markdown cell)*

```markdown
# Customer Churn Prediction Model Using Artificial Neural Network (ANN)
```

**Explanation:** Just a heading — pure documentation. No code runs here. Good practice to always title your notebooks clearly.

---

### 2. Import Libraries *(Code cell)*

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

**Explanation:**

- `pandas` — Reads CSV files and manipulates tabular data (think: Excel in Python).
- `numpy` — Fast math and array operations that pandas relies on under the hood.
- `matplotlib.pyplot` — Creates professional plots and charts.

> **Tip:** These three libraries are the foundation of almost every data science project in Python.

---

### 3. Loading the Dataset *(Code cell)*

```python
df = pd.read_csv(r"C:\Customer-Churn.csv")
```

**Explanation:** Reads the CSV file from disk into a Pandas DataFrame named `df`. The `r` prefix before the file path treats backslashes as raw characters — essential for Windows paths. After this line, all 7,043 customer records are in memory and ready to work with.

---

### 4. Quick Peek at Data *(Code cell)*

```python
df.sample(5)
```

**Explanation:** Displays 5 randomly selected rows so you can eyeball the data — see what columns look like, what values exist, and do a sanity check before proceeding.

---

### 5. Data Cleaning — Drop Useless Column *(Code cell)*

```python
df.drop('customerID', axis='columns', inplace=True)
```

**Explanation:**

- `customerID` is unique per row and carries zero predictive signal.
- `axis='columns'` tells pandas to drop a column (not a row).
- `inplace=True` modifies `df` directly instead of creating a copy.

After this step, we're left with 20 meaningful features.

---

### 6. Confirm Column Removal *(Code cell)*

```python
df.sample(5)
```

**Explanation:** Another quick peek to confirm `customerID` is gone. Always verify your cleaning steps!

---

### 7. Handling `TotalCharges` — Critical Fix! *(Code cell)*

```python
pd.to_numeric(df.TotalCharges, errors='coerce').isnull()
```

**Explanation:**

- Some rows have blank strings (`""`) in `TotalCharges` — these belong to customers with `tenure = 0` (brand-new, never billed).
- `pd.to_numeric(..., errors='coerce')` attempts conversion to a number; anything that fails becomes `NaN`.
- `.isnull()` returns a boolean mask showing which rows have the problem.

**Full fix applied in the notebook:**

```python
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(subset=['TotalCharges'], inplace=True)
```

---

### 8. Encoding Categorical Features *(Code cell)*

```python
df = pd.get_dummies(df, drop_first=True)
```

**Explanation:** Neural networks require numerical inputs — they cannot work with strings like `"Male"` or `"Month-to-month"`. `pd.get_dummies()` converts every categorical column into binary 0/1 indicator columns (one-hot encoding). `drop_first=True` removes one redundant column per category to avoid multicollinearity.

---

### 9. Encoding the Target Variable *(Code cell)*

```python
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
```

**Explanation:** Converts the target column from text (`Yes`/`No`) to integers (`1`/`0`) so the neural network can compute loss and gradients against it.

---

### 10. Splitting Features & Target *(Code cell)*

```python
X = df.drop('Churn', axis='columns')
y = df['Churn']
```

**Explanation:**

- `X` = all input features (everything except the target).
- `y` = the target column (what we're trying to predict).

---

### 11. Train/Test Split *(Code cell)*

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Explanation:** Splits data into 80% training and 20% testing. `random_state=42` ensures reproducibility — you'll get the same split every time you run the notebook.

---

### 12. Feature Scaling *(Code cell)*

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
```

**Explanation:** Neural networks are sensitive to the scale of inputs. `StandardScaler` standardizes each feature to have mean = 0 and standard deviation = 1. Key rule: **fit only on training data**, then transform both train and test — this prevents data leakage.

---

### 13. Building the ANN *(Code cell)*

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

**Explanation:**

- `Sequential` = layers stacked one after another.
- `Dense(32, activation='relu')` = 32 neurons, each connected to every input, using ReLU activation.
- `Dropout(0.2)` = randomly deactivates 20% of neurons during each training step to prevent overfitting.
- `Dense(1, activation='sigmoid')` = output layer producing a probability between 0 and 1.

---

### 14. Compiling the Model *(Code cell)*

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

**Explanation:** `compile()` configures the learning process. Adam optimizer adapts the learning rate automatically. Binary crossentropy is the standard loss for yes/no classification problems.

---

### 15. Training the Model *(Code cell)*

```python
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test)
)
```

**Explanation:** Trains the ANN for 50 passes over the training data. `batch_size=32` means the model updates weights after every 32 samples. `validation_data` lets us monitor performance on unseen data at each epoch.

---

### 16. Visualizing Training History *(Code cell)*

```python
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_title('Model Accuracy')
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Epoch')
ax1.legend(loc='upper left')

ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_title('Model Loss')
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()
```

**Explanation:**

- `history` stores accuracy and loss recorded after each epoch.
- We create **two subplots** stacked vertically:
  - **Top:** Accuracy over time for both train and validation sets.
  - **Bottom:** Loss (error) over time for both sets.
- Watching both curves together helps diagnose overfitting (training accuracy rises while validation drops).
- `plt.tight_layout()` prevents labels from overlapping.

---

### 17. Evaluating the Model *(Code cell)*

```python
from sklearn.metrics import classification_report, confusion_matrix

y_pred = (model.predict(X_test) > 0.5).astype(int)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

**Explanation:**

- `model.predict()` outputs probabilities (e.g., `0.73`).
- `> 0.5` converts to binary predictions: above 0.5 → churn, below → no churn.
- `confusion_matrix` shows true positives, false positives, true negatives, false negatives.
- `classification_report` prints precision, recall, and F1-score for both classes.

---

## 🚀 Installation & How to Run

**Step 1 — Clone the repository**

```bash
git clone <https://github.com/ibtesaamaslam/Customer-Churn-Prediction-Model>
cd Customer-Churn-Prediction-ANN
```

**Step 2 — Create a virtual environment** *(recommended)*

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux
```

**Step 3 — Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 4 — Launch Jupyter Notebook**

```bash
jupyter notebook
```

Open **`Churn Prediction Model.ipynb`** and run all cells. 🎉

> **Note:** `Customer-Churn.csv` is already included in the repository — no additional download needed.

---

## 📈 Results & Visualizations

Training curves show accuracy steadily increasing and loss decreasing over 50 epochs, indicating stable learning without severe overfitting.

**Typical performance on this dataset:**

| Metric | Score |
|---|---|
| Accuracy | ~80–82% |
| Precision (churn class) | ~75%+ |
| Recall (churn class) | ~70%+ |
| F1-Score (churn class) | ~72%+ |

Accuracy and loss plots are generated directly inside the notebook as interactive Matplotlib figures.

---

## 🔧 Dependencies

**`requirements.txt`**

```
pandas
numpy
matplotlib
tensorflow>=2.10
scikit-learn
jupyter
seaborn
```

Install all at once with:

```bash
pip install -r requirements.txt
```

---

## 💡 Business Impact

**Why does this model matter in the real world?**

- **Cost savings** — Target only high-risk customers with discounts or retention campaigns instead of blanket offers to everyone.
- **Revenue protection** — Reducing churn by even 5–10% can translate to significant recurring revenue gains.
- **Actionable insights** — The model highlights which features most strongly drive churn, such as month-to-month contracts, fiber optic service, and high monthly charges, giving product and CX teams clear levers to pull.
- **Proactive vs reactive** — Shift from reacting to cancellations to preventing them before the customer decides to leave.

---

## 📌 Future Improvements

- [ ] Hyperparameter tuning with **Keras Tuner** or **GridSearchCV**
- [ ] Model explainability with **SHAP** or **LIME**
- [ ] Handle class imbalance using **SMOTE** or class weights
- [ ] Deploy as a web app using **FastAPI** + **Streamlit**
- [ ] Benchmark against **XGBoost** and **LightGBM**
- [ ] Add cross-validation for more robust evaluation

---

## 🤝 Contributing

Contributions are very welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature-name`)
3. Commit your changes (`git commit -m 'Add: your feature description'`)
4. Push to the branch (`git push origin feature/your-feature-name`)
5. Open a Pull Request

Please open an issue first to discuss major changes. All contributions — bug fixes, documentation, new features — are appreciated. ❤️

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for full details.

---

<div align="center">

Made with ❤️ for the data science community
⭐ If you found this project helpful, please consider giving it a star!

</div>
