# **Credit Card Fraud Detection Using Unsupervised Learning**

## **Introduction**
Credit card fraud is a growing concern in the financial industry, leading to billions of dollars in losses annually. This project explores the application of unsupervised learning techniques to identify fraudulent transactions in credit card data. Using methods like k-Means clustering, DBSCAN, and Isolation Forest, the project aims to detect anomalies and uncover patterns that differentiate legitimate transactions from fraudulent ones.

---

## **Dataset**
The dataset used for this project is the **Credit Card Fraud Detection Dataset**, sourced from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud).

### **Key Features**
- **Time**: Seconds elapsed between this transaction and the first in the dataset.
- **Amount**: Transaction amount.
- **V1-V28**: PCA-transformed features for privacy.
- **Class**: Binary label (0 = Legitimate, 1 = Fraudulent). Only used for evaluation.

---

## **Setup and Installation**
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/project-name.git
   cd project-name
   ```

2. **Install Dependencies**:
   Make sure Python 3.8+ is installed. Then, run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Dataset**:
   - Place the raw dataset in the `data/raw/` directory as `creditcard.csv`.

4. **Run Notebooks**:
   Open Jupyter notebooks in the `notebooks/` directory:
   ```bash
   jupyter notebook
   ```

---

## **How to Use**

### **Step 1: Preprocessing**
Run `preprocessing.py` to clean and transform the dataset:
```bash
python src/preprocessing.py
```

### **Step 2: Exploratory Data Analysis**
Explore the dataset using the `eda.ipynb` notebook or run `eda.py` for specific visualizations:
```bash
python src/eda.py
```

### **Step 3: Model Training**
Train and evaluate models using the `modeling.ipynb` notebook or execute specific models with `modeling.py`:
```bash
python src/modeling.py
```

### **Step 4: Visualizations**
Generate evaluation plots, such as the confusion matrix or ROC curve, using `utils.py`:
```bash
python src/utils.py
```

---

## **Results**
### **Best Model: Isolation Forest**
- **Precision**: 0.72
- **Recall**: 0.68
- **F1-Score**: 0.70
- **ROC AUC**: 0.78

### **Visualizations**
- **Confusion Matrix**: ![Confusion Matrix](results/confusion_matrix.png)
- **ROC Curve**: ![ROC Curve](results/roc_curve.png)
- **Feature Importance**: ![Feature Importance](results/feature_importance.png)

---

## **Key Insights**
1. The **Isolation Forest** model outperformed k-Means and DBSCAN in detecting fraudulent transactions.
2. The PCA-transformed features (`V1` to `V28`) played a critical role in anomaly detection.
3. Severe class imbalance posed challenges, making unsupervised learning a viable approach over supervised methods.

---

## **Technologies Used**
- **Python**: Programming language
- **Jupyter Notebooks**: Interactive environment for EDA and modeling
- **Libraries**:
  - `pandas` for data manipulation
  - `numpy` for numerical operations
  - `matplotlib` and `seaborn` for visualizations
  - `scikit-learn` for machine learning models and evaluation

---

## **Contributions**
Contributions are welcome! If you'd like to contribute, please fork the repository, create a feature branch, and submit a pull request.

---

## **License**
This project is licensed under the [MIT License](LICENSE).

---

## **Acknowledgments**
- Dataset: Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, Gianluca Bontempi.
- Kaggle community for dataset accessibility and inspiration.