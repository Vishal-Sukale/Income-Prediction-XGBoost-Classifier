# Income Prediction Using XGBoost Classifier with Hyperparameter Tuning

## Problem Statement
The income classification problem involves predicting whether an individual
earns more or less than $50K annually based on demographic and employment
attributes such as age, education, occupation, marital status and work hours.

XGBoost (Extreme Gradient Boosting) is one of the most powerful and
widely-used ML algorithms in industry and data science competitions.
It builds trees sequentially with gradient boosting optimization —
delivering superior accuracy, speed and regularization compared to
traditional boosting methods. This project applies XGBoost on the
Adult Census Dataset (32,561 records) to build a highly accurate
income prediction model.

## Objective
- Predict income category (<=50K or >50K) using XGBoost Classifier
- Handle missing values encoded as ' ?' in the Adult dataset
- Apply One-Hot Encoding with drop_first=True to avoid multicollinearity
- Tune n_estimators, max_depth, learning_rate, subsample and
  colsample_bytree using GridSearchCV
- Evaluate using Accuracy, Precision, Recall, F1 and Classification Report
- Compare Before and After tuning performance
  
## Dataset
| Detail | Info |
|---|---|
| Name | Adult Census Income Dataset |
| Records | 32,561 |
| Features | 14 |
| Target | income (<=50K or >50K) |
| Missing Values | Encoded as ' ?' |
| Class Distribution | 24,720 (<=50K) vs 7,841 (>50K) |

## Tech Stack
| Tool | Usage |
|---|---|
| Python | Programming Language |
| Pandas | Data Manipulation |
| NumPy | Numerical Operations |
| Scikit-learn | ML Evaluation & GridSearch |
| XGBoost | ML Model |
| Jupyter Notebook | Development Environment |

## ML Pipeline
```
1. Import Libraries
2. Load Dataset
3. Data Understanding
4. Data Preprocessing
   - Handle Missing Values (' ?' → NaN → dropna)
   - One-Hot Encoding (get_dummies, drop_first=True, dtype=int)
5. Model Building
   - Split Features & Target
   - Train-Test Split (80/20)
   - Model Before Tuning (default XGBoost)
   - GridSearchCV Hyperparameter Tuning
   - Model After Tuning (best params)
6. Evaluation & Classification Report
```

## Results

| Metric | Before Tuning | After Tuning |
|---|---|---|
| **Accuracy** | 87.25% | **87.67%**  |
| **Precision** | 0.777 | **0.796** |
| **Recall** | 0.645 | 0.642 |
| **F1 Score** | 0.705 | **0.711**  |

### Best Parameters Found
| Parameter | Value |
|---|---|
| colsample_bytree | 0.8 |
| learning_rate | 0.1 |
| max_depth | 5 |
| n_estimators | 200 |
| subsample | 1.0 |

## 💡 Key Insights
- Accuracy improved from **87.25% → 87.67%** after tuning 
- Precision improved from **0.777 → 0.796** 
- F1 Score improved from **0.705 → 0.711** 
- XGBoost outperformed all previous models 
  - Decision Tree: ~85%
  - Random Forest: ~86%
  - AdaBoost: ~86.4%
  - **XGBoost: 87.67%** ← Best! 
- colsample_bytree=0.8 → reduces overfitting 
- learning_rate=0.1 with n_estimators=200 → optimal balance 
- Class imbalance: 4976 (<=50K) vs 1537 (>50K)
- Weighted avg F1 Score = 0.87 → strongest performance 
- GridSearchCV with 5-fold CV used to find optimal parameters 

## Model Comparison

| Model | Accuracy |
|---|---|
| Decision Tree | ~85.00% |
| Random Forest | ~86.17% |
| AdaBoost | ~86.40% |
| **XGBoost** | **87.67%** |

## Project Structure
```
Income-Prediction-XGBoost-Classifier/
│
├── Xgboost_Classifier_Project.ipynb    # Main Jupyter Notebook
├── Adult_Dataset.csv                   # Dataset
└── README.md                           # Project Documentation
```

---

## How to Run

1. Clone the repository
```bash
git clone https://github.com/yourusername/Income-Prediction-XGBoost-Classifier.git
```

2. Install dependencies
```bash
pip install pandas numpy scikit-learn xgboost
```

3. Open Jupyter Notebook
```bash
jupyter notebook Xgboost_Classifier_Project.ipynb
```

---

## Conclusion
The XGBoost Classifier successfully predicted individual income categories
with the highest accuracy of **87.67%** after hyperparameter tuning —
the best result across all 4 ensemble models tested.

XGBoost's gradient boosting optimization, built-in regularization and
efficient tree building gave it an edge over Decision Tree (~85%),
Random Forest (~86%) and AdaBoost (~86.4%). GridSearchCV found the
optimal combination of n_estimators=200, learning_rate=0.1, max_depth=5,
subsample=1.0 and colsample_bytree=0.8.

The Classification Report confirmed strong performance with weighted
average F1 Score of **0.87** — the highest in this portfolio. This project
demonstrates why XGBoost is the go-to algorithm for tabular data
classification in real-world ML applications.

## Author
**Vishal Sukale**

