# 🧱 Concrete Strength Predictor

A machine learning project to predict the compressive strength of concrete based on its composition. This project features two models — a **Random Forest Regressor** for high-accuracy predictions, and a **Linear Ridge Regression** model with PCA for dimensionality reduction. Both models include rich visualizations using Bokeh.

---

## 📊 Features

- Predicts concrete compressive strength using supervised learning
- Supports two model versions:
  - **Random Forest**: High performance and full diagnostic plots
  - **Linear Ridge Regression with PCA**: Simpler model with dimensionality reduction
- Cross-validation scoring and test set evaluation
- Interactive visualizations:
  - Actual vs Predicted strength
  - Residual analysis
  - Feature importance (Random Forest)
  - PCA component vs strength (Linear)
  - Learning curve (Random Forest)
  - Error distribution histogram

---

## 🧪 Model Performance

**Random Forest Regressor:**
- Cross-Validation R² Score: ~0.90
- Test Set R² Score: ~0.89
- MSE: (Displayed on evaluation)

**Linear Ridge Regression:**
- Model Score and MSE printed after training
- Uses top 4 PCA components to simplify features

---

## 📁 File Structure

- `Capstone_forest.py`: Main script using Random Forest
- `Capstone_linear.py`: Linear regression version
- `concrete.csv`: Dataset used (replace or update as needed)
- `requirements.txt`: Required Python packages

---

## 🔧 How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/concrete-strength-predictor
    cd concrete-strength-predictor
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the script:
    ```bash
    python Capstone_forest.py
    python Capstone_linear.py
    ```

---

## 💡 Future Improvements

- Add support for more model types (e.g., GradientBoosting, SVR)
- Export results to CSV or web dashboard
- Automate hyperparameter tuning (GridSearchCV)

---

## 🗃️ Dataset

[UCI Concrete Compressive Strength Dataset](https://archive.ics.uci.edu/ml/datasets/concrete+compressive+strength)

---

## 📬 Contact

Built by [Subhan Noor](https://www.linkedin.com/in/subhannoor)  
Feel free to reach out for collaboration or feedback!
