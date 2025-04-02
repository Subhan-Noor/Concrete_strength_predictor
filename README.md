# 🧱 Concrete Strength Predictor

A machine learning model that predicts the compressive strength of concrete based on its composition using Random Forest Regression. Includes extensive visualizations of results, residuals, feature importance, and learning curves.

---

## 📊 Features

- Reads and processes concrete composition data from CSV
- Trains a `RandomForestRegressor` model with 90%+ R² accuracy
- Evaluates performance using cross-validation and test set metrics
- Visualizes:
  - Actual vs Predicted Strength
  - Feature Importances
  - Residual Errors
  - Error Distributions
  - Learning Curves
- Supports both Random Forest and Linear Regression models (in separate files)

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
    ```

---

## 📈 Example Outputs

### Feature Importance Plot
*(include a screenshot of the bar chart)*

### Actual vs Predicted Strength
*(include a sample image)*

---

## 🧠 Model Performance

- Cross-Validation R² Score: ~0.90
- Test Set R² Score: ~0.89
- MSE: _(insert number if you want)_

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
