import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from bokeh.plotting import figure, show
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.layouts import gridplot

class ConcreteStrengthPredictor:
    def __init__(self, file_path):
        self._data = pd.read_csv(file_path)

        self._model = linear_model.RidgeCV()
        self._scaler = StandardScaler()
        self._pca = PCA(n_components=4)

        self._X_train = self._X_test = self._y_train = self._y_test = None
        self._X_train_pca = self._X_test_pca = None

        self._predictions = None

    def _addReciprocalLogFeatures(self, numeric):
        """Add reciprocal logarithmic features to the numeric data."""
        log_feats = numeric.copy()
        valid = (log_feats != 1) & (log_feats > 0)
        log_feats[valid] = np.log(log_feats[valid]) / np.log(10)
        log_feats[log_feats <= 0] = 1e-10
        rec_log_feats = 1 / log_feats
        return np.hstack([numeric, rec_log_feats, numeric * rec_log_feats])

    def prepare_data(self):
        X = self._data.drop(columns='strength')
        y = self._data['strength']

        X_enhanced = self._addReciprocalLogFeatures(X.values)

        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(X_enhanced, y, test_size=0.2, random_state=21)

        X_train_scaled = self._scaler.fit_transform(self._X_train)
        X_test_scaled = self._scaler.transform(self._X_test)

        self._X_train_pca = self._pca.fit_transform(X_train_scaled)
        self._X_test_pca = self._pca.transform(X_test_scaled)

    
    def train_model(self):
        self._model.fit(self._X_train_pca, self._y_train)

    def evaluate_model(self):
        self._predictions = self._model.predict(self._X_test_pca)

        model_score = self._model.score(self._X_test_pca, self._y_test)
        mse = mean_squared_error(self._y_test, self._predictions)

        print(f"Model Score: {model_score:.2f}")
        print(f"Mean Squared Error: {mse:.2f}")

    def visualize_results(self):
        # Scatter plot of actual vs PCA
        source_actual = ColumnDataSource(data=dict(
        PCA=self._X_test_pca.flatten(),
        strength=self._y_test
        ))

        p1 = figure(title="PCA vs Actual Strength", x_axis_label='PCA Component', y_axis_label='Actual Strength', width=400, height=400)
        p1.scatter('PCA', 'strength', source=source_actual, color="blue", legend_label="Actual")

        # Scatter plot of predicted vs PCA
        source_predicted = ColumnDataSource(data=dict(
        PCA=self._X_test_pca.flatten(),
        strength=self._predictions
        ))

        p2 = figure(title="PCA vs Predicted Strength", x_axis_label='PCA Component', y_axis_label='Predicted Strength', width=400, height=400)
        p2.scatter('PCA', 'strength', source=source_predicted, color="red", legend_label="Predicted")

        # Combined scatter plot of actual and predicted vs PCA
        p3 = figure(title="PCA vs Actual and Predicted Strength", x_axis_label='PCA Component', y_axis_label='Strength', width=400, height=400)
        p3.scatter('PCA', 'strength', source=source_actual, color="blue", legend_label="Actual")
        p3.scatter('PCA', 'strength', source=source_predicted, color="red", legend_label="Predicted", alpha=0.6)

        hover = HoverTool()
        hover.tooltips = [("PCA", "@PCA"), ("Strength", "@strength")]
        p1.add_tools(hover)
        p2.add_tools(hover)
        p3.add_tools(hover)
    
        # Arrange plots in a grid layout
        grid = gridplot([[p1, p2, p3]])
    
        show(grid)

file_path = 'concrete.csv'
predictor = ConcreteStrengthPredictor(file_path)
predictor.prepare_data()
predictor.train_model()
predictor.evaluate_model()
predictor.visualize_results()