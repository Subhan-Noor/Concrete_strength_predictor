import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from bokeh.plotting import figure, show
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.layouts import gridplot

class ConcreteStrengthPredictor:
    def __init__(self, file_path):
        self._data = pd.read_csv(file_path)

        # Using RandomForestRegressor for the model
        self._model = RandomForestRegressor(random_state=21, n_estimators=100)
        self._scaler = StandardScaler()

        self._X_train = self._X_test = self._y_train = self._y_test = None
        self._X_train_scaled = self._X_test_scaled = None

        self._predictions = None

    def prepare_data(self):
        X = self._data.drop(columns='strength')
        y = self._data['strength']

        # Train-test split
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(X, y, test_size=0.2, random_state=21)

        # Scaling the features
        self._X_train_scaled = self._scaler.fit_transform(self._X_train)
        self._X_test_scaled = self._scaler.transform(self._X_test)


    
    def train_model(self):
        # Perform cross-validation to assess model performance
        scores = cross_val_score(self._model, self._X_train_scaled, self._y_train, cv=5, scoring='r2')
        print(f"Cross-Validation R² Scores: {scores}")
        print(f"Average R² Score: {scores.mean():.2f}")

        self._model.fit(self._X_train_scaled, self._y_train)

    def evaluate_model(self):
        self._predictions = self._model.predict(self._X_test_scaled)

        r2 = r2_score(self._y_test, self._predictions)
        mse = mean_squared_error(self._y_test, self._predictions)

        print(f"R² Score: {r2:.2f}")
        print(f"Mean Squared Error: {mse:.2f}")

    def visualize_results(self):
        # Scatter plot of actual vs predicted
        source_actual = ColumnDataSource(data=dict(
            Index=np.arange(len(self._y_test)),
            strength=self._y_test
        ))

        p1 = figure(title="Actual Strength", x_axis_label='Index', y_axis_label='Actual Strength', width=400, height=400)
        p1.scatter('Index', 'strength', source=source_actual, color="blue", legend_label="Actual")

        # Scatter plot of predicted vs actual
        source_predicted = ColumnDataSource(data=dict(
            Index=np.arange(len(self._y_test)),
            strength=self._predictions
        ))

        p2 = figure(title="Predicted Strength", x_axis_label='Index', y_axis_label='Predicted Strength', width=400, height=400)
        p2.scatter('Index', 'strength', source=source_predicted, color="red", legend_label="Predicted")

        # Combined scatter plot of actual and predicted
        p3 = figure(title="Actual vs Predicted Strength", x_axis_label='Index', y_axis_label='Strength', width=400, height=400)
        p3.scatter('Index', 'strength', source=source_actual, color="blue", legend_label="Actual")
        p3.scatter('Index', 'strength', source=source_predicted, color="red", legend_label="Predicted", alpha=0.6)

        hover = HoverTool()
        hover.tooltips = [("Index", "@Index"), ("Strength", "@strength")]
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