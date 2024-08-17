import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from bokeh.plotting import figure, show
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.layouts import gridplot

def main():
    # Define the file path to your dataset
    file_path = 'concrete.csv'  # Replace with the correct path to your dataset

    # Initialize the ConcreteStrengthPredictor class
    predictor = ConcreteStrengthPredictor(file_path)
    
    # Prepare the data (split into training and test sets, and scale features)
    predictor.prepare_data()
    
    # Train the model and evaluate it
    predictor.train_model()
    predictor.evaluate_model()
    
    # Perform visualizations with a buffer between each one
    print("\nVisualizing Results:")
    
    predictor.visualize_results()           # Visualize actual vs predicted results
    input("\nPress Enter to continue to the next visualization...")

    predictor.plot_feature_importance()     # Plot feature importance
    input("\nPress Enter to continue to the next visualization...")

    predictor.plot_residuals()              # Plot residuals
    input("\nPress Enter to continue to the next visualization...")

    predictor.plot_actual_vs_predicted()    # Plot actual vs predicted values
    input("\nPress Enter to continue to the next visualization...")

    predictor.plot_error_distribution()     # Plot error distribution
    input("\nPress Enter to continue to the next visualization...")

    predictor.plot_learning_curve()         # Plot learning curve

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
        # Separate features (X) and target (y) from the dataset
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

        # Train the model on the scaled training data
        self._model.fit(self._X_train_scaled, self._y_train)

    def evaluate_model(self):
        # Predict the target variable using the test data
        self._predictions = self._model.predict(self._X_test_scaled)

        # Calculate and print the R² score and Mean Squared Error (MSE)
        r2 = r2_score(self._y_test, self._predictions)
        mse = mean_squared_error(self._y_test, self._predictions)
        print(f"R² Score: {r2:.2f}")
        print(f"Mean Squared Error: {mse:.2f}")

    def visualize_results(self):
        # Prepare the data source for the actual values plot
        source_actual = ColumnDataSource(data=dict(
            Index=np.arange(len(self._y_test)),
            strength=self._y_test
        ))

        # Create a figure for the actual strength values
        p1 = figure(title="Actual Strength", x_axis_label='Index', y_axis_label='Actual Strength', width=400, height=400)
        p1.scatter('Index', 'strength', source=source_actual, color="blue", legend_label="Actual")

        # Prepare the data source for the predicted values plot
        source_predicted = ColumnDataSource(data=dict(
            Index=np.arange(len(self._y_test)),
            strength=self._predictions
        ))

        # Create a figure for the predicted strength values
        p2 = figure(title="Predicted Strength", x_axis_label='Index', y_axis_label='Predicted Strength', width=400, height=400)
        p2.scatter('Index', 'strength', source=source_predicted, color="red", legend_label="Predicted")

        # Create a combined figure for actual vs predicted values
        p3 = figure(title="Actual vs Predicted Strength", x_axis_label='Index', y_axis_label='Strength', width=400, height=400)
        p3.scatter('Index', 'strength', source=source_actual, color="blue", legend_label="Actual")
        p3.scatter('Index', 'strength', source=source_predicted, color="red", legend_label="Predicted", alpha=0.6)

        # Add hover tool to the plots for interactivity
        hover = HoverTool()
        hover.tooltips = [("Index", "@Index"), ("Strength", "@strength")]
        p1.add_tools(hover)
        p2.add_tools(hover)
        p3.add_tools(hover)
    
        # Arrange plots in a grid layout
        grid = gridplot([[p1, p2, p3]])
    
        show(grid)

    def plot_feature_importance(self):
        # Extract feature importances from the trained model
        importances = self._model.feature_importances_
        features = self._data.columns[:-1]

        # Sort by importance
        indices = np.argsort(importances)[::-1]
        sorted_features = [features[i] for i in indices]
        sorted_importances = importances[indices]

        # Create data source
        source = ColumnDataSource(data=dict(
            features=sorted_features,
            importances=sorted_importances
        ))

        # Create figure
        p = figure(x_range=sorted_features, height=400, width=800, title="Feature Importance")

        # Bar glyph
        p.vbar(x='features', top='importances', width=0.9, source=source)
        p.y_range.start = 0
        p.xaxis.major_label_orientation = 1.2

        show(p)

    def plot_residuals(self):
        # Calculate residuals (differences between actual and predicted values)
        residuals = self._y_test - self._predictions

        # Convert residuals to a numpy array to ensure correct indexing
        residuals = residuals.to_numpy()

        # Sort the residuals and corresponding predicted values for plotting
        sorted_indices = np.argsort(self._predictions)
        sorted_predictions = self._predictions[sorted_indices]
        sorted_residuals = residuals[sorted_indices]

        # Create a data source for the sorted residuals plot
        source = ColumnDataSource(data=dict(
            predicted=sorted_predictions,
            residuals=sorted_residuals
        ))

        # Create a figure for the residuals vs predicted values plot
        p = figure(width=800, height=400, title="Sorted Residuals vs Predicted",
                    x_axis_label='Predicted Strength', y_axis_label='Residuals')

        # Add a scatter plot for residuals
        p.scatter('predicted', 'residuals', source=source, color='red', alpha=0.6)

        # Add a horizontal line at y=0 to indicate no error
        p.line([sorted_predictions.min(), sorted_predictions.max()], [0, 0], color='blue', line_width=2, line_dash='dashed')

        show(p)

    def plot_actual_vs_predicted(self):
        # Create the data source for Bokeh
        source = ColumnDataSource(data=dict(
            actual=self._y_test,
            predicted=self._predictions
        ))

        # Create the figure
        p = figure(width=800, height=400, title="Actual vs Predicted",
                x_axis_label='Actual Strength', y_axis_label='Predicted Strength')

        # Add a scatter glyph
        p.scatter('actual', 'predicted', source=source, color='green', alpha=0.6)
        
        # Add a 45-degree reference line
        p.line([self._y_test.min(), self._y_test.max()], [self._y_test.min(), self._y_test.max()],
            color='blue', line_width=2)
        
        show(p)


    def plot_error_distribution(self):
        residuals = self._y_test - self._predictions

        # Create the figure
        p = figure(width=800, height=400, title="Distribution of Residuals",
                x_axis_label='Residuals', y_axis_label='Frequency')

        # Create a histogram of residuals
        hist, edges = np.histogram(residuals, bins=20)

        # Add the quad glyph to represent the histogram
        p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color="purple", line_color="white", alpha=0.7)
        
        show(p)
    
    def plot_learning_curve(self):
        train_sizes, train_scores, test_scores = learning_curve(self._model, self._X_train_scaled, self._y_train, cv=5)

        # Compute mean and standard deviation
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        # Create the figure
        p = figure(width=800, height=400, title="Learning Curve",
                x_axis_label='Training set size', y_axis_label='Score')

        # Add a line glyph for the mean scores
        p.line(train_sizes, train_mean, color='blue', legend_label='Training score')
        p.line(train_sizes, test_mean, color='orange', legend_label='Cross-validation score')

        # Add a band glyph to represent the standard deviation
        p.varea(x=train_sizes, y1=train_mean - train_std, y2=train_mean + train_std, color='blue', alpha=0.1)
        p.varea(x=train_sizes, y1=test_mean - test_std, y2=test_mean + test_std, color='orange', alpha=0.1)

        # Position the legend at a specific location, such as 'top_left'
        p.legend.location = "top_left"
        
        show(p)

main()