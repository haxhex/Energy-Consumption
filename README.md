# Machine Learning for Energy Consumption Prediction and IoT Modeling

## Overview  
This project implements machine learning models for predicting energy consumption (heating and cooling loads) in buildings, using the **Energy Efficiency Dataset**. Various regression models, including **K-Nearest Neighbors (KNN)**, **Random Forest**, **AdaBoost**, **XGBoost**, and **Support Vector Machines (SVM)**, are used to predict the heating and cooling loads. The project also explores feature selection and network-based relationship modeling in an **IoT environment** using **NetworkX** and **Gephi**.

---

## Features  
- **Machine Learning Models:** KNN, Random Forest, AdaBoost, XGBoost, MLP, SVM, CatBoost, and ExplainableBoosting.
- **Feature Selection:** Identifies key features affecting energy consumption using univariate regression and `SelectKBest`.
- **IoT Simulation:** Simulates relationships between features and target variables using correlation-based network graphs.
- **Evaluation Metrics:** MSE, RMSE, MAE, and R² to evaluate model performance.
- **Visualization:** Feature importance and regression plots to analyze prediction quality.

---

## Dataset  
The **Energy Efficiency Dataset** contains 768 samples and 10 attributes. The features include:
- **Relative_Compactness, Surface_Area, Wall_Area, Roof_Area, Overall_Height, Orientation, Glazing_Area, Glazing_Area_Distribution.**  
The target variables are:
- **Heating_Load** (y1) and **Cooling_Load** (y2)  

The dataset is publicly available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Energy+efficiency).

---

## Machine Learning Models and Parameters  
- **K-Nearest Neighbors (KNN):** Hyperparameters tuned using `GridSearchCV` for the number of neighbors, distance weights, and p-norm.  
- **Random Forest:** Number of estimators, maximum depth, and leaf node criteria optimized using grid search.  
- **AdaBoost:** Gradient boosting, decision trees, and XGBoost as base learners, with estimators and learning rates tuned.  
- **XGBoost:** Extensive tuning of learning rates, depth, and regularization parameters.  
- **Support Vector Machines (SVM):** Kernel types, regularization parameter (C), and epsilon optimized.

---

## Evaluation Metrics  
- **Mean Squared Error (MSE):** Measures the average squared error between actual and predicted values.  
- **Root Mean Squared Error (RMSE):** Provides the square root of MSE, indicating the magnitude of prediction errors.  
- **Mean Absolute Error (MAE):** Measures the average magnitude of prediction errors.  
- **R² Score:** Measures how well predictions match the actual target values.

---

## Feature Selection and Importance  
- Feature selection using **univariate regression** and **SelectKBest** identifies the top features affecting heating and cooling loads, including **Overall Height**, **Surface Area**, and **Roof Area**.  
- Graph-based relationships are visualized using correlation networks.

---

## Results  
- **Best Performing Model:** AdaBoost with a GradientBoostingRegressor as the base learner achieved the lowest MAE and RMSE across both tasks.  
- **Optimized AdaBoost Parameters:** Learning rate = 0.1, 200 estimators, base learner = GradientBoosting.  
- **Key Features:** Overall Height, Roof Area, and Relative Compactness were the most influential factors in energy consumption prediction.

---

## IoT Network Simulation  
- Relationships between features and target variables are modeled using **NetworkX** and visualized as graphs.
- Pearson and Spearman correlation networks are constructed to analyze the strength of relationships between nodes (features).

---

## Future Improvements  
- Test deep learning models like LSTMs to capture sequential dependencies in energy data.  
- Introduce ensemble-based methods combining predictions from multiple models.  
- Extend the IoT simulation to real-time sensor data to create dynamic predictive systems.

---

## References  
- **Paper:** [Machine Learning Algorithms for Prediction of Energy Consumption and IoT Modeling](https://doi.org/10.1016/j.micpro.2021.104423) by R.H. Fard and S. Hosseini  
- **Dataset:** [UCI Energy Efficiency Dataset](https://archive.ics.uci.edu/ml/datasets/Energy+efficiency)
