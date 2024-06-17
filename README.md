# Vaccine Prediction Project

This project aims to predict the likelihood of individuals receiving the XYZ and seasonal flu vaccines using machine learning on a dataset of demographic, behavioral, and opinion-based features.

## Objectives

1. **Data Preprocessing**: Gain insights into the dataset and preprocess it to ensure the features are suitable for model training.
2. **Model Development**: Develop machine learning models to predict the probabilities of receiving the XYZ and seasonal flu vaccines.
3. **Performance Evaluation**: Evaluate the models using the ROC AUC metric to ensure accurate predictions.
4. **Interpretability and Insights**: Provide insights into significant factors influencing vaccination probabilities and interpret the model results.

## Dataset

The dataset includes various demographic, behavioral, and opinion-based features. It is divided into:
- `training_set_features.csv`
- `training_set_labels.csv`
- `test_set_features.csv`
- `submission_format.csv`

## Project Structure

- `data/`: Contains the dataset files.
- `notebooks/`: Jupyter notebooks for data analysis, model development, and evaluation.
- `src/`: Source code for data preprocessing, model training, and evaluation.
- `README.md`: Project overview and instructions.

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/vaccine-prediction.git
    cd vaccine-prediction
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Preprocess the data:
    ```bash
    python src/preprocess_data.py
    ```

2. Train the models:
    ```bash
    python src/train_model.py
    ```

3. Evaluate the models:
    ```bash
    python src/evaluate_model.py
    ```

4. Generate predictions for the test set:
    ```bash
    python src/predict.py
    ```

## Evaluation Metrics

The primary metric for evaluating the models is the ROC AUC score. Additional metrics such as precision, recall, and F1-score are also considered.

## Insights and Interpretability

The project provides insights into the significant factors influencing vaccination probabilities using techniques like feature importance and SHAP values.

## Contributing

Contributions are welcome. Please submit a pull request or open an issue to discuss changes.


