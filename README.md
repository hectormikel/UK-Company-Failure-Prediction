# UK Company Failure Prediction

## Overview
This project builds and trains neural network models using PyTorch to predict company failures in the UK. The dataset is sourced from `UK failures data full.xlsx`, and features are extracted from financial records. The model is trained using binary classification techniques, with evaluation performed using ROC-AUC analysis.

## Requirements
Ensure you have the following dependencies installed:

```bash
pip install torch scikit-learn numpy polars matplotlib openpyxl
```

## Dataset
The dataset should be an Excel file named `UK failures data full.xlsx` containing financial and company-related data. The target variable is `Failure`, which indicates whether a company failed.

## Model Training
1. Load and preprocess data using `polars`.
2. Split data into training and test sets.
3. Convert `polars` DataFrame into `numpy` arrays for PyTorch compatibility.
4. Define neural network architectures:
   - `Net`: A basic feedforward neural network.
   - `TunedNet`: An optimized model with batch normalization and dropout.
   - `SimpleNet`: A minimalistic model designed to mimic a Keras implementation.
5. Train the model using the Adam optimizer and BCEWithLogitsLoss.
6. Evaluate model performance using accuracy and ROC-AUC.

## Usage
Run the script in a Python environment to train and evaluate the model.

```bash
python train.py
```

Replace `train.py` with the filename containing your script.

## Evaluation
- The trained model's performance is assessed using test accuracy and the ROC curve.
- AUC scores are calculated for different models.

## Results
The trained models are evaluated based on their ability to predict company failures. The final ROC-AUC scores and accuracy metrics help determine the best-performing architecture.

## Future Improvements
- Feature engineering to enhance model inputs.
- Hyperparameter tuning using grid search or Bayesian optimization.
- Deployment of the model as a web service for real-world applications.

## License
This project is for educational and research purposes.

