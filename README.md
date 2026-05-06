# Bank Churn Prediction

Predicting customer churn for a retail bank using artificial neural networks. The model takes customer attributes (credit score, geography, gender, age, tenure, balance, number of products, activity status) and outputs a churn probability.

## Approach

- **Data:** 10,000 customer records with 10 features and binary churn label
- **Preprocessing:** Label encoding for categorical features, one-hot encoding for geography, standard scaling
- **Model:** Feedforward neural network with progressive layer widening
- **Output:** Binary sigmoid activation (churn / no churn)
- **Evaluation:** Confusion matrix, training/validation learning curves over 100 epochs

## Architecture

```
Dense (8, relu)
Dense (16, relu)
Dense (32, relu)
Dense (1, sigmoid)
```

Compiled with Adam optimizer and binary cross-entropy loss.

## Stack

Python, TensorFlow/Keras, scikit-learn, NumPy, pandas, seaborn, matplotlib

## Usage

```bash
pip install tensorflow scikit-learn pandas seaborn matplotlib
python code.py
```

Requires the [Churn Modelling dataset](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling) (CSV).
