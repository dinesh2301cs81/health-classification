# ==============================
# MODEL 3: MLP CLASSIFIER
# ==============================

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV

# Define base model
mlp = MLPClassifier(
    max_iter=1000,
    early_stopping=True,
    random_state=42
)

# Hyperparameter space
param_dist = {
    'hidden_layer_sizes': [(32,), (64,32), (128,64)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01],  
    'learning_rate_init': [0.001, 0.01]
}

# Random search (faster than grid)
random_mlp = RandomizedSearchCV(
    mlp,
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)


random_mlp.fit(X_train, y_train)


model_mlp = random_mlp.best_estimator_


y_pred_mlp = model_mlp.predict(X_test)


print("MLP Best Params:", random_mlp.best_params_)
print("\nClassification Report:\n", classification_report(y_test, y_pred_mlp))
