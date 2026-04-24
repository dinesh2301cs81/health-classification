# ==============================
# MODEL 1: XGBOOST CLASSIFIER
# ==============================

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

# Define base model
xgb = XGBClassifier(
    objective='multi:softprob',   
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid_xgb = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)


grid_xgb.fit(X_train, y_train)


model_xgb = grid_xgb.best_estimator_
y_pred_xgb = model_xgb.predict(X_test)


print("XGBoost Best Params:", grid_xgb.best_params_)
print("\nClassification Report:\n", classification_report(y_test, y_pred_xgb))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))
