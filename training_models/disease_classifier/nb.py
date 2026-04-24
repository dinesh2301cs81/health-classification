# ==============================
# MODEL 2: GAUSSIAN NAIVE BAYES
# ==============================

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, log_loss

# Initialize model
model_nb = GaussianNB()

# Train model
model_nb.fit(X_train, y_train)

# Predictions
y_pred_nb = model_nb.predict(X_test)
y_prob_nb = model_nb.predict_proba(X_test)


print("\nNaive Bayes Results")
print("\nClassification Report:\n", classification_report(y_test, y_pred_nb))


print("Log Loss:", log_loss(y_test, y_prob_nb))


print("\nSample Probabilities:\n", y_prob_nb[:5])
