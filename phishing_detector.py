# phishing_detector.py
# Simple AI demo to detect phishing emails

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Example email dataset
emails = [
    "Congratulations! You've won a free iPhone. Click here to claim.",
    "Please find attached the agenda for tomorrow's meeting.",
    "Your account has been suspended. Login now to reactivate.",
    "Lunch meeting at 1 PM today, see you in the office."
]
labels = [1, 0, 1, 0]  # 1 = phishing, 0 = safe

# Convert text to numerical features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(emails)

# Train a simple Logistic Regression model
model = LogisticRegression()
model.fit(X, labels)

# Test the model with new emails
test_emails = [
    "Your PayPal account is limited. Click to verify immediately.",
    "Team meeting tomorrow at 10 AM in Conference Room B."
]

X_test = vectorizer.transform(test_emails)
predictions = model.predict(X_test)

# Print results
for email, pred in zip(test_emails, predictions):
    label = "Phishing" if pred == 1 else "Safe"
    print(f"Email: {email}\nPrediction: {label}\n")
