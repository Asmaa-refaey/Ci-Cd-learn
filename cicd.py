# train.py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. تحميل البيانات
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42)

# 2. تدريب النموذج
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 3. تقييم النموذج
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc}")

# 4. حفظ النموذج
joblib.dump(model, "model.pkl")
