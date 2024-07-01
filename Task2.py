import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Step 1: Data Collection

data = pd.read_csv("C:\\Users\\lohit\\OneDrive\\Desktop\\creditcard.csv")

# Step 2: Data Preprocessing
data = data.dropna()
X = data.drop(columns=['Class'])
y = data['Class']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Feature Engineering (if needed)
# Add your feature engineering code here

# Step 4: Exploratory Data Analysis (EDA)
# Add your EDA code here (e.g., using matplotlib and seaborn)

# Step 5: Model Selection
model = RandomForestClassifier()

# Step 6: Model Training
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
model.fit(X_train, y_train)

# Step 7: Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")