import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import pandas as pd


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(f"clustering/output/clustering_result_TIL_cptacCoad.csv")
num_classes = len(df.labelCluster.unique())

X = df[["l_mean", "l_std", "a_mean", "a_std", "b_mean", "b_std"]]
y = df.labelCluster
y = np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (not always necessary for Random Forest, but can be good practice)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Random Forest Accuracy: {accuracy * 100:.2f}%')

