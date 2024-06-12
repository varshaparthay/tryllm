import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

# Read the CSV file and load it into a DataFrame
df = pd.read_csv("/Users/varshaparthasarathy/Downloads/random_data.csv")

# Continue with the rest of your code...
le = LabelEncoder()
df["description"] = le.fit_transform(df["description"])
df["name"] = le.fit_transform(df["name"])

df["visit_type"] = df["visit_type"].astype(str)
df["visit_type"] = le.fit_transform(df["visit_type"])

df["hsa_fsa_eligibility"] = le.fit_transform(df["hsa_fsa_eligibility"])

df
# Extract the features and target variable
X = df.iloc[:, :-1]
y = df["hsa_fsa_eligibility"]

# print("Features:", X)
# print("Target variable:", y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=100
)

# Create a decision tree classifier
clf = DecisionTreeClassifier()

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Calculate the recall score
recall = recall_score(y_test, y_pred, average=None)
print("Recall:", recall)

# Calculate the precision score
precision = precision_score(y_test, y_pred, average=None)
print("Precision:", precision)
