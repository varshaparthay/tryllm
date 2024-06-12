from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

# Read the CSV file and load it into a DataFrame
df = pd.read_csv('/Users/varshaparthasarathy/Downloads/random_data.csv')

# Continue with the rest of your code...
le=LabelEncoder()
df['description']=le.fit_transform(df['description'])
df['name']=le.fit_transform(df['name'])

df['visit_type'] = df['visit_type'].astype(str)
df['visit_type']=le.fit_transform(df['visit_type'])

df['hsa_fsa_eligibility']=le.fit_transform(df['hsa_fsa_eligibility'])

df 
# Extract the features and target variable
X = df.iloc[:, :-1]
y = df['hsa_fsa_eligibility']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=100)

# Create a random forest classifier
rf_classifier = RandomForestClassifier(n_estimators=250, random_state=100)

# Train the classifier on the training data
rf_classifier.fit(X_train, y_train)

# Use the trained classifier to make predictions on the test data
y_pred = rf_classifier.predict(X_test)

# Evaluate the performance of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate the recall score
recall = recall_score(y_test, y_pred, average=None)

# Calculate the precision score
precision = precision_score(y_test, y_pred, average=None)
print("Precision:", precision)
print("Recall:", recall)