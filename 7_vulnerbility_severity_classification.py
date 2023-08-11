import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle

logo = r"""
   /\_/\  
  ( o.o ) 
   > ^ <
milosilo.com
"""
print(logo)
# Introduction
print("=" * 50)
print("Vulnerability Severity Classification Script")
print("=" * 50)
print("This script demonstrates a vulnerability severity classification task using machine learning.")
print("The task involves predicting the severity of vulnerabilities based on their associated attributes.")
print("The script utilizes a RandomForestClassifier model trained on a training dataset.")
print("The trained model is then used to predict the vulnerability severity for a demo dataset.")
print("Various techniques such as data preprocessing, feature vectorization, and model training are employed.")
print("Additionally, recommended actions are provided based on the predicted severity levels.")
print("\n")
time.sleep(25)

# Importing necessary libraries
print("Importing libraries for the vulnerability severity classification task...")
time.sleep(1)
print("The 'pandas' library is used for data manipulation and analysis.")
time.sleep(4)
print("The 'scikit-learn' library is used for machine learning tasks.")
time.sleep(4)
print("Libraries imported.")
print("\n")
time.sleep(1)

# Training dataset
train_data = {
    'host': [
        "192.168.0.2",
        "192.168.0.5",
        "192.168.0.7",
        "192.168.0.3",
        "192.168.0.8",
        "192.168.0.9",
        "192.168.0.12",
        "192.168.0.15",
        "192.168.0.10",
        "192.168.0.13",
        "192.168.0.18",
        "192.168.0.20"
    ],
    'port': [
        "80",
        "443",
        "22",
        "3306",
        "8080",
        "8080",
        "22",
        "3306",
        "80",
        "443",
        "5900",
        "3389"
    ],
    'protocol': [
        "tcp",
        "tcp",
        "tcp",
        "tcp",
        "tcp",
        "tcp",
        "tcp",
        "tcp",
        "tcp",
        "tcp",
        "tcp",
        "tcp"
    ],
    'service': [
        "HTTP",
        "HTTPS",
        "SSH",
        "MySQL",
        "HTTP Proxy",
        "HTTP Proxy",
        "SSH",
        "MySQL",
        "HTTP",
        "HTTPS",
        "VNC",
        "RDP"
    ],
    'vulnerabilities': [
        "Critical",
        "Medium",
        "Low",
        "Medium",
        "Critical",
        "Medium",
        "Low",
        "High",
        "High",
        "Low",
        "Medium",
        "Low"
    ]
}

# Demo dataset
demo_data = {
    'host': [
        "192.168.0.22",
        "192.168.0.25",
        "192.168.0.28",
        "192.168.0.30",
        "192.168.0.33"
    ],
    'port': [
        "443",
        "8080",
        "22",
        "80",
        "3389"
    ],
    'protocol': [
        "tcp",
        "tcp",
        "tcp",
        "tcp",
        "tcp"
    ],
    'service': [
        "HTTPS",
        "HTTP Proxy",
        "SSH",
        "HTTP",
        "RDP"
    ],
    'vulnerabilities': [
        "", "", "", "", ""
    ]
}

# Check if all arrays in train_data have the same length
data_lengths = set(len(values) for values in train_data.values())
max_length = max(data_lengths)
train_data = {k: v[:max_length] + [v[-1]] * (max_length - len(v)) for k, v in train_data.items()}

# Creating DataFrames from the datasets
print("Creating DataFrames from the training and demo datasets...")
time.sleep(2)
train_df = pd.DataFrame(train_data)
demo_df = pd.DataFrame(demo_data)

# Reorder demo_df columns to match train_df
demo_df = demo_df[train_df.columns]

print("DataFrames created.")
print("\n")
time.sleep(3)

# Print original datasets
print("Original datasets:")
print("Training dataset sample:")
print(train_df.head())
print("\n")
print("Demo dataset:")
print(demo_df.head())
print("\n")
time.sleep(15)

# Splitting training dataset into features and target variable
X_train = train_df.drop('vulnerabilities', axis=1)
y_train = train_df['vulnerabilities']

# Vectorizing textual features
print("Initiating the process of vectorizing textual features in the training dataset to enable effective machine learning analysis of key text attributes...")
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train.astype(str).apply(lambda x: ' '.join(x), axis=1))
print("Textual features vectorized.")
print("\n")
time.sleep(5)

# Save the trained model or load the pre-trained model
model_file = 'trained_model.pkl'

try:
    # Try to load the model from the file
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    print("Pre-trained model loaded.")
except FileNotFoundError:
    # Train the model and save it for future use
    print("Training the machine learning model...")
    model = RandomForestClassifier()
    model.fit(X_train_vectorized, y_train)
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print("Model trained and saved as 'trained_model.pkl'.")
print("\n")
time.sleep(7)

# Vectorizing demo dataset textual features
X_demo_vectorized = vectorizer.transform(demo_df.astype(str).apply(lambda x: ' '.join(x), axis=1))

# Check if X_demo_vectorized has the same number of features as X_train_vectorized
if X_demo_vectorized.shape[1] < X_train_vectorized.shape[1]:
    # Pad X_demo_vectorized with zeros to match the number of features
    X_demo_vectorized = pd.DataFrame.sparse.from_spmatrix(
        X_demo_vectorized,
        columns=vectorizer.get_feature_names()
    )
    X_demo_vectorized = pd.concat([X_demo_vectorized, pd.DataFrame(0, index=X_demo_vectorized.index, columns=X_train_vectorized.columns.difference(X_demo_vectorized.columns))], axis=1)

# Perform vulnerability severity prediction on demo dataset
print("Performing vulnerability severity prediction on the demo dataset...")
demo_prediction = model.predict(X_demo_vectorized)
demo_df['vulnerabilities'] = demo_prediction

# Print final analysis or actions
print("Final analysis or actions:")
print("Vulnerabilities successfully analyzed and severity predictions made on the demo dataset.")
print("\n")
time.sleep(4)

# Example vulnerability analysis on the demo dataset
print("=" * 50)
print("Vulnerability Analysis Example on Demo Dataset")
print("=" * 50)
print("\n")
time.sleep(4)

for i in range(len(demo_df)):
    vulnerability = demo_df.iloc[i]
    print("Selected Vulnerability", i + 1)
    print("- Host:", vulnerability['host'])
    print("- Port:", vulnerability['port'])
    print("- Protocol:", vulnerability['protocol'])
    print("- Service:", vulnerability['service'])
    print("- Predicted Vulnerability Severity:", vulnerability['vulnerabilities'])
    print("\n")
    time.sleep(5)

# Additional red team-oriented explanations and recommendations
print("Closing remarks:")
print("- The vulnerability severity predictions are based on machine learning models trained on historical data.")
print("- The severity levels are adjusted based on the vulnerable service to reflect realistic risks.")
print("- Critical and high severity vulnerabilities demand immediate attention and thorough exploitation.")
print("- Medium and low severity vulnerabilities should not be underestimated, as they can lead to further compromise.")
print("- Red team exercises, penetration testing, and continuous security assessments are crucial for effective defense.")
print("- Collaboration between red teamers, blue teamers, and system administrators is essential for successful red teaming.")
print("\n")
time.sleep(15)

