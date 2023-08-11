import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import numpy as np

logo = r"""
   /\_/\  
  ( o.o ) 
   > ^ <
milosilo.com
"""
print(logo)
# Introduction and Overview
print("APT Detection AI Model")
print("This AI model is designed to detect and predict the severity of Advanced Persistent Threats (APTs) based on various features such as attack vectors, target systems, and vulnerabilities.")
print("The model utilizes the Random Forest algorithm and text vectorization techniques to train and predict the severity of APTs.")
print("Let's proceed with creating the model.")
print("\n")
time.sleep(5)

# Importing necessary libraries
print("Importing libraries...")
time.sleep(2)
print("Importing pandas for data manipulation and analysis...")
time.sleep(2)
print("Importing scikit-learn for machine learning tasks...")
time.sleep(2)
print("Libraries imported.")
print("\n")
time.sleep(2)

# Large training dataset for APT detection
print("Preparing training dataset...")
time.sleep(2)
train_data = {
    'attack_vector': [
        "Exploit",
        "Phishing",
        "Brute Force",
        "Social Engineering",
        "Command and Control",
        "Physical Access",
        "Malware",
        "Denial of Service",
        "Insider Threat",
        "Eavesdropping",
        "Password Attack",
        "Man-in-the-Middle",
        "SQL Injection",
        "Cross-Site Scripting",
        "Zero-day Exploit",
        "Domain Fronting",
        "Data Breach",
        "Ransomware",
        "Spyware",
        "Fileless Attack"
    ],
    'target_system': [
        "Web Server",
        "Email Server",
        "Database",
        "Employee Workstation",
        "Network Firewall",
        "Physical Infrastructure",
        "Mobile Device",
        "Cloud Service",
        "Application Server",
        "Wireless Network",
        "VPN",
        "IoT Device",
        "Payment Gateway",
        "Router",
        "SCADA System",
        "Blockchain Network",
        "POS Terminal",
        "VoIP System",
        "Smart Home",
        "Medical Device",
        "Industrial Control System"
    ],
    'vulnerability': [
        "Remote Code Execution",
        "Credential Theft",
        "Weak Password",
        "Human Error",
        "Command Injection",
        "Unauthorized Access",
        "Data Exfiltration",
        "Buffer Overflow",
        "Misconfiguration",
        "Phishing",
        "Keylogger",
        "SSL/TLS Exploit",
        "Sensitive Data Exposure",
        "XML External Entity (XXE)",
        "Remote File Inclusion",
        "Advanced Persistent Threat",
        "Zero-day Vulnerability",
        "Software Vulnerability",
        "Network Vulnerability",
        "Physical Vulnerability",
        "Privilege Escalation"
    ],
    'severity': [
        "Critical",
        "Medium",
        "High",
        "Low",
        "Medium",
        "High",
        "Medium",
        "High",
        "Medium",
        "Low",
        "Medium",
        "High",
        "High",
        "Medium",
        "Critical",
        "High",
        "Medium",
        "High",
        "Medium",
        "Low",
        "High"
    ]
}
print("Training dataset prepared.")
print("\n")
time.sleep(2)

# Creating DataFrame from the training dataset
print("Creating a DataFrame from the training dataset...")
time.sleep(2)
# Padding if required
max_length = max(len(train_data['attack_vector']), len(train_data['target_system']), len(train_data['vulnerability']), len(train_data['severity']))
padded_data = {
    'attack_vector': train_data['attack_vector'] + [''] * (max_length - len(train_data['attack_vector'])),
    'target_system': train_data['target_system'] + [''] * (max_length - len(train_data['target_system'])),
    'vulnerability': train_data['vulnerability'] + [''] * (max_length - len(train_data['vulnerability'])),
    'severity': train_data['severity'] + [''] * (max_length - len(train_data['severity']))
}
train_df = pd.DataFrame(padded_data)
print("DataFrame created. (DataFrame is a 2-dimensional labeled data structure with columns of potentially different types.)")
print("\n")
time.sleep(2)

# Print original training dataset
print("Original training dataset:")
print(train_df.head())
print("\n")
time.sleep(2)

# Cleaning and preprocessing training data
print("Cleaning and preprocessing training data...")
time.sleep(2)

# Example: Removing duplicate entries
print("Removing duplicate entries... (Duplicate entries are data points with identical features and labels that can negatively affect the model's performance.)")
time.sleep(2)
train_df.drop_duplicates(inplace=True)
print("Duplicate entries removed.")
print("\n")
time.sleep(2)

# Example: Handling missing values
print("Handling missing values... (Missing values are empty or NaN (Not a Number) entries in the dataset that can cause issues during model training.)")
time.sleep(2)
train_df.dropna(inplace=True)
print("Missing values handled.")
print("\n")
time.sleep(2)

# Print processed training dataset
print("Processed training dataset:")
print(train_df.head())
print("\n")
time.sleep(2)

# Splitting training dataset into features and target variable
print("Splitting training dataset into features and target variable... (Features are the input variables used to make predictions, and the target variable is the variable to be predicted.)")
time.sleep(2)
X_train = train_df.drop('severity', axis=1)
y_train = train_df['severity']
print("Features and target variable split.")
print("\n")
time.sleep(2)

# Vectorizing textual features
print("Vectorizing textual features... (Text vectorization is the process of converting text data into numerical representations suitable for machine learning algorithms.)")
time.sleep(2)
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train.astype(str).apply(lambda x: ' '.join(x), axis=1))
print("Textual features vectorized.")
print("\n")
time.sleep(2)

# Training the machine learning model
print("Training the machine learning model... (Training a machine learning model involves feeding the model with labeled data to learn patterns and relationships between features and the target variable.)")
time.sleep(2)
model = RandomForestClassifier()
model.fit(X_train_vectorized, y_train)
print("Model trained.")
print("\n")
time.sleep(2)

# Save the trained model
print("Saving the trained model...")
time.sleep(2)
with open('apt_detection_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Trained model saved as 'apt_detection_model.pkl'.")
print("\n")
time.sleep(2)

# Demo data for prediction
print("Preparing demo data for prediction...")
time.sleep(2)
demo_data = {
    'attack_vector': ["Phishing", "Physical Access"],
    'target_system': ["Employee Workstation", "Web Server"],
    'vulnerability': ["Phishing", "Weak Password"]
}
print("Demo data prepared.")
print("\n")
time.sleep(2)

# Creating DataFrame from the demo data
print("Creating a DataFrame from the demo data...")
time.sleep(2)
demo_df = pd.DataFrame(demo_data)
print("DataFrame created.")
print("\n")
time.sleep(2)

# Print demo data
print("Demo data for prediction:")
print(demo_df)
print("\n")
time.sleep(2)

# Vectorizing textual features of demo data
print("Vectorizing textual features of demo data...")
time.sleep(2)
demo_vectorized = vectorizer.transform(demo_df.astype(str).apply(lambda x: ' '.join(x), axis=1))
print("Textual features of demo data vectorized.")
print("\n")
time.sleep(2)

# Predicting severity for demo data
print("Predicting severity for demo data...")
time.sleep(2)
predicted_severity = model.predict(demo_vectorized)
print("Prediction completed.")
print("\n")
time.sleep(2)

# Displaying the predicted severity for the demo data
print("Predicted severity for demo data:")
for i, severity in enumerate(predicted_severity):
    print(f"Data {i+1}: {severity}")
    time.sleep(1)

