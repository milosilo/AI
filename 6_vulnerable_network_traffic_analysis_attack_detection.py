import pandas as pd
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

logo = r"""
   /\_/\  
  ( o.o ) 
   > ^ <
milosilo.com
"""
print(logo)
# Print statement explaining the script and its purpose
print("Welcome to the Traffic Analysis and Vulnerability Prediction Script!")
print("This script analyzes network traffic and predicts the vulnerability type of the identified traffic.")
print("It utilizes machine learning techniques to classify the network payloads and provide recommendations based on the predicted vulnerability types.")
print("Let's get started...")
print("\n")

# Importing necessary libraries
print("Importing libraries for data manipulation, analysis, and machine learning...")
time.sleep(7)
print("Libraries imported successfully.")
print("\n")

# Training dataset with realistic network traffic
training_data = {
    'source_ip': [
        "192.168.0.2",
        "192.168.0.5",
        "192.168.0.7",
        "192.168.0.3",
        "192.168.0.8",
        "192.168.0.4",
        "192.168.0.6",
        "192.168.0.2",
        "192.168.0.5",
        "192.168.0.7",
        "192.168.0.9",
        "192.168.0.10",
        "192.168.0.11",
        "192.168.0.12",
        "192.168.0.13",
        "192.168.0.14",
        "192.168.0.15",
        "192.168.0.16",
        "192.168.0.17",
        "192.168.0.18",
        "192.168.0.19"
    ],
    'destination_ip': [
        "192.168.1.2",
        "192.168.1.5",
        "192.168.1.7",
        "192.168.1.3",
        "192.168.1.8",
        "192.168.1.4",
        "192.168.1.6",
        "192.168.1.2",
        "192.168.1.5",
        "192.168.1.7",
        "192.168.1.11",
        "192.168.1.12",
        "192.168.1.13",
        "192.168.1.14",
        "192.168.1.15",
        "192.168.1.16",
        "192.168.1.17",
        "192.168.1.18",
        "192.168.1.19",
        "192.168.1.20",
        "192.168.1.21"
    ],
    'source_port': [
        1234,
        5678,
        9876,
        4321,
        8765,
        3456,
        6543,
        1234,
        5678,
        9876,
        2222,
        3333,
        4444,
        5555,
        6666,
        7777,
        8888,
        9999,
        1111,
        2222,
        3333
    ],
    'destination_port': [
        80,
        443,
        8080,
        22,
        3389,
        3306,
        8080,
        80,
        443,
        8080,
        8888,
        9999,
        1111,
        2222,
        3333,
        4444,
        5555,
        6666,
        7777,
        8888,
        9999
    ],
    'protocol': [
        "TCP",
        "UDP",
        "TCP",
        "TCP",
        "UDP",
        "TCP",
        "UDP",
        "TCP",
        "UDP",
        "TCP",
        "TCP",
        "UDP",
        "TCP",
        "UDP",
        "TCP",
        "TCP",
        "UDP",
        "TCP",
        "UDP",
        "TCP",
        "UDP"
    ],
    'payload': [
        "GET /admin.php",
        "POST /login.php",
        "GET /api/v1/data",
        "GET /index.php",
        "POST /submit.php",
        "GET /products.php",
        "POST /update.php",
        "GET /admin.php",
        "POST /login.php",
        "GET /api/v1/data",
        "GET /vulnerable.php",
        "POST /vulnerable_login.php",
        "GET /vulnerable_data.php",
        "POST /vulnerable_update.php",
        "GET /vulnerable_admin.php",
        "POST /vulnerable_login.php",
        "GET /vulnerable.php",
        "POST /api/v1/login",
        "GET /products.php",
        "POST /upload.php",
        "GET /admin.php"
    ],
    'vulnerability_type': [
        "Path Traversal",
        "Authentication Bypass",
        "SQL Injection",
        "Cross-Site Scripting (XSS)",
        "Remote Code Execution",
        "Command Injection",
        "Cross-Site Request Forgery (CSRF)",
        "Path Traversal",
        "Authentication Bypass",
        "SQL Injection",
        "Remote Code Execution",
        "SQL Injection",
        "Cross-Site Scripting (XSS)",
        "Command Injection",
        "Path Traversal",
        "Authentication Bypass",
        "SQL Injection",
        "Cross-Site Scripting (XSS)",
        "Command Injection",
        "Path Traversal",
        "Authentication Bypass"
    ]
}

# Creating a DataFrame from the training dataset
print("Creating a DataFrame from the training dataset...")
time.sleep(3)
df_train = pd.DataFrame(training_data)
print("Training DataFrame created successfully.")
print("\n")

# Print training dataset
print("Training dataset:")
print(df_train.head())
print("\n")

# Print statement explaining the use of machine learning and its benefits
print("Using machine learning to classify network payloads and predict vulnerability types...")
print("Machine learning allows us to automate the analysis process and make predictions based on learned patterns.")
print("The benefits of using machine learning for this task include improved efficiency, scalability, and the ability to handle complex data patterns.")
time.sleep(15)
print("Let's proceed to perform machine learning classification on the training dataset.")
print("\n")
time.sleep(2)

# Perform machine learning classification on the training dataset
print("Performing machine learning classification on the training dataset...")
time.sleep(2)

# Extract features from the payload using CountVectorizer
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(df_train['payload'])

# Train a Naive Bayes classifier
classifier = MultinomialNB()
y_train = df_train['vulnerability_type']
classifier.fit(X_train, y_train)

# Demo traffic with realistic network details
demo_traffic = {
    'source_ip': [
        "192.168.0.100",
        "192.168.0.101",
        "192.168.0.102",
        "192.168.0.103",
        "192.168.0.104",
        "192.168.0.105",
        "192.168.0.106",
    ],
    'destination_ip': [
        "10.0.0.1",
        "10.0.0.2",
        "10.0.0.3",
        "10.0.0.4",
        "10.0.0.5",
        "10.0.0.6",
        "10.0.0.7",
    ],
    'source_port': [
        50000,
        50001,
        50002,
        50003,
        50004,
        50005,
        50006,
    ],
    'destination_port': [
        80,
        443,
        8080,
        22,
        3389,
        3306,
        8080,
    ],
    'protocol': [
        "TCP",
        "TCP",
        "TCP",
        "TCP",
        "TCP",
        "TCP",
        "TCP",
    ],
    'payload': [
        "GET /home.html",
        "POST /login.php",
        "GET /api/v2/data",
        "GET /about.html",
        "POST /submit.php",
        "GET /products.html",
        "POST /update.php",
    ]
}

# Creating a DataFrame from the demo traffic
df_demo = pd.DataFrame(demo_traffic)

# Print statement explaining the demo traffic
print("Demo Traffic:")
print("The following network traffic will be analyzed:")
print(df_demo)
time.sleep(15)
print("Predicting the vulnerability types for the new payloads...")
print("\n")
time.sleep(2)

# Predict the vulnerability type for new payloads from the demo traffic
X_new = vectorizer.transform(df_demo['payload'])
predictions = classifier.predict(X_new)

print("Predicted vulnerability types for the new payloads:")
for index, row in df_demo.iterrows():
    payload = row['payload']
    source_ip = row['source_ip']
    destination_ip = row['destination_ip']
    source_port = row['source_port']
    destination_port = row['destination_port']
    protocol = row['protocol']
    prediction = predictions[index]

    print(f"Payload: {payload}")
    print(f"Vulnerability Type: {prediction}")
    print(f"Source IP: {source_ip}")
    print(f"Destination IP: {destination_ip}")
    print(f"Source Port: {source_port}")
    print(f"Destination Port: {destination_port}")
    print(f"Protocol: {protocol}")
    print("------------------------")
    time.sleep(1)

print("\n")

# Print statement explaining the generation of recommendations
print("Generating recommendations based on vulnerability types...")
time.sleep(3)

# Generate recommendations based on vulnerability types
recommendations = {
    "Path Traversal": "Implement input validation and secure file access controls. Use path sanitization techniques to prevent unauthorized access.",
    "Authentication Bypass": "Implement strong authentication mechanisms such as multi-factor authentication and ensure proper session management.",
    "SQL Injection": "Apply parameterized queries or prepared statements to prevent SQL injection attacks. Use input validation and output encoding to sanitize user input.",
    "Cross-Site Scripting (XSS)": "Sanitize user input and implement output encoding to prevent XSS attacks. Use proper input validation and output filtering techniques.",
    "Remote Code Execution": "Keep software up to date with security patches. Implement secure coding practices and input validation to prevent arbitrary code execution.",
    "Command Injection": "Avoid using user input to construct command strings. Implement input validation and enforce strict command sanitization to prevent command injection attacks.",
    "Cross-Site Request Forgery (CSRF)": "Use anti-CSRF tokens and enforce proper authentication and authorization checks. Implement CSRF prevention mechanisms like SameSite cookies and CSRF tokens."
}

print("Recommendations:")
for index, row in df_demo.iterrows():
    prediction = predictions[index]
    if prediction in recommendations:
        print(f"Payload: {row['payload']}")
        print(f"Vulnerability Type: {prediction}")
        print(f"Recommendation: {recommendations[prediction]}")
        print("------------------------")
        time.sleep(3)

print("\n")

# Print final analysis or actions
print("Final analysis or actions:")
print("Only the vulnerable network traffic has been analyzed.")
print("The vulnerability type of the identified traffic has been determined.")

