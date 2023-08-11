import pandas as pd
import io
from sklearn.ensemble import RandomForestClassifier
import time

logo = r"""
   /\_/\  
  ( o.o ) 
   > ^ <
milosilo.com
"""
print(logo)
# Comprehensive Explanation
print("=== Network Service and Version Detection using Machine Learning ===\n")
time.sleep(2)
print("This script demonstrates the use of machine learning models to predict the network service and version based on various network parameters.")
print("The script follows these steps:")
print("1. Data Cleaning and Preprocessing: The dataset is cleaned by removing unnecessary columns and converting categorical variables into numerical values using one-hot encoding.")
print("2. Model Training: Separate models are trained for service prediction and version prediction using the Random Forest Classifier algorithm.")
print("3. Predictions on the Demo Test Dataset: The trained models are applied to a demo test dataset to predict the service and version.")
print("4. Comparison with Expected Results: The predicted results are compared with the expected results to evaluate the accuracy of the models.\n")
time.sleep(15)

# Dataset containing network service signatures
data = """
Service Name,Product Name,Version,Protocol,Port Number,Packet Size,Packet Frequency,Data Encryption,Authentication Required,Max Connections,Max Bandwidth
HTTP,Apache,2.4.29,TCP,80,512 bytes,100 packets/second,No,Yes,1000,100 Mbps
HTTPS,nginx,1.16.1,TCP,443,1024 bytes,50 packets/second,Yes,Yes,500,50 Mbps
SSH,OpenSSH,7.6p1,TCP,22,2048 bytes,20 packets/second,Yes,Yes,100,10 Mbps
SMTP,Postfix,3.3.0,TCP,25,1024 bytes,10 packets/second,No,Yes,1000,5 Mbps
FTP,vsftpd,3.0.3,TCP,21,4096 bytes,5 packets/second,No,No,500,20 Mbps
DNS,BIND,9.11.3-1ubuntu1.15,UDP,53,256 bytes,200 packets/second,No,No,10000,1 Gbps
SNMP,Net-SNMP,5.7.3,UDP,161,512 bytes,50 packets/second,No,No,100,100 Mbps
RDP,Microsoft Terminal Services,10.0,TCP,3389,4096 bytes,30 packets/second,No,Yes,500,50 Mbps
SMTPS,Postfix,3.3.0,TCP,465,2048 bytes,10 packets/second,Yes,Yes,500,10 Mbps
POP3,Dovecot,2.3.4.1,TCP,110,1024 bytes,15 packets/second,No,Yes,1000,1 Mbps
"""

# Read the dataset into a Pandas DataFrame
df = pd.read_csv(io.StringIO(data))

# Data Cleaning and Preprocessing
print("=== Data Cleaning and Preprocessing ===\n")
time.sleep(2)

# Print the raw dataset
print("Raw dataset:")
print(df.head())
time.sleep(20)

# Split the dataset into features (X) and targets (y)
X = df.drop(['Service Name', 'Product Name', 'Version'], axis=1)
y_service = df['Service Name']
y_version = df['Version']

# Print the features (X) dataset after dropping the target columns
print("\nFeatures (X) dataset after dropping the target columns:")
print(X.head())
time.sleep(5)

# Print the target (service) dataset
print("\nTarget (Service) dataset:")
print(y_service.head())
time.sleep(5)

# Print the target (version) dataset
print("\nTarget (Version) dataset:")
print(y_version.head())
time.sleep(5)

# Convert categorical variables into numerical values using one-hot encoding
X_encoded = pd.get_dummies(X)

# Print the encoded features dataset after one-hot encoding
print("\nEncoded features after one-hot encoding:")
print(X_encoded.head())
time.sleep(5)

# Model training for service prediction
print("\n=== Model Training for Service Prediction ===\n")
time.sleep(2)

model_service = RandomForestClassifier(random_state=42)
model_service.fit(X_encoded, y_service)

# Model training for version prediction
print("\n=== Model Training for Version Prediction ===\n")
time.sleep(2)

model_version = RandomForestClassifier(random_state=42)
model_version.fit(X_encoded, y_version)

# Demo test dataset for prediction
demo_data = """
Protocol,Port Number,Packet Size,Packet Frequency,Data Encryption,Authentication Required,Max Connections,Max Bandwidth
TCP,465,2048 bytes,10 packets/second,Yes,Yes,500,10 Mbps
UDP,53,256 bytes,200 packets/second,No,No,10000,1 Gbps
TCP,3389,4096 bytes,30 packets/second,No,Yes,500,50 Mbps
"""

# Preprocess the demo test dataset
demo_df_encoded = pd.get_dummies(pd.read_csv(io.StringIO(demo_data)))

# Reorder the columns to match the order used during model training
demo_df_encoded = demo_df_encoded.reindex(columns=X_encoded.columns, fill_value=0)

# Display the demo test dataset
print("\nDemo Test Dataset:")
print(demo_df_encoded)
time.sleep(10)

# Predictions on the demo test dataset
print("\n=== Predictions for the Demo Test Dataset ===\n")
time.sleep(2)

demo_pred_service = model_service.predict(demo_df_encoded)
demo_pred_version = model_version.predict(demo_df_encoded)

# Retrieve product name and version based on the predicted service names
demo_results = pd.DataFrame({'Service Name': demo_pred_service})
demo_results['Product Name'] = demo_results['Service Name'].map(df.set_index('Service Name')['Product Name'])
demo_results['Version'] = demo_results['Service Name'].map(df.set_index('Service Name')['Version'])

# Expected predictions on the demo test dataset
expected_results = pd.DataFrame({
    'Service Name': ['SMTPS', 'DNS', 'RDP'],
    'Product Name': ['Postfix', 'BIND', 'Microsoft Terminal Services'],
    'Version': ['3.3.0', '9.11.3-1ubuntu1.15', '10.0']
})

# Print the expected predictions on the demo test dataset
print("Expected Demo Test Dataset Predictions:")
print(expected_results)
time.sleep(2)

# Print the predicted demo test dataset
print("\nPredicted Demo Test Dataset:")
print(demo_results)
time.sleep(2)

