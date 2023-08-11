import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import time

logo = r"""
   /\_/\  
  ( o.o ) 
   > ^ <
milosilo.com
"""
print(logo)
print("Machine Learning Demo - Cyber Security Question Answer FAQ")

# Importing necessary libraries and classes
print("Importing libraries...")
time.sleep(2)
print("Importing pandas for data manipulation and analysis...")
time.sleep(2)
print("Importing nltk for natural language processing tasks such as tokenization...")
time.sleep(2)
print("Importing TfidfVectorizer for converting text data into numerical vectors based on term frequency-inverse document frequency...")
time.sleep(2)
print("Importing RandomForestClassifier for training a machine learning model based on decision trees ensemble...")
time.sleep(2)
print("Importing train_test_split for splitting the dataset into training and testing sets...")
time.sleep(2)
print("Libraries imported.")
print("\n")

# Example FAQ dataset
data = {
    'question': [
        "What is a firewall?",
        "How can I protect my computer from malware?",
        "What are common types of cyber attacks?",
        "How does encryption work?",
        "What should I do if my email account is hacked?",
        "What is a VPN?",
        "How do I secure my wireless network?",
        "What is social engineering?",
        "What is two-factor authentication?",
        "How can I create strong passwords?",
        "What is data encryption?",
        "How do I detect phishing emails?",
        "What is a DDoS attack?",
        "How does a password manager work?",
        "What is endpoint protection?",
        "What is network monitoring?",
        "How can I secure my mobile device?",
        "What is a vulnerability assessment?",
        "How do I report a cybercrime?",
        "What is secure coding?"
    ],
    'answer': [
        "A firewall is a network security device that monitors and filters incoming and outgoing network traffic.",
        "To protect your computer from malware, you should regularly update your antivirus software and avoid clicking on suspicious links or downloading files from untrusted sources.",
        "Common types of cyber attacks include phishing, malware attacks, DDoS attacks, and ransomware.",
        "Encryption is the process of converting plaintext into ciphertext to secure sensitive data during transmission or storage.",
        "If your email account is hacked, you should immediately change your password, enable two-factor authentication, and scan your computer for malware.",
        "A VPN, or Virtual Private Network, is a secure connection that allows you to access the internet privately and securely.",
        "To secure your wireless network, you should change the default router password, enable network encryption (e.g., WPA2), and disable remote administration.",
        "Social engineering is a method used by attackers to manipulate individuals into divulging confidential information or performing certain actions.",
        "Two-factor authentication adds an extra layer of security by requiring users to provide two forms of identification, typically a password and a verification code.",
        "To create strong passwords, use a combination of uppercase and lowercase letters, numbers, and special characters. Avoid using easily guessable information, such as your name or birthdate.",
        "Data encryption is the process of converting data into a format that is unreadable by unauthorized users, ensuring its confidentiality.",
        "To detect phishing emails, check for suspicious email addresses, grammar and spelling errors, and avoid clicking on links or downloading attachments from unfamiliar sources.",
        "A DDoS attack, or Distributed Denial of Service attack, is an attempt to disrupt the normal functioning of a network, service, or website by overwhelming it with a flood of internet traffic.",
        "A password manager is a tool that securely stores and manages passwords for various online accounts, eliminating the need to remember multiple passwords.",
        "Endpoint protection refers to the security measures implemented to protect endpoints, such as computers, smartphones, and tablets, from cyber threats.",
        "Network monitoring involves observing and analyzing network traffic to detect and respond to any suspicious or malicious activities.",
        "To secure your mobile device, use a strong passcode or biometric authentication, keep your software up to date, and only download apps from trusted sources.",
        "A vulnerability assessment is a systematic evaluation of potential vulnerabilities in a system, network, or application to identify security weaknesses.",
        "To report a cybercrime, contact your local law enforcement agency and provide them with as much information as possible, including any evidence or suspicious activities.",
        "Secure coding involves following best practices and guidelines to develop software with built-in security measures, minimizing the risk of vulnerabilities and exploits."
    ]
}

# Creating a DataFrame from the dataset
print("Creating a DataFrame from the dataset...")
time.sleep(2)
df = pd.DataFrame(data)
print("DataFrame created.")
print("\n")

# Print original FAQ dataset
print("Original FAQ dataset:")
print(df.head())
print("\n")

# Selecting relevant columns
print("Selecting relevant columns for analysis...")
time.sleep(2)
df = df[['question', 'answer']]
print("Relevant columns selected.")
print("\n")

# Lowercase conversion
print("Converting text to lowercase for consistency...")
time.sleep(2)
df['question'] = df['question'].str.lower()
print("Text converted to lowercase.")
print("\n")

# Print dataset after lowercase conversion
print("Dataset after lowercase conversion:")
print(df.head())
time.sleep(4)
print("\n")

# Tokenization
print("Tokenizing the text to split it into individual words or tokens...")
time.sleep(2)
df['tokens'] = df['question'].apply(word_tokenize)
print("Text tokenized.")
print("\n")

# Print dataset after tokenization
print("Dataset after tokenization:")
print(df.head())
time.sleep(4)
print("\n")

# Remove stopwords
print("Removing stopwords to eliminate common words that do not carry much information...")
time.sleep(2)
stop_words = set(stopwords.words('english'))
df['filtered_tokens'] = df['tokens'].apply(lambda tokens: [token for token in tokens if token not in stop_words])
print("Stopwords removed.")
print("\n")

# Print dataset after stopwords removal
print("Dataset after stopwords removal:")
print(df.head())
time.sleep(4)
print("\n")

# Vectorization
print("Vectorizing the text to convert it into numerical representation using TF-IDF (Term Frequency-Inverse Document Frequency)...")
time.sleep(2)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['filtered_tokens'].apply(lambda tokens: ' '.join(tokens)))
print("Text vectorized.")
print("\n")

# Splitting into training and testing sets
print("Splitting the dataset into training and testing sets to evaluate the model's performance...")
time.sleep(2)
X_train, X_test, y_train, y_test = train_test_split(X, df['answer'], test_size=0.2, random_state=42)
print("Dataset split into training and testing sets.")
print("\n")

# Training
print("Training the model using RandomForestClassifier...")
print("RandomForestClassifier was chosen because it is an ensemble model based on decision trees, which can handle high-dimensional datasets and capture complex relationships.")
print("It is also less prone to overfitting compared to individual decision trees.")
time.sleep(2)
model = RandomForestClassifier()
model.fit(X_train, y_train)
print("Model fitted using the training data.")
print("\n")

# Evaluation
print("Evaluating the model's accuracy on the test set...")
time.sleep(2)
accuracy = model.score(X_test, y_test)
print("Model accuracy on the test set:", accuracy)
print("\n")

# Demonstration
demo_question = "How can I protect my online privacy?"
print("=" * 50)
print(" " * 15 + "Demonstration")
print("=" * 50)
print("\n")
print("Demo Question:", demo_question)
print("\n")

# Preprocess demo question
print("Preprocessing the demo question...")
time.sleep(2)
demo_question = demo_question.lower()
demo_tokens = word_tokenize(demo_question)
demo_filtered_tokens = [token for token in demo_tokens if token not in stop_words]
demo_input_vector = vectorizer.transform([' '.join(demo_filtered_tokens)])
print("Demo question preprocessed.")
print("\n")

# Predict answer for demo question
print("Predicting the answer for the demo question...")
time.sleep(2)
demo_predicted_answer = model.predict(demo_input_vector)[0]
print("Answer predicted.")
print("\n")

# Print predicted answer for demo question
print("-" * 50)
print("\n")
print("Predicted Answer:", demo_predicted_answer)
print("\n")
print("-" * 50)
print("\n")

