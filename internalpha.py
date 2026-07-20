import nltk
import string
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources (only first time)
nltk.download('punkt')
nltk.download('stopwords')

# ==========================
# FAQ DATASET
# ==========================

faq_data = {
    "Question": [
        "What are your working hours?",
        "How can I reset my password?",
        "How do I contact customer support?",
        "Where is your office located?",
        "Do you provide refunds?",
        "How can I track my order?",
        "Do you provide home delivery?",
        "What payment methods are accepted?",
        "Can I change my delivery address?",
        "How long does shipping take?"
    ],

    "Answer": [
        "Our working hours are Monday to Friday from 9 AM to 6 PM.",
        "Click on 'Forgot Password' on the login page.",
        "Email support@example.com or call +91-9876543210.",
        "Our office is located in Chennai, Tamil Nadu.",
        "Yes. Refunds are available within 7 days.",
        "Login and click on Track Order.",
        "Yes, we deliver across India.",
        "We accept UPI, Credit Card, Debit Card and Net Banking.",
        "Yes, before the order is shipped.",
        "Shipping usually takes 3-7 business days."
    ]
}

df = pd.DataFrame(faq_data)

# ==========================
# NLP PREPROCESSING
# ==========================

stop_words = set(stopwords.words("english"))

def preprocess(text):

    text = text.lower()

    text = text.translate(str.maketrans('', '', string.punctuation))

    words = word_tokenize(text)

    words = [word for word in words if word not in stop_words]

    return " ".join(words)

df["Processed"] = df["Question"].apply(preprocess)

# ==========================
# TF-IDF
# ==========================

vectorizer = TfidfVectorizer()

faq_vectors = vectorizer.fit_transform(df["Processed"])

# ==========================
# CHATBOT
# ==========================

print("="*60)
print("        🤖 FAQ CHATBOT")
print("="*60)
print("Type 'exit' to quit.")
print()

while True:

    user_input = input("You : ")

    if user_input.lower() == "exit":
        print("\nBot : Thank you! Have a great day.")
        break

    processed_input = preprocess(user_input)

    user_vector = vectorizer.transform([processed_input])

    similarity = cosine_similarity(user_vector, faq_vectors)

    best_match = similarity.argmax()

    score = similarity[0][best_match]

    if score < 0.25:
        print("\nBot : Sorry, I couldn't understand your question.\n")
    else:
        print(f"\nBot : {df.iloc[best_match]['Answer']}\n")