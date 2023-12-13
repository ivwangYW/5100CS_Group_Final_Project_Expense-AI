import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import os
import re
import dataConverter
import spacy

import re
from dateutil.parser import parse

# Ensure necessary NLTK resources are available
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Obtain dataset and DataFrame from dataConverter.py
dataset = dataConverter.dataset_invoiceText_expenseCategory
df_expenseCategory = dataConverter.df_expenseCategory


import joblib  # Use joblib for model persistence

from sklearn.preprocessing import LabelEncoder



# Label Encoding
le = LabelEncoder()
df_expenseCategory['Expense Category'] = le.fit_transform(df_expenseCategory)

# Split the dataset into training and testing sets
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

# Unpack training and testing data
x_train, y_train = zip(*train_set)
x_test, y_test = zip(*test_set)
x_train, y_train, x_test, y_test = list(x_train), list(y_train), list(x_test), list(y_test)

# Custom Preprocessor Function
def custom_preprocessor(text):
    custom_stopwords = set(stopwords.words('english'))
    text = re.sub(r'\d+', '', text)  # Remove numbers
    tokenized_text = word_tokenize(text.lower())
    tokenized_text = [token for token in tokenized_text if token.isalpha()]
    lemmatizer = WordNetLemmatizer()
    tokenized_text = [lemmatizer.lemmatize(token) for token in tokenized_text]
    tokenized_text = [token for token in tokenized_text if token not in custom_stopwords]
    return ' '.join(tokenized_text)

# Preprocess the stop words
processed_stop_words = [custom_preprocessor(word) for word in stopwords.words('english')]
processed_stop_words = list(set(processed_stop_words))  # Removing duplicates

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(

    max_df=0.7,
    min_df=10,
    ngram_range=(1, 2),

    preprocessor=custom_preprocessor,
    stop_words=processed_stop_words  # Use the preprocessed stop words
)



# Define Pipeline with Hyperparameter Tuning
pipeline = Pipeline([
    ('tfidf', vectorizer),
    ('clf', SVC())
])

# Hyperparameter Grid
param_grid = {
    'clf__C': [0.1, 1, 10],
    'clf__gamma': [0.001, 0.01, 0.1, 1],
    'clf__kernel': ['linear', 'rbf', 'poly']
}

#Grid Search with Cross-Validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=1)
grid_search.fit(x_train, y_train)
# Save the trained model to a file
joblib.dump(grid_search.best_estimator_, 'trained_model.joblib')
# Model Evaluation
accuracy = grid_search.score(x_test, y_test)
print(f"Optimized Model Accuracy: {accuracy}")

# # Predicting New Invoice Category
# print('Please input text of invoice for classification: ')
# one_text_of_invoice = input()
# if one_text_of_invoice.isalpha():
#     answer = grid_search.predict([one_text_of_invoice])[0]
#     answer = le.inverse_transform([answer])[0]  # Inverse transform to get original category
#     print(f'Expense Category: {answer}\n')



##Date training.
dataset = dataConverter.dataset_invoiceText_invoiceDate
df_invoiceDate = dataConverter.df_invoiceDate


# Load dataset
df = dataConverter.df_invoiceText

# Function to extract date, total amount, and currency unit
def extract_details(text):

    invoice_number_pattern = r'\b(?:\w+[/\-_])*\w+[/\-_]\w+\b'
    invoice_number_match = re.search(invoice_number_pattern, text)
    invoice_number = invoice_number_match.group(0) if invoice_number_match else None
    # Regex pattern for date (YYYY-MM-DD format)

    # Regex for currency symbols/codes
    currency_pattern = r'\$|€|£|USD|EUR|GBP'
    # Regex for monetary values
    amount_pattern = r'(\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?|€\d{1,3}(?:,\d{3})*(?:\.\d{2})?|£\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'

    date = None
    date_candidates = re.findall(r'\d{4}-\d{1,2}-\d{1,2}|\b(?:January|Jan|February|Feb|March|Mar|April|Apr|May|June|July|August|September|Sep|October|November|December)\b \d{1,2}, \d{4}|\d{1,2}/\d{1,2}/\d{4}', text)
    if date_candidates:
        # Attempt to parse the first date candidate found
        try:
            parsed_date = parse(date_candidates[0], fuzzy_with_tokens=False)
            date = parsed_date.strftime('%Y-%m-%d')  # Format the date as needed
        except ValueError:
            pass  # If parsing fails, keep date as None    
    
    currency_match = re.search(currency_pattern, text)
    amounts = re.findall(amount_pattern, text)

    amounts = [float(amount[1:].replace(',', '')) for amount in amounts]
    largest_amount = max(amounts, default=None)


    currency = currency_match.group(0) if currency_match else None
    amount = largest_amount

    return invoice_number, date, amount, currency

# Apply the extraction function
extracted_data = df.apply(extract_details)
trained_model = joblib.load('trained_model.joblib')

# Create a new DataFrame from the extracted data
df_extracted = pd.DataFrame(extracted_data.tolist(), columns=['InvoiceNumber','Invoice Date', 'Invoice Amount', 'Currency Unit'])


df_extracted['Predicted Expense Category'] = trained_model.predict(df)

df_final = df_extracted[['Predicted Expense Category', 'InvoiceNumber','Invoice Date', 'Invoice Amount', 'Currency Unit']]

df_final.to_csv('processed_invoices.csv', index=False)



def appInvoiceFinder(text, trained_model):
        InvoiceId,InvoiceDate, InvoiceAmount, InvoiceCurrency = extract_details(text)
        print(f'InvoiceId :{InvoiceId}\n')
        print(f'InvoiceDate : {InvoiceDate}\n')
        print(f'InvoiceAmount :{ InvoiceAmount}\n')
        print(f'InvoiceCurrency :{ InvoiceCurrency}\n')

        ExpensiveCategory = trained_model.predict([text])  # Inverse transform to get original category
        #decoded_category = le.inverse_transform(ExpensiveCategory)

        print(f'InvoiceExpenseCategory{ExpensiveCategory[0]}\n')
        return   InvoiceId, InvoiceDate, InvoiceAmount, InvoiceCurrency,ExpensiveCategory




text = "#Summit-INV-20XHKD15 - Date of Issue: October 15, 2039 - Supplier: Summit Catering Services - Catering for company-wide summit, gourmet cuisine, and dessert buffet - Subamounts: $2,500.00 for catering - Tax: $250.00 - Total Amount: $2,750.00, Summit-INV-20XHKD15, Meals and Entertainment, 2039-10-15, 2750.00, USD"
#appInvoiceFinder(text, trained_model)



df2 = 'processed_invoices.csv'
df_extractd = pd.read_csv(df2,encoding='latin1')



df = dataConverter.df_invoiceDate.str.strip()
df_date = df_extractd['Invoice Date']


accuracy = (df_date == df).mean()
print(f"Accuracy for Date: {accuracy:.2f}")



