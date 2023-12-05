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
from sklearn.preprocessing import LabelEncoder
import joblib  # Use joblib for model persistence
from dateutil.parser import parse
from io import StringIO 


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


def getProcessed_stop_words():
    processed_stop_words = [custom_preprocessor(word) for word in stopwords.words('english')]
    processed_stop_words = list(set(processed_stop_words))  # Removing duplicates

    return processed_stop_words
# Preprocess the stop words


# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(

    max_df=0.7,
    min_df=10,
    ngram_range=(1, 2),

    preprocessor=custom_preprocessor,
    stop_words=getProcessed_stop_words()  # Use the preprocessed stop words
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
##################################added by Ivy###############################################
#helper function to convert currency symbol to abbreviation.
def convert_currency_symbol_to_abbreviation(currency_symbol):
    currency_mapping = {
        'CA$': 'CAD',
        'A$': 'AUD',
        '€': 'EUR',
        '£': 'GBP',
        '$': 'USD',
        '¥': 'CNY',
        'HK$': 'HKD',
        '¥': 'JPY',
    }
    return currency_mapping.get(currency_symbol, 'Unknown')

# Test convert_currency_cymbol_to_abbreviation function:
#currency_symbol = '€'
#abbreviation = convert_currency_symbol_to_abbreviation(currency_symbol)
#print(f"{currency_symbol} -> {abbreviation}")


def NLP_getTextInfo(text, trained_NLP_model_path):
    '''
    Get extracted invoice number, invoice date, invoice amount, currency unit info in text format from the text.
    '''
    str_nlp_invoiceNumber, str_nlp_invoiceDate, str_nlp_invoiceAmount, str_nlp_currencyUnit  = extract_details(text)
    #convert currency symbol to abbreviation
    str_nlp_currencyUnit = convert_currency_symbol_to_abbreviation(str_nlp_currencyUnit)
    """
    get predicted expense category for the text.
    """
    #convert source text to dataframe to be ready for nlp processing
    df_text = pd.read_csv(StringIO(text))
    #convert df to nlp processed info in text format for all info elements
    
    #extracted_data = df.apply(extract_details)
    #load trained model for classifying expense category
    trained_model = joblib.load(trained_NLP_model_path)
    expenseCategory_predicted = trained_model.predict([text])[0]
    #le = LabelEncoder()
    #expenseCategory_predicted = le.inverse_transform([expenseCategory_predicted])[0]  # Inverse transform to get original category
    return str_nlp_invoiceNumber, str_nlp_invoiceDate, str_nlp_invoiceAmount, expenseCategory_predicted,  str_nlp_currencyUnit

#testing
#trained_NLP_model_path = 'trained_model.joblib'
#str_nlp_invoiceNumber, str_nlp_invoiceDate, str_nlp_invoiceAmount, expenseCategory_predicted,  str_nlp_currencyUnit = NLP_getTextInfo("#Summit-INV-20XHKD15 - Date of Issue: October 15, 2039 - Supplier: Summit Catering Services - Catering for company-wide summit, gourmet cuisine, and dessert buffet - Subamounts: $2,500.00 for catering - Tax: $250.00 - Total Amount: $2,750.00, Summit-INV-20XHKD15", trained_NLP_model_path)