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
import expenseNLP_modeling as nlpMod



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




#Grid Search with Cross-Validation
grid_search = GridSearchCV(nlpMod.pipeline, nlpMod.param_grid, cv=5, verbose=1)
grid_search.fit(x_train, y_train)


"""
Store trained model to a new file
"""
# Save the trained model to a file
joblib.dump(grid_search.best_estimator_, 'trained_model.joblib')

"""
Model Accuracy Evaluation
"""
# Model Evaluation
accuracy = grid_search.score(x_test, y_test)
# Round to 2 decimal places and multiply by 100 to get percentage
accuracy_percent = round(accuracy * 100, 2)
print(f"Optimized Model Accuracy - for expense category classification: {accuracy_percent}")

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

# Apply the extraction function to source data frame of text that is in one column
extracted_data = df.apply(nlpMod.extract_details)   #data extracted and split into 4 columns
trained_model = joblib.load('trained_model.joblib')

# Create a new DataFrame from the extracted data
df_extracted = pd.DataFrame(extracted_data.tolist(), columns=['InvoiceNumber','Invoice Date', 'Invoice Amount', 'Currency Unit'])


df_extracted['Predicted Expense Category'] = trained_model.predict(df)

df_final = df_extracted[['Predicted Expense Category', 'InvoiceNumber','Invoice Date', 'Invoice Amount', 'Currency Unit']]

df_final.to_csv('processed_invoices.csv', index=False)



text = "#Summit-INV-20XHKD15 - Date of Issue: October 15, 2039 - Supplier: Summit Catering Services - Catering for company-wide summit, gourmet cuisine, and dessert buffet - Subamounts: $2,500.00 for catering - Tax: $250.00 - Total Amount: $2,750.00, Summit-INV-20XHKD15, Meals and Entertainment, 2039-10-15, 2750.00, USD"
#appInvoiceFinder(text, trained_model)
