import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import os
import re



"""
1.  Setting up data:    First,  Using the source file dataset.csv to create 'dataset' as a list of tuples where each tuple contains the text 
and expense category.  
"""
# Load the Excel file into a DataFrame and skip the header row
df = pd.read_csv('dataset_3labels.csv')

# Split the 'Source Data'column into 'Text of Invoice' , 'Expense Category' , 'Invoice Date', 'Invoice Amount', and 'Currency Unit' columns.  Using the ',' in the dataset as the separater, and split column into two from right to left for one time using rsplit.
df[['Text of Invoice,Expense Category,Invoice Date, Invoice Amount', 'Currency Unit']] = df['Source Data'].str.rsplit(",", expand=True, n=1)
df[['Text of Invoice,Expense Category,Invoice Date', 'Invoice Amount']] = df['Text of Invoice,Expense Category,Invoice Date, Invoice Amount'].str.rsplit(",", expand=True, n=1)
df[['Text of Invoice,Expense Category','Invoice Date']] = df['Text of Invoice,Expense Category,Invoice Date'].str.rsplit(",", expand=True, n=1)
df[['Text of Invoice','Expense Category']] = df['Text of Invoice,Expense Category'].str.rsplit(",", expand=True, n=1)

# Now we have df['Text of Invoice'], df['Expense Category'], df['Invoice Amount'], df[Currency Unit'].     
df_invoiceText = df['Text of Invoice']
df_expenseCategory = df['Expense Category']
df_invoiceAmount = df['Invoice Amount']
df_invoiceDate = df['Invoice Date']
df_currencyUnit = df['Currency Unit']

# For team members to work separately on data frames combining 'Text of Invoice' and 
    # any one of the 'Expenese Category', or 'Invoice Date', or 'Invoice Amount', 
    # We created data frames separately for each of the combination. 

# Create a list of tuples representing your dataset
dataset_invoiceText_expenseCategory = list(zip(df['Text of Invoice'].astype(str), df['Expense Category'].replace(' ','').astype(str)))
dataset_invoiceText_invoiceAmount = list(zip(df['Text of Invoice'].astype(str), df['Invoice Amount'].str.replace(' ', '').astype(str)))
dataset_invoiceText_invoiceDate = list(zip(df['Text of Invoice'].astype(str), df['Invoice Date'].str.replace(' ', '').astype(str)))
dataset_invoiceText_invoiceCurrency = list(zip(df['Text of Invoice'].astype(str), df['Currency Unit'].str.replace(' ', '').astype(str)))

#[(row['Text of Invoice'], row['Expense Category']) for _, row in df.iterrows()]