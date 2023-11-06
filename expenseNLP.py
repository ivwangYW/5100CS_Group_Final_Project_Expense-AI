import pandas as pd

# Load the Excel file into a DataFrame and skip the header row
df = pd.read_csv('dataset.csv')

# Split the 'Source Data'column into 'Text of Invoice' and 'Expense Category' columns
df[['Text of Invoice', 'Expense Category']] = df['Source Data'].str.split(',', 1, expand=True)

# Create a list of tuples representing your dataset
dataset = [(row['Text of Invoice'], row['Expense Category']) for _, row in df.iterrows()]

# Now, 'dataset' is a list of tuples where each tuple contains the text and expense category

# You can use this 'dataset' for data processing and machine learning