import csv
import sqlite3
import pandas as pd
from sqlalchemy import create_engine

# Replace 'your_file.csv' with the actual path to your CSV file
csv_file_ReimbursementRequestRecords_path = 'database_data/Table_ReimbursementRequestRecords.csv'

# Read CSV into DataFrame
df_ReimbursementRequestRecords = pd.read_csv(csv_file_ReimbursementRequestRecords_path, delimiter='\t', encoding='latin1')

# Connect to the SQLite database using SQLAlchemy engine
engine = create_engine('sqlite:///db_expenseAI.db')

# Replace 'ReimbursementRequestRecords' with the desired table name
dbTableName = 'ReimbursementRequestRecords'

# Insert data into an existing SQLite table (append if table already exists)
df_ReimbursementRequestRecords.to_sql(dbTableName, engine, index=False, if_exists='append')

# Close the connection
engine.dispose()

"""
import csv
import sqlite3
import pandas as pd
from sqlalchemy import create_engine

# Replace 'your_file.csv' with the actual path to your CSV file
csv_file_ReimbursementRequestRecords_path = 'database_data/Table_ReimbursementRequestRecords.csv'

# Read CSV into DataFrame
df_ReimbursementRequestRecords = pd.read_csv(csv_file_ReimbursementRequestRecords_path, delimiter='\t', encoding='latin1')

# Rename DataFrame columns to match the expected table columns
df_ReimbursementRequestRecords = df_ReimbursementRequestRecords.rename(columns={
    'RecordID': 'RecordID',
    'EmployeeID': 'EmployeeID',
    'ProjectID': 'ProjectID',
    'ExpenseCategory': 'ExpenseCategory',
    'SubmissionDate': 'SubmissionDate',
    'InvoiceDate': 'InvoiceDate',
    'InvoiceID': 'InvoiceID',
    'InvoiceAmount': 'InvoiceAmount'
})

# Connect to the SQLite database using SQLAlchemy engine
engine = create_engine('sqlite:///db_expenseAI.db')

# Replace 'ReimbursementRequestRecords' with the desired table name
dbTableName = 'ReimbursementRequestRecords'

# Insert data into an existing SQLite table (append if table already exists)
df_ReimbursementRequestRecords.to_sql(dbTableName, engine, index=False, if_exists='append')

# Close the connection
engine.dispose()
"""