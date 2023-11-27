import csv
import sqlite3
import pandas as pd

'''
#add a Switch to control original database data 
'''
#（Note:  loadOriginalData cannot be True as if it's true then original 
# data from csv files will be inserted to database again causing 
# duplication )
database_path = 'db_expenseAI_2.db'
loadOriginalData_Employees = False    #data already in database. Delete db table rows before switching to True.
loadOriginalData_Projects = False     #data already in database. Delete db table rows before switching to True.
loadOriginalData_ReimbursementRequestRecords = False  #data already in database. Delete db table rows before switching to True.
loadOriginalData_ProjectEmployeesRelation = False    #data already in database. Delete db table rows before switching to True.
'''
Check number of raw in the csv file
'''
# Open the CSV file and count the rows
def checkNumRaws_csv(csv_file_path):
    with open(csv_file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        
        # Use enumerate to get both the row and its index
        for i, row in enumerate(csv_reader):
            pass  # Do nothing, just counting rows

    # Print the number of rows
    print(f'The number of rows in the CSV file is: {i + 1}')
# Drop the rows after a certain row
def dropRowAfterNumRow(df, row_number_to_keep): 
    df_Employees = df_Employees.drop(index=df_Employees.index[row_number_to_keep+1:])
    return df

def splitRowToColumns(pandasDF):
    # Store the original column name as a single string
    original_column = pandasDF.columns[0]

    # Split the original column name into multiple names
    original_column_names = original_column.split(',')

    # Split the original column values into multiple columns
    split_columns = pandasDF.iloc[:, 0].str.split(',', expand=True)

    # Assign the split columns back to the DataFrame with the new names
    split_columns.columns = original_column_names
    pandasDF = pd.concat([pandasDF, split_columns], axis=1)

    # Drop the original combined column
    pandasDF = pandasDF.drop(pandasDF.columns[0], axis=1)

    return pandasDF

'''
Specify path of the csv files that stores the fake initial 
data created by us in the database before any new invoice claims.
'''
# Provide path to the CSV file for original database start data
csv_file_ReimbursementRequestRecords_path = 'database_data/Table_ReimbursementRequestRecords.csv'
csv_file_Employees_path = 'database_data/Table_Employees.csv'
csv_file_Projects_path = 'database_data/Table_Projects.csv'
csv_file_ProjectEmployeesRelation_path = 'database_data/Table_ProjectEmployeesRelation.csv'
"""
Read CSV tables into DataFrames
"""
#data frame for table ReimbursementRequestRecords
if loadOriginalData_ReimbursementRequestRecords is True: 
    df_ReimbursementRequestRecords_toBeSplited = pd.read_csv(csv_file_ReimbursementRequestRecords_path, delimiter='\t', encoding='latin1')
    df_ReimbursementRequestRecords = splitRowToColumns(df_ReimbursementRequestRecords_toBeSplited)
#data frame for table Employees
if loadOriginalData_Employees is True:
    df_Employees_toBeSplited = pd.read_csv(csv_file_Employees_path, delimiter='\t', encoding='latin1')
    df_Employees = splitRowToColumns(df_Employees_toBeSplited)
#data frame for table Projects
if loadOriginalData_Projects is True:
    df_Projects_toBeSplited = pd.read_csv(csv_file_Projects_path, delimiter='\t', encoding='latin1')
    df_Projects = splitRowToColumns(df_Projects_toBeSplited)
#data frame for table ProjectEmployeesRelation
if loadOriginalData_ProjectEmployeesRelation is True:
    df_ProjectEmployeesRelation_toBeSplited = pd.read_csv(csv_file_ProjectEmployeesRelation_path, delimiter='\t', encoding='latin1')
    df_ProjectEmployeesRelation = splitRowToColumns(df_ProjectEmployeesRelation_toBeSplited)


"""
Define function to create connection to the database
"""
connection = None
# Function to connect to the database
def connect_to_database(database_path):
    global connection  # Use the global connection variable

    try:
        # Check if the connection is closed or not
        if connection and not connection.closed:
            print("Already connected to the database.")
        else:
            # If the connection is closed, establish a new connection
            connection = sqlite3.connect(database_path)
            print("Connected to the database.")
    except Exception as e:
        print(f"Error connecting to the database: {e}")


"""
Insert data frame data into database tables
"""

def insertDataFrameToDB(dbTableName, df, dbcon): 
    # Convert DataFrame to a list of tuples
    data_tuples = [tuple(row) for row in df.values if all(pd.notna(row))]
    print(data_tuples)
    # Create placeholders for SQL query
    placeholders = ','.join(['?' for _ in range(len(df.columns))])
    # Insert data into the existing table
    dbcon.executemany(f'INSERT INTO {dbTableName} VALUES ({placeholders})', data_tuples)
    print(f'Insertion Complete: inserted dataframe into database table {dbTableName}. ')

if loadOriginalData_ReimbursementRequestRecords is True:
    # Insert data frame data into database tables
    connect_to_database(database_path = database_path)
    dbcon = connection.cursor()
    dbTableName_ReimbursementRequestRecords = 'ReimbursementRequestRecords'
    insertDataFrameToDB(dbTableName_ReimbursementRequestRecords, df_ReimbursementRequestRecords, dbcon)
    connection.commit()
    connection.close()

if loadOriginalData_Employees is True:
    # Insert data frame data into database tables
    connect_to_database(database_path = database_path)
    dbcon = connection.cursor()
    dbTableName_Employees = 'Employees'
    insertDataFrameToDB(dbTableName_Employees, df_Employees, dbcon)
    connection.commit()
    connection.close()

if loadOriginalData_Projects is True:
    # Insert data frame data into database tables
    connect_to_database(database_path = database_path)
    dbcon = connection.cursor()
    dbTableName_Projects = 'Projects'
    insertDataFrameToDB(dbTableName_Projects, df_Projects, dbcon)
    connection.commit()
    connection.close()

if loadOriginalData_ProjectEmployeesRelation is True:
    # Insert data frame data into database tables
    connect_to_database(database_path = database_path)
    dbcon = connection.cursor()
    dbTableName_ProjectEmployeesRelation = 'ProjectEmployeesRelation'
    insertDataFrameToDB(dbTableName_ProjectEmployeesRelation, df_ProjectEmployeesRelation, dbcon)
    connection.commit()
    connection.close()








"""
Draft

def csv_to_db (csvPath, dbcon, dbTableName):
    # Open the CSV file and insert data into the table
   with open(csvPath,'r') as csvfile:
        # Create a CSV reader
        csv_reader = csv.DictReader(csvfile, delimiter='\t', encoding='latin1')
        
        # Iterate over rows in the CSV file and insert into the table
        for row in csv_reader:
            print(row)
            # Extract values from the row
            record_id = row['RecordID']
            employee_id = row['EmployeeID']
            project_id = row['ProjectID']
            expense_category = row['ExpenseCategory']
            submission_date = row['SubmissionDate']
            invoice_date = row['InvoiceDate']
            invoice_id = row['InvoiceID']
            invoice_amount = row['InvoiceAmount']

            # Insert into the table
            dbcon.execute(f'''
                INSERT INTO {dbTableName} (
                    RecordID, EmployeeID, ProjectID, ExpenseCategory,
                    SubmissionDate, InvoiceDate, InvoiceID, InvoiceAmount
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (record_id, employee_id, project_id, expense_category,
                submission_date, invoice_date, invoice_id, invoice_amount))


# import csv data to database

csv_file_ReimbursementRequestRecords_path = 'database_data/Table_ReimbursementRequestRecords.csv'
csv_to_db(csv_file_ReimbursementRequestRecords_path, dbcon, dbTableName = "ReimbursementRequestRecords")


# for deleting table content before replacing with new data
        #Execute the DELETE statement to remove all rows
        #cursor.execute(f'DELETE FROM {table_name}')

# Commit the changes
connection.commit()
"""