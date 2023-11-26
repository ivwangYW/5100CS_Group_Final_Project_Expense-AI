import pandas as pd
import sqlite3

# Connect to the SQLite database
db_path = 'db_expenseAI_2.db'  # Replace with the actual path to your SQLite database file


# Specify the table name you want to retrieve
table_Employees = 'Employees' 
table_Projects = 'Projects'
table_ProjectEmployeesRelation = 'ProjectEmployeesRelation'
table_ReimbursementRequestRecords = 'ReimbursementRequestRecords'

def queryEmployeesFromDB():
    connection = sqlite3.connect(db_path)
    # Construct the SQL query to select all rows from the table
    query = f"SELECT * FROM {table_Employees}"

    # Use pd.read_sql_query to fetch the data into a DataFrame
    df = pd.read_sql_query(query, connection)

    # Print the DataFrame
    #print(df)

    # Close the connection
    connection.close()
    return df

def queryProjectsFromDB():
    connection = sqlite3.connect(db_path)
    # Construct the SQL query to select all rows from the table
    query = f"SELECT * FROM {table_Projects}"

    # Use pd.read_sql_query to fetch the data into a DataFrame
    df = pd.read_sql_query(query, connection)

    # Print the DataFrame
    #print(df)

    # Close the connection
    connection.close()
    return df

def queryProjectEmployeesRelationFromDB():
    connection = sqlite3.connect(db_path)
    # Construct the SQL query to select all rows from the table
    query = f"SELECT * FROM {table_ProjectEmployeesRelation}"

    # Use pd.read_sql_query to fetch the data into a DataFrame
    df = pd.read_sql_query(query, connection)

    # Print the DataFrame
    #print(df)

    # Close the connection
    connection.close()
    return df

def queryReimbursementRequestRecordsFromDB():
    connection = sqlite3.connect(db_path)
    # Construct the SQL query to select all rows from the table
    query = f"SELECT * FROM {table_ReimbursementRequestRecords}"

    # Use pd.read_sql_query to fetch the data into a DataFrame
    df = pd.read_sql_query(query, connection)

    # Print the DataFrame
    #print(df)

    # Close the connection
    connection.close()
    return df

print('Employees table: ')
queryEmployeesFromDB()
print('projects table: ')
queryProjectsFromDB()
print('ProjectEmployeesRelation table: ')
queryProjectEmployeesRelationFromDB()
print('ReimbursementRequestRecords table: ')
queryReimbursementRequestRecordsFromDB()