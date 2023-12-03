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



def queryAllProjectsUnderCertainEmployee(employee_id):
    """
    Get all projects under a certain employee's responsibility.

    Parameters:
    - db_path: str, path to the SQLite database file
    - employee_id: str, the employee ID to check

    Returns:
    - A DataFrame with ProjectID and ProjectName columns for projects under the employee's responsibility
    """
    # Connect to the SQLite database
    connection = sqlite3.connect(db_path)
    dbcon = connection.cursor()

    try:
        # Execute a query to get all projects under the employee's responsibility
        query = """
            SELECT P.ProjectID, P.ProjectName
            FROM ProjectEmployeesRelation AS PER
            JOIN Projects AS P ON PER.ProjectID = P.ProjectID
            WHERE PER.EmployeeID = ?
        """
        projects_df = pd.read_sql_query(query, connection, params=(employee_id,))

        return projects_df

    finally:
        # Close the database connection
        connection.close()




#Query dataframe from Employees table in database: 
queryEmployeesFromDB()
#Query dataframe from Projects table in database: 
queryProjectsFromDB()
#Query dataframe from ProjectEmployeesRelation table in database:  
queryProjectEmployeesRelationFromDB()
#Query datafram from ReimbursementRequestRecords table in database:
queryReimbursementRequestRecordsFromDB()