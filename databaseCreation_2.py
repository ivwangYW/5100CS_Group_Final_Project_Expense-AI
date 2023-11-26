import sqlite3

'''
DO NOT MODIFY
This file successfully created the database and db tables.

'''
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

#Connect to database
connect_to_database('db_expenseAI_2.db')
#Create a dbcon:
#A dbcon is used to execute SQL commands and fetch results.
dbcon = connection.cursor()

#create Projects table
dbcon.execute('''
    CREATE TABLE IF NOT EXISTS Projects (
        PROJECTID TEXT PRIMARY KEY,
        PROJECTNAME TEXT,
        ProjectDuration_startDate TEXT,
        ProjectDuration_endDate TEXT,
        ProjectBudget NUMBER
    )
    
''')

dbcon.execute('''
CREATE TABLE IF NOT EXISTS Employees (
        EmployeeID TEXT PRIMARY KEY,
        EmployeeName TEXT,
        EmployeeDepartment TEXT
    )
''')


dbcon.execute('''
CREATE TABLE IF NOT EXISTS ProjectEmployeesRelation (
        ProjectID TEXT,
        EmployeeID TEXT,
        FOREIGN KEY (ProjectID) REFERENCES Projects(ProjectID),
        FOREIGN KEY (EmployeeID) REFERENCES Employees(EmployeeID),
        PRIMARY KEY (ProjectID, EmployeeID)   
    )
''')
dbcon.execute('''
CREATE TABLE IF NOT EXISTS ReimbursementRequestRecords (
        RecordID TEXT PRIMARY KEY,
        EmployeeID TEXT,
        ProjectID TEXT,
        ExpenseCategory TEXT,
        SubmissionDate TEXT,
        InvoiceDate TEXT,
        InvoiceID TEXT,
        InvoiceAmount TEXT,

        FOREIGN KEY (ProjectID) REFERENCES Projects(ProjectID),
        FOREIGN KEY (EmployeeID) REFERENCES Employees(EmployeeID)
    )
''')

# Close the connection when done
if connection:
    connection.close()
    print("Connection closed.")