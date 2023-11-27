import sqlite3

# Connect to the SQLite database
connection = sqlite3.connect('db_expenseAI_2.db')
cursor = connection.cursor()

# Use the following query to get the names of all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

# Fetch all the table names
table_names = cursor.fetchall()

# Print the table names
for name in table_names:
    print(name[0])

# Close the connection
connection.close()