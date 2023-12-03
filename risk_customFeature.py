import pandas as pd
import expenseNLP
import dataConverter_fromDB as db
import sqlite3
import spacy
from datetime import datetime



"""
This file will be extracting custom features based on the extracted information from the text of invoice from the expenseNLP.py file.

"""

######################placeholder to be updated once expenseNLP function is available.  Data are extracted based on expenseNLP.py file
'''
from extraction of invoice text
'''
dateExtracted = '02-02-2020'
amountExtracted = '37.1'
currencyExtracted = 'GBP'
invoiceNumberExtracted = '34344'
expenseCategoryExtracted = 'Travel'
'''
from user input
'''
input_invoiceAmount = '37.1'
submissionDate = '03-02-2020'
input_currency = 'GBP'
project_name = 'Project G'
employeeID = 'E000435'


#######################################################################################
#get reimbursement history date
df_reimbursementHistory = db.queryProjectEmployeesRelationFromDB()
df_projectEmployee = db.queryProjectEmployeesRelationFromDB()



#####################################################################################
"""
Custom Feature Extraction 1:  function to return feature value 0 or 1 (False or True) for whether the invoice date is on weekend or holiday.  
"""

def is_weekend(date_string):
    # Convert the date string to a datetime object
    invoice_date = datetime.datetime.strptime(date_string, '%Y-%m-%d').date()

    # Check if the day of the week is Saturday or Sunday (5 and 6 are Saturday and Sunday)
    if invoice_date.weekday() in [5, 6]:
        return True
    else:
        return False

def is_holiday(date_string):
    # Convert the date string to a datetime object
    invoice_date = datetime.datetime.strptime(date_string, '%Y-%m-%d').date()

    # Define holidays using the US calendar
    holidays = US(years=invoice_date.year)

    # Check if the invoice date is a holiday
    if invoice_date in holidays:
        return True
    else:
        return False
    
#feature 1. invoice date is Weekend or holiday?
def fea1_weekendOrHoliday(dateExtracted):
    is_weekend = 0
    if is_holiday(dateExtracted) or is_weekend(dateExtracted):
        is_weekend = 1
    return is_weekend



"""
Custom Feature Extraction 2: Define function to extract feature value of whether the user is Submitting the Same Invoice Multiple Times
"""    
def fea2_is_duplicate_invoice(extractedInvoiceDate, input_invoiceAmount , invoiceNumberExtracted, existing_reimbHistory_df):
    """
    Check if the current invoice is duplicated with any of the claims in the existing DataFrame.

    Parameters:
    - current_invoice: pd.Series or dictionary containing the current invoice data
    - existing_data_frame: pd.DataFrame containing the existing invoice reimbursement history

    Returns:
    - True if the current invoice is a duplicate, False otherwise
    """
    # Check for duplicates based on all three conditions
    duplicate_mask = (
        (existing_reimbHistory_df['InvoiceDate'] == extractedInvoiceDate) &
        (existing_reimbHistory_df['InvoiceID'] == invoiceNumberExtracted) &
        (existing_reimbHistory_df['InvoiceAmount'] == input_invoiceAmount)
    )

    # If any duplicates are found, return True
    if duplicate_mask.any():
        return 1
    else:
        return 0



'''
Custom Feature Extraction 3:  Unusual Spending?      check if the employee is on the claimed project
'''
def fea3_isEmployeeUnderProject(employeeID, project_name):
    """
    Check if an employee is under a certain project.

    Parameters:
    - db_path: str, path to the SQLite database file
    - employee_id: str, the employee ID to check
    - project_name: str, the project name to check

    Returns:
    - True if the employee is under the specified project, False otherwise
    """
    bool = 1
    # Get the DataFrame of projects under the employee's responsibility
    projects_under_employee_df = db.queryAllProjectsUnderCertainEmployee(employee_id = employeeID)

    # Check if the project_name is in the ProjectName column of the DataFrame
    if (project_name in projects_under_employee_df['PROJECTNAME'].values):
        bool = 0
        return 1
    
#testing
#result = isEmployeeUnderProject(employeeID=employeeID, project_name = project_name)
#print(result)

'''
Custom Feature Extraction 4:  Amount overclaimed?
return: 0 if no overclaim.   1 if input amount is greater than amount extracted from invoice.
'''
def fea4_amountIsOverclaimed(input_invoiceAmount, amountExtracted ):
    bool = 0
    variance = input_invoiceAmount - amountExtracted
    if variance > 0:
        bool = 1
    else: 
        bool = 0
    return bool

'''
Custom Feature Extraction 5:   if the amount is multiple of 100. 
'''
def fea5_is_invoice_amount_multiple_of_100(input_invoiceAmount):
    """
    Check if the invoice amount is a multiple of 100.

    Returns:
    - True if the invoice amount is a multiple of 100, False otherwise
    """
    bool = 0
    if input_invoiceAmount % 100 == 0:
        bool = 1
    return bool

# Test:
#invoice_amount_to_check = 1200
#result = fea5_is_invoice_amount_multiple_of_100(invoice_amount_to_check)


"""
Custom Feature Extraction 6: rounding amounts repeatedly claimed by same person?
"""

#helper function:
def isRoundingAmount(inputAmount):
    tolerance = 0.1   
    rounded_amount = round(float(inputAmount))
    difference = abs(float(inputAmount) - rounded_amount)
    return difference < tolerance
#feature to assess , if the current invoice amount is a rounding number, whether rounding amounts repeatedly claimed by same person over the last three claims from the same person. 
def fea6_has_repeated_rounding_numbers(employee_id, input_invoiceAmount):
    """
    Check if an employee has repeatedly claimed rounding numbers in the reimbursement history.

    Parameters:
    - employee_id: str, the employee ID to check
    - reimbursement_df: DataFrame, reimbursement history DataFrame

    Returns:
    - True if the employee has repeatedly claimed rounding numbers, False otherwise
    """
    bool = 0
    df_reimbursementHist_temp = db.queryReimbursementRequestRecordsFromDB()
    df_reimbursementHist_forEmployeeID = df_reimbursementHist_temp[df_reimbursementHist_temp['EmployeeID'] == employee_id]
    if isRoundingAmount(inputAmount=input_invoiceAmount):
        # Get the last five lines of the reimbursement data frame or all existing lines if there are fewer than five
        last_three_lines = df_reimbursementHist_forEmployeeID.tail(3)
        # Check if all invoice amounts in the last three lines (or existing lines) are rounding numbers
        if all(isRoundingAmount(amount) for amount in last_three_lines['InvoiceAmount']) is True:
            bool = 1
    return bool

"""
    Custom Feature Extraction 7:   Check if the text contains similar keywords related to personal expenses.
"""
def fea7_contains_personalExpense_keywords(textOfInvoice):
    
    nlp = spacy.load("en_core_web_md")

    # Define the keywords and their related words
    keyword_relations = {
        "grocery": ["groceries", "food", "shopping"],
        "supermarket": ["store", "shopping", "groceries"],
        "tuition": ["education", "school"],
        "rent": ["housing", "apartment", "lease", "leasing"],
        "housing": ["rent", "home", "residence"],
        "car": ["vehicle", "automobile", "auto", "vehicle"],
    }

    # Tokenize the text using spaCy
    doc = nlp(textOfInvoice.lower())

    # Check if any related keyword is present in the tokenized text
    for keyword, related_words in keyword_relations.items():
        if any(token.text == keyword or token.text in related_words for token in doc):
            return True

    return False
"""
    Custom Feature Extraction 8:   Check if the the current user input invoice amount has a big variance 
    with predicted invoice amount for the current date of claim  for this employee for the same expense 
    category.
"""
def fea8_suddenChangeInBehavior(inputInvoiceAmount):
    pass             # TODO



"""
    Custom Feature Extraction 9:   Check if Invoice Date falls outside of Project Duration Dates
    returns: 1 if the project does not fall into the project duration in the database.
             0 if the project falls into the duration of project
"""
def fea9_is_project_duration_covering(project_df, project_name, invoice_date):
    """
    Check if the project duration covers the provided project name based on the invoice date.

    Returns:
    - True if the project duration covers the provided project name based on the invoice date, False otherwise
    """
    bool = 0
    try:
        # Filter the project DataFrame for the specified project name
        project = project_df[project_df['PROJECTNAME'] == project_name].iloc[0]

        # Parse the project start and end dates to datetime objects
        project_start_date = datetime.strptime(project['ProjectDuration_startDate'], '%m/%d/%Y')
        project_end_date = datetime.strptime(project['ProjectDuration_endDate'], '%m/%d/%Y')
        invoice_datetime = datetime.strptime(invoice_date, '%m-%d-%Y')

        # Check if the invoice date falls between the project start and end date
        if invoice_datetime < project_start_date or invoice_datetime > project_end_date:
            bool = 1
    except Exception as e:
        # Handle exceptions (e.g., project name not found)
        print(f"Error: {e}")
    return bool