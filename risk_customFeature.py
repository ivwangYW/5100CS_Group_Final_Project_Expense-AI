import pandas as pd
#import expenseNLP
import dataConverter_fromDB as db
import sqlite3
import spacy
import datetime
from dateutil import parser
import predict_future_InvoiceAmount as forcast
#from workalendar.usa import UnitedStates



"""
This file will be extracting custom features based on the extracted information from the text of invoice from the expenseNLP.py file.

"""

######################placeholder for testing.  Data are extracted based on expenseNLP.py file
'''
from extraction of invoice text
'''
dateExtracted = '2020-02-02'     #TODO      if invoice date comes in not as YYYY-MM-DD, can revise code parse_invoice_date(date_string) to be added to the date conversion section to convert date to date format
amountExtracted = '37.1'
currencyExtracted = 'GBP'       #TODO     probably come in as currency symbol, need conversion code to convert symbol to english abbreviation
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

################################ helper functions #####################################################
# In case where extracted invoice date are not in  format MM-DD-YYYY
def parse_invoice_date(date_string):
    # If date_string is already a datetime.date object, no need to parse, just return the datetime.date object.
    if isinstance(date_string, datetime.date):
        return date_string
    else:
        potential_formats = [
            '%Y-%m-%d',  # YYYY-MM-DD
            '%m/%d/%Y',  # MM/DD/YYYY
            '%d/%m/%Y',  # DD/MM/YYYY
            '%m/%d/%y',  # MM/DD/YY
            '%d/%m/%y',  # DD/MM/YY
            '%m-%d-%Y',  # MM-DD-YYYY
            '%m-%d-%y',  # MM-DD-YY
            '%m/%d/%y',  # MM/DD/YY
            '%d-%m-%Y',  # DD-MM-YYYY
            '%d-%m-%y',  # DD-MM-YY
            '%m-%d-%Y',  # MM-DD-YYYY
            '%m-%d-%y',  # MM-DD-YY
            '%m/%d/%Y',  # MM/DD/YYYY
            '%d-%m-%Y',  # DD-MM-YYYY
        ]

        for date_format in potential_formats:
            try:
                # Print the format being attempted and the date_string
                print(f"Trying format: {date_format}")
                print(f"Date string: {date_string}")
                
                # Return the parsed date if parsing is successful
                return parser.parse(date_string, dayfirst=(date_format == '%d-%m-%Y')).date()
            except ValueError:
                # Print an error message if parsing fails for the current format
                print(f"Failed to parse using format: {date_format}")
                

        # Print a message indicating that none of the formats matched
        print("Unsupported date format. Can try adding potential_formats in the parseInvocieDate function in risk_customFeature.py.")
        #(DO NOT RAISE ERROR)  raise ValueError("Unsupported date format.  Can try adding potential_formats in the parseInvocieDate function in risk_customFeature.py.")
        #Return -2 to indicate parsing failure.  If -2 is returned, it will be received by app.py to send invoice to management for review directly.
        return -2

#####################################################################################
"""
Custom Feature Extraction 1:  function to return feature value 0 or 1 (False or True) for whether the invoice date is on weekend or holiday.  
"""

def is_weekend(date_string):
    if isinstance(date_string, datetime.date):
        invoice_date = date_string
    else:
        # Convert the date string to a datetime object
        invoice_date = parse_invoice_date(date_string) #datetime.strptime(date_string, '%Y-%m-%d').date()

    # Check if the day of the week is Saturday or Sunday (5 and 6 are Saturday and Sunday)
    if invoice_date.weekday() in [5, 6]:
        return True
    else:
        return False

def is_holiday(date_string):
     # Convert the date string to a datetime object
    if isinstance(date_string, datetime.date):
        invoice_date = date_string
    else:
        invoice_date = parse_invoice_date(date_string) #datetime.strptime(date_string, '%Y-%m-%d').date()

    # List of holidays (replace with your specific holidays)
    us_holidays = [
        (1, 1),    # New Year's Day
        (7, 4),    # Independence Day
        (12, 25),  # Christmas Day
        # Add more holidays as needed
    ]

    # Check if the month and day of the invoice date match any holiday
    return (invoice_date.month, invoice_date.day) in us_holidays
    
#feature 1. invoice date is Weekend or holiday?
def fea1_weekendOrHoliday(dateExtracted):
    dateExtracted = parse_invoice_date(dateExtracted)
    is_weekend_int = 0
    if is_holiday(dateExtracted) or is_weekend(dateExtracted):
        is_weekendint = 1
    return is_weekend_int



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
    # in case invoice date is not in form YYYY-MM-DD, convert the format into it.
    extractedInvoiceDate = parse_invoice_date(extractedInvoiceDate)
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
    bool_result = 1
    # Get the DataFrame of projects under the employee's responsibility
    projects_under_employee_df = db.queryAllProjectsUnderCertainEmployee(employee_id = employeeID)

    # Check if the project_name is in the ProjectName column of the DataFrame
    if (project_name in projects_under_employee_df['PROJECTNAME'].values):
        bool_result = 0
    return bool_result
    
#testing
#result = isEmployeeUnderProject(employeeID=employeeID, project_name = project_name)
#print(result)

'''
Custom Feature Extraction 4:  Amount overclaimed?
return: 0 if no overclaim.   1 if input amount is greater than amount extracted from invoice.
'''
def fea4_amountIsOverclaimed(input_invoiceAmount, amountExtracted ):
    if input_invoiceAmount is None:
        return -6
    if amountExtracted is None: 
        return -3
    bool_result = 0
    if not isinstance(input_invoiceAmount, float):
        try:
            input_invoiceAmount = float(input_invoiceAmount)
        except ValueError:
            # Handle the case where conversion to float is not possible
            # You might want to log an error or handle it according to your needs
    
            #(NO NOT RAISE ERROR)raise ValueError("Unsupported input_invoiceAmount format in fea4 function in risk_customFeature.py.")
            #if format not convertable to float, should send to management for review directly by returning -3.
            
            print('invoice amount from user input not convertable to float.  Will send to management for review directly.')
            return -6   #input invoice amount not valid
    if not isinstance(amountExtracted, float):
        try:
            amountExtracted = float(amountExtracted)
        except ValueError:
            # Handle the case where conversion to float is not possible
            # You might want to log an error or handle it according to your needs
            
            #(DO NOT Raise error)raise ValueError("Unsupported amountExtracted format in fea4 function in risk_customFeature.py.")
            print('invoice amount from invoice nlp extraction not convertable to float.  Will send to management for review directly.')
            return -3
    variance = input_invoiceAmount - amountExtracted
    if variance > 0:
        bool_result = 1
    else: 
        bool_result = 0
    return bool_result

'''
Custom Feature Extraction 5:   if the amount is multiple of 100. 
'''
def fea5_is_invoice_amount_multiple_of_100(input_invoiceAmount):
    """
    Check if the invoice amount is a multiple of 100.

    Returns:
    - True if the invoice amount is a multiple of 100, False otherwise
    """
    #convert input_invoiceAmount to numerical type
    try:
        # Convert input_invoiceAmount to a numeric type
        input_invoiceAmount = float(input_invoiceAmount)
    except ValueError:
        print("unsupported operand type(s) for parameter - input_invoice Amount: -  Check data type of parameter for fea5 function in risk_customFeature.py. ")
        raise ValueError("Unsupported date format.  Can try adding potential_formats in the function of fea5 in risk_customFeature, or convert all amount to float type.")
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
    if not isinstance(inputAmount, float):
        # If not, try to convert it to float
        try:
            inputAmount = float(inputAmount)
        except ValueError:
            # Handle the case where conversion to float is not possible
            raise ValueError("Unsupported inputAmount format in isRoundingAmount function in risk_customFeature.py.") 
    tolerance = 0.1   
    rounded_amount = round(inputAmount)
    difference = abs(inputAmount) - rounded_amount
    return difference < tolerance
#feature to assess , if the current invoice amount is a rounding number, whether rounding amounts repeatedly claimed by same person over the last three claims from the same person. 
def fea6_has_repeated_rounding_numbers(employee_id, input_invoiceAmount):
    """
    Check if an employee has repeatedly claimed rounding numbers in the reimbursement history.

    Parameters:
    - employee_id: str, the employee ID to check
    - reimbursement_df: DataFrame, reimbursement history DataFrame

    Returns:
      Ratio of rounding amount claimed within the last three claims for the same person.
    - a value between 0 to 1.   Good if 0, bad if 1. if the employee has repeatedly claimed rounding numbers.
    """
    ratio = 0
    df_reimbursementHist_temp = db.queryReimbursementRequestRecordsFromDB()
    df_reimbursementHist_forEmployeeID = df_reimbursementHist_temp[df_reimbursementHist_temp['EmployeeID'] == employee_id]
    if isRoundingAmount(inputAmount=input_invoiceAmount):
        # Get the last three lines of the reimbursement data frame or all existing lines if there are fewer than five
        last_three_lines = df_reimbursementHist_forEmployeeID.tail(3)

        # Count the number of rounding amounts within the last three claims
        rounding_count = sum(isRoundingAmount(amount) for amount in last_three_lines['InvoiceAmount'])
        # Calculate the ratio

        total_claims = min(3, len(last_three_lines))  # Use the actual number of claims or 3, whichever is smaller
        if total_claims != 0:
            ratio = rounding_count / total_claims
    return ratio

"""
    Custom Feature Extraction 7:   Check if the text contains similar keywords related to personal expenses.
"""
def fea7_contains_personalExpense_keywords(textOfInvoice):
    
    nlp = spacy.load("en_core_web_md")
    bool_flag = 0
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
            bool_flag = 1

    return bool_flag
"""
    Custom Feature Extraction 8:   Check if the the current user input invoice amount has a big variance 
    with predicted invoice amount for the current date of claim  for this employee for the same expense 
    category.
"""
def fea8_suddenChangeInBehavior(inputInvoiceAmount, futureDate, employeeID, expenseCategory, df_reimbursementHistory):
    
    if not isinstance(inputInvoiceAmount, float):
        try:
            inputInvoiceAmount = float(inputInvoiceAmount)
        except ValueError:
            # Handle the case where conversion to float is not possible
            print("Unsupported inputInvoiceAmount parameter in fea8 function in risk_customFeature.py.")
            return -6
    # to use prediction function prepared
    # # TODO     done
    predictedInvoiceAmount = forcast.predictInvoiceAmount(futureDate, employeeID, expenseCategory, df_reimbursementHistory)
    variance = inputInvoiceAmount - predictedInvoiceAmount
    featureValue_scaled = min_max_scaling(variance)
    print(f'feature 8 -scaled sudden change of behavior index = {featureValue_scaled}')
    return featureValue_scaled          

# helper function for scaling.
def min_max_scaling(variance):
    if variance > 1000:
        return 1
    min_value = 0  # Set your minimum value here
    max_value = 1000  # Set your maximum value here
    
    scaled_variance = (variance - min_value) / (max_value - min_value)
    
    return scaled_variance

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
    # In case where invoice_date is not in format YYYY-MM-DD, convert format to this form. 
    invoice_date = parse_invoice_date(invoice_date)
    bool = 0
    try:
        # Filter the project DataFrame for the specified project name
        project = project_df[project_df['PROJECTNAME'] == project_name].iloc[0]

        # Parse the project start and end dates to datetime objects
        project_start_date = datetime.strptime(project['ProjectDuration_startDate'], '%m/%d/%Y')
        project_end_date = datetime.strptime(project['ProjectDuration_endDate'], '%m/%d/%Y')
        invoice_datetime = invoice_date #datetime.strptime(invoice_date, '%m-%d-%Y')

        # Check if the invoice date falls between the project start and end date
        if invoice_datetime < project_start_date or invoice_datetime > project_end_date:
            bool = 1
    except Exception as e:
        # Handle exceptions (e.g., project name not found)
        print(f"Error: {e}")
    return bool