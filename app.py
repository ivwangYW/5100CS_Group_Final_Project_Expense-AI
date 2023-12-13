from flask import Flask, render_template, request
import datetime
import risk_customFeature as fea
import dataConverter_fromDB as db
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import sqlite3
import predictRiskScore_nnModel as net
import CONSTANTS as const
import torch
import torch.nn as nn
import MDP 
import expenseNLP_modeling as nlp


#import expenseNLP as nlp

app = Flask(__name__)
suggestedNextState_result = 'submit_to_management'  #initiate result to review due to data validation check. 
optimum_path_result = ['start', 'submit_to_management']



"""
helper functions and variables and data frames
"""
bool_validData = True
bool_invoiceDate_notDetectable = False
bool_invoiceAmount_notDetectable = False
bool_currencyUnit_notDetectable = False
bool_expenseCategory_notDetectable = False

bool_invoiceAmountFromInput_notDetectable = False
#Variable for path of trained model from model file stored by expenseNLP.py file
trained_expenseNLP_model_path = 'trained_model.joblib'
#Variable for path of trained model for classifying fraud risk score  (model was trained already in file predictRiskScore_trainning.py).
trained_fraudRiskScore_model_path = 'trained_riskScore_model.pth'
#Variable for path of trained model for classifying invoice text to corresponding expense category (model was trained already in file expenseNLP_trainning.py)
trained_NLP_model_path = 'trained_model.joblib'

#get reimbursement history date
df_reimbursementHistory = db.queryReimbursementRequestRecordsFromDB()
df_projectEmployee = db.queryProjectEmployeesRelationFromDB()
df_employees = db.queryEmployeesFromDB
df_projects = db.queryProjectsFromDB()



def dataValidation(invoiceAmount_nlp, invoiceDate_nlp, invoice_amount_from_input):             
	#Use trained model for expense category classification from expenseNLP.py and store prediction of expense category into a variable.
	#expenseCategoryPredictingModel = joblib.load(trained_expenseNLP_model_path) 
	#predicted_expense_category = expenseCategoryPredictingModel.predict([invoice_text])    #TODO  to check what the returned prediction is (not sure if it's expense category or all)
	#Testing (Printing Status on Terminal) :   invoice info from NLP obtained by app.py
	#print(f'Invoice info extracted using NLP is obtained by app.py file and stored to variable.')
	#return predicted_expense_category          #TODO to update for invoice Number, invoice amount, currency unit, invoice date
    #str_nlp_invoiceNumber, str_nlp_invoiceDate, str_nlp_invoiceAmount, expenseCategory_predicted,  str_nlp_currencyUnit = nlp.NLP_getTextInfo(invoice_text, trained_NLP_model_path)
    print(f'Validating nlp invoice date from text: {invoiceDate_nlp}')
    if invoiceDate_nlp is None:
        bool_invoiceDate_notDetectable = True
        return False
    if invoiceDate_nlp is not None:
        temp_parsedDate = fea.parse_invoice_date(invoiceDate_nlp)
    temp_invoiceAmount_validation = fea.fea4_amountIsOverclaimed(invoice_amount_from_input, invoiceAmount_nlp )
    if temp_parsedDate == (-2):
        bool_invoiceDate_notDetectable = True
        #set decision result to be displayed as review
        return False

    if temp_invoiceAmount_validation == (-6):
        bool_invoiceAmountFromInput_notDetectable = True
        return False
    if temp_invoiceAmount_validation == (-3):
        bool_invoiceAmount_notDetectable = True
        return False
    #return false indicating invalid data 
    return True



# Define function to assess whether it is a special case invoice claim.  
# Special case include:  for Travel, or invoice falls outside of project duration.
def isSpecialCase(fea9):
    return fea9 == 1

# Define function to get the optimum policy of actions(decision) from the MDP process
# parameter r:  fraud risk score for this invoice
# parameter x:  invoice amount from user input
# parameter a:  variance  =  invoice amount from user input - invoice amount from extraction from text using NLP
def getOptimumPath(x, r, a,isSpecialCase_variable):
    #get the unique mdp specifically for this invoice based on it's unique r,x,a
    mdp = MDP.ReimbursementMDP(x, r, a, isSpecialCase_variable)
    #Call q_learning function in the MDP.py file to get the output of optimum path index 
    optimum_path_index = mdp.q_learning()

    #get index for suggested next state
    suggestedNextState_index = optimum_path_index[1]

    #convert to string type names of states from indexes
    mdpStatesMapping = MDP.mdpState_mapping
    optimum_path_result = ', '.join(mdpStatesMapping[state] for state in optimum_path_index)  
    suggestedNextState_result = mdpStatesMapping[suggestedNextState_index]


    return suggestedNextState_result, optimum_path_result


# Define function to predict Fraud Risk Score based on a certain feature vector of 9 elements. 


def getPredictedRiskScore(feature_vector, model_path, input_size, output_size):   #takes feature vector in example format [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#call prediction model trained by riskEvaluation.py 
    def load_model(model_path, input_size, output_size):
        model = net.NeuralNetwork(input_size, output_size)  # Adjust input_size and output_size accordingly
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model


    #return fraudRiskScore
    loaded_model = load_model(model_path, input_size, output_size)    
    '''
    forward propagation to predict output class
    '''
    # Convert the feature_vector to a PyTorch tensor
    feature_tensor = torch.tensor(feature_vector, dtype=torch.float32)  
    # Ensure the model is in evaluation mode
    loaded_model.eval()
    # Make the prediction
    with torch.no_grad():
        output = loaded_model(feature_tensor)

    # Get the predicted class index
    _, predicted_class = torch.max(output.data, 1)
    # Convert the predicted class to fraud risk score.
    fraudRiskScore_mapping = const.fraudRiskScore_mapping
    fraudRiskScore_result = fraudRiskScore_mapping[predicted_class.item()]
    return fraudRiskScore_result



"""
Main function that reflects all our steps of invoice decision making processes from user input to the final result
"""
def process_form_data(employee_id, project_name, invoice_amount, currency_unit, invoice_text):
    global suggestedNextState_result, optimum_path_result  # Use the global variable
    
    """
    Step 1:   Getting user input and save to variables.
              Input elements:   Invoice Text, employee_id, project_name, invoice_amount, currency_unit, invoice_text.
              Automatically generated:  submission_date
    """
    # Get the current date automatically and store in submission_date variable.
    submission_date = datetime.datetime.now().strftime('%Y-%m-%d')
    # Add 'E' in front of employee id input from user
    employee_id = 'E' + employee_id
    # Testing (Printing Status on Terminal) : Now we have all the variables. Print the variables obtained from user input to the console.
    print(f'*********************User Input***********************************')
    print(f"Employee ID: {employee_id}")
    print(f"Project Name: {project_name}")
    print(f"Invoice Amount: {invoice_amount}")
    print(f"Currency Unit: {currency_unit}")
    print(f"Invoice Text: {invoice_text}")
    print(f'Submission Date: {submission_date}')
    print(f'******************************************************************')

    
    '''
    Step 2:   Using NLP to process invoice to extract detail from the text of invoice: 
              extracted data include:  classified invoice category, and invoice date, invoice amount, invoice number, currency unit. 
    '''
    """
    2.1: Predict invoice info
    """
    # to call function 
    # store the extracted/ predicted data into variables

    str_nlp_invoiceNumber, str_nlp_invoiceDate, str_nlp_invoiceAmount, str_nlp_expenseCategory,  str_nlp_currencyUnit = nlp.NLP_getTextInfo(invoice_text, trained_NLP_model_path )
    #storing invoice info processed from nlp into variables 
    expenseCategory_InvoiceTextNLP = str_nlp_expenseCategory                #TODO done
    invoiceAmount_InvoiceTextNLP = str_nlp_invoiceAmount                    #TODO done
    invoiceNumber_InvoiceTextNLP = str_nlp_invoiceNumber                  #TODO done
    currency_unit_InvoiceTextNLP = str_nlp_currencyUnit                     #TODO done
    invoiceDate_invoiceTextNLP = str_nlp_invoiceDate                #TODO done
    #testing
    print('*************************data from invoice text extracted and stored using NLP.***********************')
    print(f'Expense Category (NLP predicted):   {expenseCategory_InvoiceTextNLP} ')
    print(f'Invoice Amount (NLP predicted):   {invoiceAmount_InvoiceTextNLP} ')
    print(f'Invoice Number (NLP predicted):   {invoiceNumber_InvoiceTextNLP} ')
    print(f'Currency Unit (NLP predicted):   {currency_unit_InvoiceTextNLP} ')
    print(f'Invoice Date (NLP predicted):   {invoiceDate_invoiceTextNLP} ')


    '''
        Data Validation.   If data not valid, DO NOT PROCEED.  send to management for review directly.
    '''
    #data validation check (for invoiceAmount nlp, invoiceDate nlp, invoice_amount_from_input). if not valid, send to managment for review directly without proceeding.
    bool_validData = dataValidation(invoiceAmount_InvoiceTextNLP, invoiceDate_invoiceTextNLP, invoice_amount)   
    
    
    if bool_validData is False:
        print('**********************Decision***********************************')
        print('Note: Missing Valid Info.  Please submit invoice info again so that the system can extract useful info: ')
        print(f'Suggested Next State: {suggestedNextState_result}')
        print(f'Suggested Optimum Policy(Path): {optimum_path_result}')
        return suggestedNextState_result, optimum_path_result #suggestedNextState_result, [0, 4]

    
    
    """
    Step 3: Fraud Risk Score generation
    """
    #getting scaled values of Custom Features from file risk_customFeature.py based on 9 different risk factors.
    print('custom feature extracting started.')
    fea1 = fea.fea1_weekendOrHoliday(invoiceDate_invoiceTextNLP)
    print(f'custom feature 1 extracted. Feature 1 (isWeekendOrHoliday)= {fea1}')     #testing
    fea2 = fea.fea2_is_duplicate_invoice(invoiceDate_invoiceTextNLP, invoice_amount, invoiceNumber_InvoiceTextNLP,df_reimbursementHistory)
    print(f'custom feature 2 extracted. Feature 2 (is_duplicate_invoice)= {fea2}')     #testing 
    fea3 = fea.fea3_isEmployeeUnderProject(employeeID=employee_id, project_name=project_name)   #testing
    print(f'custom feature 3 extracted. Feature 3 (isEmployeeUnderProject)= {fea3}')     #testing 
    fea4 = fea.fea4_amountIsOverclaimed(invoice_amount,invoiceAmount_InvoiceTextNLP)
    print(f'custom feature 4 extracted. Feature 4 (amountIsOverclaimed)= {fea4}')     #testing
    fea5 = fea.fea5_is_invoice_amount_multiple_of_100(invoice_amount)
    print(f'custom feature 5 extracted. Feature 5 (is_invoice_amount_multiple_of_100)= {fea5}')     #testing
    fea6 = fea.fea6_has_repeated_rounding_numbers(employee_id=employee_id,input_invoiceAmount=invoice_amount)
    print(f'custom feature 6 extracted. Feature 6 (has_repeated_rounding_numbers)= {fea6}')     #testing
    fea7 = fea.fea7_contains_personalExpense_keywords(textOfInvoice=invoice_text)
    print(f'custom feature 7 extracted. Feature 7 (contains_personalExpense_keywords)= {fea7}')     #testing
    fea8 = fea.fea8_suddenChangeInBehavior(inputInvoiceAmount=invoice_amount, futureDate = invoiceDate_invoiceTextNLP, employeeID = employee_id, expenseCategory = expenseCategory_InvoiceTextNLP, df_reimbursementHistory = df_reimbursementHistory)
    print(f'custom feature 8 extracted. Feature 8 (sudden_change_in_behavior)= {fea8}')     #testing
    fea9 = fea.fea9_is_project_duration_covering(df_projects, project_name=project_name, invoice_date=invoiceDate_invoiceTextNLP)
    print(f'custom feature 9 extracted. Feature 9 (Invoice data falls outside of Project Duration)= {fea9}')     #testing
    
    #generate a vector of customized feature values using the above feature values
    # Create a NumPy array
    feature_vector = np.array([fea1, fea2, fea3, fea4, fea5, fea6, fea7, fea8, fea9])
    #store boolean variable to tell if the invoice is a special case.
    isSpecialCase_variable = isSpecialCase(fea9)
    # Reshape the array to be a row vector (1x9)
    feature_vector = feature_vector.reshape(1, -1)
    
    # Now, feature_vector is a 2D NumPy array representing a row vector
    print(f'custom feature vector generated : {feature_vector}.')
    # predict risk score for this feature_vector.
    fraudRiskScore = getPredictedRiskScore(feature_vector, trained_fraudRiskScore_model_path, input_size=const.input_size, output_size=const.output_size)
    # Testing (Printing Status on Terminal) : risk score generated:
    print(f'fraud risk score predicted: {fraudRiskScore}')
    

    """
    Step 4:   get optimum policy and next action according to the policy
    """
    #calculate reward indexes - See MDP process Diagram for what they are.
    r = fraudRiskScore    #r is fraud Risk Score
    x = float(invoice_amount)    # x is invoice amount      
    a = float(invoice_amount) - float(invoiceAmount_InvoiceTextNLP)     #a is variance between invoice amount from employee input and extracted invoice amount from invoice text.
    suggestedNextState_result, optimum_path_result = getOptimumPath(x, r, a,isSpecialCase_variable)
    print('')
    print('**********************MDP Decision***********************************')
    print(f'Suggested Next State: {suggestedNextState_result}')
    print(f'Suggested Optimum Policy(Path): {optimum_path_result}')




    return suggestedNextState_result, optimum_path_result

"""

"""

@app.route('/', methods=['GET', 'POST'])
def index():
    ##form_processed = False  #added
    if request.method == 'POST':

        



        employee_id = request.form['employee_id']
        project_name = request.form['project_name']
        invoice_amount = request.form['invoice_amount']
        currency_unit = request.form['currency_unit']
        invoice_text = request.form['invoice_text']

        ## Set form_processed to True only if the form data is successfully processed
        ##form_processed = True               #added

        '''
        Processing Invoice.
        '''

        suggestedNextState_result, optimum_path_result = process_form_data(employee_id, project_name, invoice_amount, currency_unit, invoice_text)
        return render_template('result.html', decision=suggestedNextState_result)

    return render_template('index_withText.html')

if __name__ == '__main__':
    app.run(debug=True)


