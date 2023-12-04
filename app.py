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


#import expenseNLP as nlp

app = Flask(__name__)
nextAction = 'approve'  #initiate random decision string

"""
helper functions and variables and data frames
"""

#Variable for path of trained model from model file stored by expenseNLP.py file
trained_expenseNLP_model_path = 'trained_model.joblib'
#Variable for path of trained model for classifying fraud risk score  (model was trained already in file predictRiskScore_trainning.py).
trained_fraudRiskScore_model_path = 'trained_riskScore_model.pth'

#get reimbursement history date
df_reimbursementHistory = db.queryReimbursementRequestRecordsFromDB()
df_projectEmployee = db.queryProjectEmployeesRelationFromDB()
df_employees = db.queryEmployeesFromDB
df_projects = db.queryProjectsFromDB()

#Define function to get the predicted expense category, invoice Number, invoice amount, currency unit, invoice date.
def getInfoFromInvoiceText(invoice_text):             
	#Use trained model for expense category classification from expenseNLP.py and store prediction of expense category into a variable.
	#expenseCategoryPredictingModel = joblib.load(trained_expenseNLP_model_path) 
	#predicted_expense_category = expenseCategoryPredictingModel.predict([invoice_text])    #TODO  to check what the returned prediction is (not sure if it's expense category or all)
	#Testing (Printing Status on Terminal) :   invoice info from NLP obtained by app.py
	#print(f'Invoice info extracted using NLP is obtained by app.py file and stored to variable.')
	#return predicted_expense_category          #TODO to update for invoice Number, invoice amount, currency unit, invoice date
    pass



# Define function to assess whether it is a special case invoice claim.  
# Special case include:  for Travel, or invoice falls outside of project duration.
def isSpecialCase(fea9):
    return fea9 == 1

# Define function to get the optimum policy of actions(decision) from the MDP process
# parameter r:  fraud risk score for this invoice
# parameter x:  invoice amount from user input
# parameter a:  variance  =  invoice amount from user input - invoice amount from extraction from text using NLP
def getOptimumPolicy(r, x, a,isSpecialCase_variable):
    #get the unique mdp specifically for this invoice based on it's unique r,x,a
    mdp = ReimbursementMDP(r, x, a, isSpecialCase_variable)
    policy, reward = mdp.iterate()
    return policy


# Define function to predict Fraud Risk Score based on a certain feature vector of 9 elements. 


def getPredictedRiskScore(feature_vector, model_path, input_size, output_size):   #takes feature vector in example format [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#TODO call prediction model trained by riskEvaluation.py 
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
    expenseCategory_InvoiceTextNLP = 'Travel'                #TODO to replace
    invoiceAmount_InvoiceTextNLP = '345'                    #TODO to replace
    invoiceNumber_InvoiceTextNLP = 'XYZ345'                  #TODO to replace
    currency_unit_InvoiceTextNLP = 'USD'                     #TODO to replace
    invoiceDate_invoiceTextNLP = '12-24-2022'                 #TODO to replace
    print(f'data from invoice text extracted and stored using NLP.')


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
    fea8 = fea.fea8_suddenChangeInBehavior(inputInvoiceAmount=invoice_amount)
    print(f'custom feature 8 extracted. Feature 8 (contains_personalExpense_keywords)= {fea8}')     #testing
    fea9 = fea.fea9_is_project_duration_covering(df_projects, project_name=project_name, invoice_date=invoiceDate_invoiceTextNLP)
    print(f'custom feature 9 extracted. Feature 9 (contains_personalExpense_keywords)= {fea9}')     #testing
    
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
    #calculate 
    r = fraudRiskScore
    x = float(invoice_amount)          
    a = float(invoice_amount) - float(invoiceAmount_InvoiceTextNLP)
    optimumPolicy = getOptimumPolicy(r, x, a,isSpecialCase_variable)





    return nextAction, optimumPolicy

"""

"""

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        employee_id = request.form['employee_id']
        project_name = request.form['project_name']
        invoice_amount = request.form['invoice_amount']
        currency_unit = request.form['currency_unit']
        invoice_text = request.form['invoice_text']


        nextAction,optimumPolicy = process_form_data(employee_id, project_name, invoice_amount, currency_unit, invoice_text)
        return render_template('result.html', decision=nextAction)

    return render_template('index_withText.html')

if __name__ == '__main__':
    app.run(debug=True)


