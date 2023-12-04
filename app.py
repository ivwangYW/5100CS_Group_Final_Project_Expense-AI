from flask import Flask, render_template, request
from datetime import datetime
import risk_customFeature as fea
import dataConverter_fromDB as db
import numpy as np
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder

#import expenseNLP as nlp

app = Flask(__name__)
nextAction = 'approve'  #initiate random decision string

"""
helper functions and variables and data frames
"""

#Variable for path of trained model from model file stored by expenseNLP.py file
trained_expenseNLP_model_path = 'trained_model.joblib'

#get reimbursement history date
df_reimbursementHistory = db.queryProjectEmployeesRelationFromDB()
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
#define function to get predicted fraud risk score using the feature vector specifially for the current invoice text.
def getPredictedRiskScore(feature_vector):
    #TODO call prediction model trained by riskEvaluation.py 
    #return fraudRiskScore
    pass

# Define function to assess whether it is a special case invoice claim.  
# Special case include:  for Travel, or invoice falls outside of project duration.
def isSpecialCase(fea9):
    return fea9 == 1

# Define function to get the optimum policy of actions(decision) from the MDP process
# parameter r:  fraud risk score for this invoice
# parameter x:  invoice amount from user input
# parameter a:  variance  =  invoice amount from user input - invoice amount from extraction from text using NLP
def getOptimumPolicy(r, x, a,isSpecialCase):
    #get the unique mdp specifically for this invoice based on it's unique r,x,a
    mdp = ReimbursementMDP(r, x, a, isSpecialCase)
    policy, reward = mdp.iterate()
    return policy

    
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
    submission_date = datetime.now().strftime('%Y-%m-%d')

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
    invoiceDate_invoiceTextNLP = '12/24/2022'                 #TODO to replace
    print(f'data from invoice text extracted and stored using NLP.')


    """
    Step 3: Fraud Risk Score generation
    """
    #getting scaled values of Custom Features from file risk_customFeature.py based on 9 different risk factors.
    print('custom feature extracting started.')
    fea1 = fea.fea1_weekendOrHoliday(invoiceDate_invoiceTextNLP)
    fea2 = fea.fea2_is_duplicate_invoice(invoiceDate_invoiceTextNLP, invoice_amount, invoiceNumber_InvoiceTextNLP,df_reimbursementHistory)
    fea3 = fea.fea3_isEmployeeUnderProject(employeeID=employee_id, project_name=project_name)
    fea4 = fea.fea4_amountIsOverclaimed(invoice_amount,invoiceAmount_InvoiceTextNLP)
    fea5 = fea.fea5_is_invoice_amount_multiple_of_100(invoice_amount)
    fea6 = fea.fea6_has_repeated_rounding_numbers(employee_id=employee_id,input_invoiceAmount=invoice_amount)
    fea7 = fea.fea7_contains_personalExpense_keywords(textOfInvoice=invoice_text)
    fea8 = fea.fea8_suddenChangeInBehavior(inputInvoiceAmount=invoice_amount)
    fea9 = fea.fea9_is_project_duration_covering(df_projects, project_name=project_name, invoice_date=invoiceDate_invoiceTextNLP)
    print('custom feature extracted.')
    #generate a vector of customized feature values using the above feature values
    # Create a NumPy array
    feature_vector = np.array([fea1, fea2, fea3, fea4, fea5, fea6, fea7, fea8, fea9])
    #store boolean variable to tell if the invoice is a special case.
    isSpecialCase = isSpecialCase(fea9)
    # Reshape the array to be a row vector (1x9)
    feature_vector = feature_vector.reshape(1, -1)
    
    # Now, feature_vector is a 2D NumPy array representing a row vector
    print(f'custom feature vector generated : {feature_vector}.')
    # predict risk score for this feature_vector.
    fraudRiskScore = float(getPredictedRiskScore(feature_vector))
    # Testing (Printing Status on Terminal) : risk score generated:
    print(f'fraud risk score predicted: {fraudRiskScore}')
    

    """
    Step 4:   get optimum policy and next action according to the policy
    """
    #calculate 
    r = fraudRiskScore
    x = invoice_amount
    a = invoice_amount - invoiceAmount_InvoiceTextNLP
    optimumPolicy = getOptimumPolicy(r, x, a,isSpecialCase)





    return nextAction, optimumPolicy

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