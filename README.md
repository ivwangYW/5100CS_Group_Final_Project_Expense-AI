# 5100CS_Group_Final_Project_Expense-AI
Automated Expense Processing Tool Using NLP/ML and MDP

### Project Process:  ### 
##### Process Summary: #####<br>
https://lucid.app/lucidchart/c1208529-7687-4675-92aa-f0c39d2d320d/edit?view_items=h9CPPX0PBLSY&invitationId=inv_8b067be2-0838-4a98-8c72-c8ffc477b320

##### Decision Making - MDP process: #####
under folder 'Guide', see file Diagram 2 - Decision Making - MDP Diagram.pdf

##### Feature process 
under folder 'Guide', see file Diagram 3 - Fraud Risk Score Worksheet

### ML Training Dataset ###
###### For geting invoice info from Text of invoice: ######   
See csv file ' dataset_with Labels.csv'
###### For training model to classify fraud risk score based on customized feature vector values: ###### 
See csv file ' dataset_MLtrainingVectors_fraudRiskScore_labeling.csv'

### TODO: ###
0. dataset label add invoice #   - done 
1. Update data.csv    ( more data)     -Ivy
2. feature labels for fraud score   -Ivy
3. design function using Time Series Forcasting (Supervised ML) for predictions of future Invoice Amount (y, output) on a certain future date (x, input) for certain employee(employeeID)'s spending of certain type of expense (Expense Category) , based on historical Invoice Dates and Amounts for each employee for each expense category in database table - ReimbursementRequestRecords
4. design a function to determine likelihood of personal expense (in features worksheet)
5. feature value functions + feature value scaling    -Ivy
6. Database tables         - Ivy  done    
7. NLP for invoice Amount extraction (can use data in dataConverter directly)
8. NLP for invoice date extraction (can use data in dataConverter directly)
9. Reward Calculation (need to update based on data from dataConverter_fromDB)
10. Reinforcement learning process        MDP   
11. fraud risk score  ( ML neural network process: add layers + select loss  function         +calculate loss + backpropagation = classifier function to predict where score is 0,5 or 10)
12. main.py    (input: see process diagram; output: actions)


13. classifier in the pipeline can be changed  (if we have time, can build our own neural network classifier, or not, or try other classification algorithms) (can use data in dataConverter directly)
14. optimizing any detail
15. try other accuracy matrics
16. any graph, if needed

## Instruction ##
Instruction and result for running the project: Our project can be started by running the app.py file, which will return a web address in the terminal. Copy the web address and paste it to your web browser,  so that our employee reimbursement website (Expense AI) will appear.  Assume we are the employee trying to submit a reimbursement request to the website.  We tested our project by  inputting text of invoice and other required information to the html webpage, and clicking the 'Submit" button.  Almost instantly, the result (The suggested next state) will be displayed. To check for what happened in our decision-making process,  please return to the terminal at the back-end.  A full process note will be printed in the terminal, including the received input from the employee, the extracted information from the text of invoice, the feature calculation process, the assessed Fraud Risk Score,  feature vector values,  and suggested optimum sequence of actions from the MDP.  
Our html website of the system will only display the next suggested step to the employee (the user), but if a reviewer or the company’s internal personnel want to dig in to check for the details of the processing progress of our system, additional information may be reviewed by checking the back-end terminal.  The MDP model will produce a sequence of actions that the agent deems optimal for our process, so that if the employee was requested to provide more evidence, the system can go from the suggested next step from the sequence of actions once the employee provides the requested evidence, instead of starting over the entire process from the very beginning.   However, our website does not have a portal to take, store and deliver additional evidence or to send the documents to management for a review.  It will just end after the html website displays decisions of the first action to take, such as ‘request for more evidence’ or ‘submit to management for review’. In the future, we might expand the project to include further steps that use the optimum policy (sequence of actions) that is shown in the back-end terminal directly, so that the system continues to the next suggested step once taking evidence from an employee or once the invoice was reviewed by management. 
So far, please note that due to the time limit of the semester, our project will end after it displays to the user the first suggested action(represented by the name of the next suggested state) within the optimal sequential policy of actions in the back-end terminal. 
Please see below for a link to the demo for running the app.py file.
https://clipchamp.com/watch/0O8Ymey6213


## Currency Signs: ## 
CA$     CAD
A$      AUD
€       EUR
£       GBP
$       USD
¥       CNY
HK$     HKD
¥       JPY
More can be found here below, but we just selected the above main currency symbols to appear in our dataset as other currencies are considered relatively rare. 
https://wise.com/gb/blog/world-currency-symbols


## Accessing SQLite Database -  Start installing sqlite3 package
type  'pip install pandas sqlite3' in the terminal  
then you can start querying data by importing the python file dataConverter_fromDB.py, the data frames for each table are there for you to use directly.

## Probablly not needed ##
SQLite download and install
Step 1. go to https://www.sqlite.org/download.html
Step 2. download:    for Windows, download'sqlite-tools-win-x64-3440200.zip',  unzip it
Step 3. then run the sqlite3.exe file and copy the file from the package/folder to it's parent folder


## See result ###
Visit http://127.0.0.1:5000/ in your browser to see the form. After submission, the decision will be displayed on the result page.


## Download before running this app ##
python -m spacy download en_core_web_md                        // to check if a text contains certain keyword
pip install pandas sqlite3                                     // for accessing SQLite database


