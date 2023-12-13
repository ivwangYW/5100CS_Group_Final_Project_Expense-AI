# 5100CS_Group_Final_Project_Expense-AI
AI Empowered Automated Employee Expense Reimbursement Process

## Project Process:  ##
##### Process Diagram: ##### 
under folder 'Guide', see file Diagram 1 - Process Diagram.pdf
<br>
https://lucid.app/lucidchart/c1208529-7687-4675-92aa-f0c39d2d320d/edit?view_items=h9CPPX0PBLSY&invitationId=inv_8b067be2-0838-4a98-8c72-c8ffc477b320

##### MDP Table  #####
under folder 'Guide', see file Diagram 2 - Decision Making - MDP Diagram.pdf

##### Feature process (Same as the one on report) 
under folder 'Guide', see file Diagram 3 - Fraud Risk Score Worksheet

## Instruction on Running the Project ##
Instruction and result for running the project: Our project can be started by running the **app.py** file, which will return a web address in the terminal. Copy the web address and paste it to your web browser,  so that our employee reimbursement website (Expense AI) will appear.  Assume we are the employee trying to submit a reimbursement request to the website.  We tested our project by  inputting text of invoice and other required information to the html webpage, and clicking the 'Submit" button.  Almost instantly, the result (The suggested next state) will be displayed. To check for what happened in our decision-making process,  please return to the terminal at the back-end.  A full process note will be printed in the terminal, including the received input from the employee, the extracted information from the text of invoice, the feature calculation process, the assessed Fraud Risk Score,  feature vector values,  and suggested optimum sequence of actions from the MDP.  <br><br>
Our html website of the system will only display the next suggested step to the employee (the user), but if a reviewer or the company’s internal personnel want to dig in to check for the details of the processing progress of our system, additional information may be reviewed by checking the back-end terminal.  The MDP model will produce a sequence of actions that the agent deems optimal for our process, so that if the employee was requested to provide more evidence, the system can go from the suggested next step from the sequence of actions once the employee provides the requested evidence, instead of starting over the entire process from the very beginning.   However, our website does not have a portal to take, store and deliver additional evidence or to send the documents to management for a review.  It will just end after the html website displays decisions of the first action to take, such as ‘request for more evidence’ or ‘submit to management for review’. In the future, we might expand the project to include further steps that use the optimum policy (sequence of actions) that is shown in the back-end terminal directly, so that the system continues to the next suggested step once taking evidence from an employee or once the invoice was reviewed by management. <br><br>
So far, please note that due to the time limit of the semester, our project will end after it displays to the user the first suggested action(represented by the name of the next suggested state) within the optimal sequential policy of actions in the back-end terminal. <br>
Please see below for a link to the demo for running the **app.py** file.<br>
https://clipchamp.com/watch/0O8Ymey6213

## Team Contributions ##
Design of the project: Yiwei Wang <br>
Dataset and database:  Yiwei Wang <br>
Extract text information and expense category classification using NLP:  Pengkun Ma, Yiwei Wang<br>
Linear Regression for predicting future invoice amount: Bingyang Ke<br>
Feature Extraction:  Yiwei Wang<br>
Fraud Risk Score Assessment using Neural Network:  Nanxiang Zhao<br>
Markov Desicion Process: Nanxiang Zhao<br>
Demo:  Yiwei Wang<br>
Report: Drafted collectively

## ML Training Dataset ##
#### For geting invoice info from Text of invoice: ####   
See csv file ' dataset_with Labels.csv'
#### For training model to classify fraud risk score based on customized feature vector values: ####
See csv file ' dataset_MLtrainingVectors_fraudRiskScore_labeling.csv'

#### Currency Signs: ####
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

## Set up before running ##
#### Accessing SQLite Database -  Start installing sqlite3 package ##
type  'pip install pandas sqlite3' in the terminal  
then you can start querying data by importing the python file dataConverter_fromDB.py, the data frames for each table are there for you to use directly.

#### Download before running this app ####
**python -m spacy download en_core_web_md**                        // to check if a text contains certain keyword<br>
**pip install pandas sqlite3**                                   // for accessing SQLite database<br>


