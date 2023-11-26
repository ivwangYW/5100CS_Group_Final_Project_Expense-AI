# 5100CS_Group_Final_Project_Expense-AI
Automated Expense Processing Tool Using NLP/ML and MDP

### Project Process:  ### 
##### Process Summary: #####<br>
https://lucid.app/lucidchart/c1208529-7687-4675-92aa-f0c39d2d320d/edit?view_items=h9CPPX0PBLSY&invitationId=inv_8b067be2-0838-4a98-8c72-c8ffc477b320

##### Decision Making - MDP process: #####
under folder 'Guide', see file Diagram 2 - Decision Making - MDP Diagram.pdf

##### Feature process 
under folder 'Guide', see file Diagram 3 - Fraud Risk Score Worksheet


### TODO: ###
0. dataset label add invoice #   - done 
0.2 Update data.csv    ( more data)     -Ivy
0.6 feature labels for fraud score   -Ivy
0.8 design function using Time Series Forcasting (Supervised ML) for predictions of future Invoice Amount (y, output) on a certain future date (x, input) for certain employee(employeeID)'s spending of certain type of expense (Expense Category) , based on historical Invoice Dates and Amounts for each employee for each expense category in database table - ReimbursementRequestRecords
0.9 feature value functions     -Ivy
0.94 design a function to determine likelihood of personal expense (in features worksheet)
1. Database tables         - Ivy  done    
3. NLP for invoice Amount extraction (can use data in dataConverter directly)
4. NLP for invoice date extraction (can use data in dataConverter directly)
5. Reward Calculation (need to update based on data from dataConverter_fromDB)
6. Reinforcement learning process        MDP   
7. fraud risk score  ( ML neural network process: feature value scaling + add layers + select loss  function         +calculate loss + backpropagation = classifier function to predict where score is 0,5 or 10)
8. main.py -logic   （input: see process diagram; output: actions)

9. classifier in the pipeline can be changed  (if we have time, can build our own neural network classifier, or not, or try other classification algorithms) (can use data in dataConverter directly)
10. optimizing any detail
11. try other accuracy matrics
12. any graph, if needed




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


