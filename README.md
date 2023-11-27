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
1. Update data.csv    ( more data)     -Ivy
2. feature labels for fraud score   -Ivy
3. design function using Time Series Forcasting (Supervised ML) for predictions of future Invoice Amount (y, output) on a certain future date (x, input) for certain employee(employeeID)'s spending of certain type of expense (Expense Category) , based on historical Invoice Dates and Amounts for each employee for each expense category in database table - ReimbursementRequestRecords
4. design a function to determine likelihood of personal expense (in features worksheet)
5. feature value functions     -Ivy
6. Database tables         - Ivy  done    
7. NLP for invoice Amount extraction (can use data in dataConverter directly)
8. NLP for invoice date extraction (can use data in dataConverter directly)
9. Reward Calculation (need to update based on data from dataConverter_fromDB)
10. Reinforcement learning process        MDP   
11. fraud risk score  ( ML neural network process: feature value scaling + add layers + select loss  function         +calculate loss + backpropagation = classifier function to predict where score is 0,5 or 10)
12. main.py    (input: see process diagram; output: actions)


13. classifier in the pipeline can be changed  (if we have time, can build our own neural network classifier, or not, or try other classification algorithms) (can use data in dataConverter directly)
14. optimizing any detail
15. try other accuracy matrics
16. any graph, if needed




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


