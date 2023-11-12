import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

# Download the NLTK stopwords corpus
nltk.download('stopwords')
nltk.download('punkt')

"""
1.  Setting up data:    First,  Using the source file dataset.csv to create 'dataset' as a list of tuples where each tuple contains the text 
and expense category.  
"""
# Load the Excel file into a DataFrame and skip the header row
df = pd.read_csv('dataset.csv')

# Split the 'Source Data'column into 'Text of Invoice' and 'Expense Category' columns.  Using the ',' in the dataset as the separater, and split column into two from right to left for one time using rsplit.
df[['Text of Invoice', 'Expense Category']] = df['Source Data'].str.rsplit(",", expand=True, n=1)

# Create a list of tuples representing your dataset
dataset = [(row['Text of Invoice'], row['Expense Category']) for _, row in df.iterrows()]



"""
2.  Label Encoding
	performing label encoding on the 'Expense Category' column of your 
	DataFrame. Label encoding is a technique used to convert categorical 
	labels (strings or other non-numeric types) into numerical values.
	It's necessary when working with machine learning algorithms that 
	expect input data to be in numerical form.
"""
le = LabelEncoder()
#Fit and transform the 'Expense Category' column into numerical code.
df['Expense Category'] = le.fit_transform(df['Expense Category'])


"""
3. Split the dataset into training and testing sets
"""
# Split the dataset into training and testing sets
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=None)

"""
4. Unpacking the elements of the train_set tuple into two separate lists, 
X_train and y_train
"""
x_train, y_train = zip(*train_set)

"""
5.  Define function for data processing 
	First, in the terminal, type the following command to install nltk package: 
	pip install nltk
"""
#tokenization
def tokenize_and_clean(each_invoice_text):
	

	# Additional stopwords specific to your domain can be added to this list
	custom_stopwords = set(stopwords.words('english'))
	# Tokenize using nltk
	tokens = word_tokenize(each_invoice_text.lower())  # Convert to lowercase for consistency

	# Remove punctuation and non-alphabetic characters
	tokens = [token for token in tokens if token.isalpha()]

	# Remove stopwords
	tokens = [token for token in tokens if token not in custom_stopwords]

	return tokens




"""
5. Define NLP pipeline.
# Define NLP pipeline
model = make_pipeline(
	CountVectorizer(analyzer=tokenize_and_clean),
	MultinomialNB()
)
"""




"""
# Train the model

model.fit(X_train, y_train)

# Make predictions on the test set
X_test, y_test = zip(*test_set)
predictions = model.predict(X_test)

"""



"""
# Inverse transform the labels for evaluation
y_test_inverse = le.inverse_transform(y_test)
predictions_inverse = le.inverse_transform(predictions)

# Evaluate the model
accuracy = metrics.accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy}")

# Now you can use 'model' for predicting the category of new invoices
# Example: category = model.predict(["New invoice text"])
"""


