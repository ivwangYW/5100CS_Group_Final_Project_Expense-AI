import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import os
import re


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
# Separate into two lists
text_of_invoice_list = [item[0] for item in dataset]
expense_category_list = [item[1] for item in dataset]


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
4. Unpacking the elements of the train_set and test_set tuple into two separate lists, 
X_train and y_train
"""
x_train, y_train = zip(*train_set)
x_test, y_test = zip(*test_set)
labels = set(y_train)

#convert the above tuples into list of strings
x_train = list(x_train)
y_train = list(y_train)
labels = list(labels)

"""
5.  Define preprocessor function for data processing before vectorization
	First, in the terminal, type the following command to install nltk package: 
	pip install nltk
"""

def custom_preprocessor(text):
	
	# Additional stopwords specific to your domain can be added to this list
	custom_stopwords = set(stopwords.words('english'))
	#remove all numbers from each text
	text = remove_numbers(text)
	# Tokenize using nltk
	tokenized_text = word_tokenize(text.lower())  # Convert to lowercase for consistency
		
	# Remove punctuation and non-alphabetic characters
	tokenized_text = [token for token in tokenized_text if token.isalpha()]

	# Remove stopwords
	tokenized_text = [token for token in tokenized_text if token not in custom_stopwords]
		

	# Concatenate the tokens into a single string
	return ' '.join(tokenized_text)

#remove all numbers to prevent vectorizer to use numbers as features
	# token_pattern=u'(?u)\b\w*[a-zA-Z]\w*\b'
def remove_numbers(text):
	# Use regex to remove all numbers
	text_without_numbers = re.sub(r'\d+', '', text)
	return text_without_numbers


"""
6. Feature Extraction : Extracting features from the training data 
using a sparse vectorizer
"""
# Vectorize the text data using TF-IDF.  (Apply the TF-IDF vectorizer to the 
# training text data to learn the vocabulary and IDF weights (for feature extraction).)

vectorizer = TfidfVectorizer(
	max_df=0.5,
	min_df=5,
	stop_words="english",
	#include the custom preprocessor into the vectorizer so that test data and other new data can be processed consistantly through the pipeline.
	preprocessor=custom_preprocessor    
	      
)	


"""
7. Define NLP pipeline.
"""
# Create a pipeline with TF-IDF vectorizer and a classifier (e.g., Support Vector Machine,  Neural Networks,  MultinomialNB, etc.)
text_classifier = Pipeline([
	('tfidf', vectorizer),
	('clf', SVC())  # classifier can be changed.     
])



"""
8.   Train the model
"""
# Fit the pipeline on the training data
text_classifier.fit(x_train, y_train)


"""
# (?? Not sure if we need it)  Inverse transform the labels for evaluation
#y_test_inverse = le.inverse_transform(y_test)
#predictions_inverse = le.inverse_transform(predictions)
"""




"""
9. Evaluate the model
	(?? will delete later if not needed--> )  accuracy = metrics.accuracy_score(y_test, predictions)
"""

# Evaluate the model on the test data
accuracy = text_classifier.score(x_test, y_test)
print(f"Accuracy on the test set: {accuracy}")


# ((?? might not work)  Assuming you have a new invoice text, not in both the training and testing datasets, try to get the classification answer
print('Now, please input text of invoice, and wait for classification of expense category:  ')
one_text_of_invoice = input()
if ( one_text_of_invoice.isalpha ):
	answer = text_classifier.predict([one_text_of_invoice])[0]
print(f'new invoice test: {answer}')

