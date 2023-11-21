import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

# Download the NLTK stopwords corpus
if not nltk.corpus.stopwords.fileids():
    nltk.download('stopwords')

if not nltk.data.find('tokenizers/punkt'):
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
	Using LabelEncoder from sklearn.model_selection
"""
le = LabelEncoder()
#Fit and transform the 'Expense Category' column into numerical code.
df['Expense Category'] = le.fit_transform(df['Expense Category'])


"""
3. Split the dataset into training and testing sets
"""
# Split the dataset into training and testing sets using sklearn.model_selection
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=None)

"""
4. Unpacking the elements of the train_set tuple into two separate lists, 
X_train and y_train
"""
x_train, y_train = list(zip(*train_set))
x_test, y_test = list(zip(*test_set))
labels = set(y_train)
labels = list(labels)

"""
5.  Define functions for data processing 
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
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import torch

# Load the pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"  # You can choose different variations of BERT
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(set(labels)))

# Tokenize and encode the text data
train_encodings = tokenizer(x_train, truncation=True, padding=True, return_tensors='pt', max_length=512)
test_encodings = tokenizer(x_test, truncation=True, padding=True, return_tensors='pt', max_length=512)
# Create tensors for y_train and y_test
y_train_tensor = torch.tensor([labels.index(label) for label in y_train], dtype=torch.long)
y_test_tensor = torch.tensor([labels.index(label) for label in y_test], dtype = torch.long)

# Create PyTorch datasets
# Create PyTorch dataset
train_dataset = torch.utils.data.TensorDataset(
    train_encodings['input_ids'],
    train_encodings['attention_mask'],
    y_train_tensor
)
test_dataset = torch.utils.data.TensorDataset(
    test_encodings['input_ids'],
    test_encodings['attention_mask'],
    y_test_tensor
)

# Define data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

# Define the optimizer and loss function
optimizer = AdamW(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()
        output = model(input_ids, attention_mask=attention_mask)[0]
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
predictions = []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, _ = batch
        output = model(input_ids, attention_mask=attention_mask)[0]
        _, predicted = output.max(1)
        predictions.extend(predicted.tolist())

# Calculate metrics
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))



"""
# Train the model

model.fit(X_train, y_train)
# Transform the training data using the vectorizer
#X_tfidf = vectorizer.fit_transform(x_train)
# Transform the test data using the same vectorizer separately
#X_test_tfidf = vectorizer.transform(x_test)

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

