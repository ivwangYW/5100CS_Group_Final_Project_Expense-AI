import pandas as pd
from flask import Flask, render_template, request
import PyPDF2

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        employee_id = request.form['employee_id']
        project_name = request.form['project_name']
        invoice_amount = request.form['invoice_amount']
        currency_unit = request.form['currency_unit']
        submission_date = request.form['submission_date']

        # Process PDF file
        pdf_file = request.files['pdf_invoice']
        if pdf_file:
            pdf_text = extract_pdf_text(pdf_file)

            # Save the data and pdf_text to a database or file
            save_data_to_database(employee_id, project_name, invoice_amount, currency_unit, submission_date, pdf_text)

    return render_template('index.html')

def extract_pdf_text(pdf_file):
    text = ""
    pdf_reader = PyPDF2.PdfFileReader(pdf_file)
    num_pages = pdf_reader.numPages

    for page_number in range(num_pages):
        page = pdf_reader.getPage(page_number)
        text += page.extractText()

    return text

def save_data_to_database(employee_id, project_name, invoice_amount, currency_unit, submission_date, pdf_text):
    # Implement saving data to your database or file system
    # For example, you can use SQLAlchemy to interact with a database

if __name__ == '__main__':
    app.run(debug=True)

//print('Please input invoice text: ')
//input(invoiceText, str)