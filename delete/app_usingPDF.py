from flask import Flask, render_template, request
import pandas as pd
import PyPDF2
from datetime import datetime

app = Flask(__name__)
def extract_text_from_pdf(pdf_file):
    pdf_text = ""
    if pdf_file and allowed_file(pdf_file.filename):
        pdf_content = pdf_file.read()
        pdf_reader = PyPDF2.PdfReader(pdf_content)
        for page_num in range(len(pdf_reader.pages)):
            pdf_text += pdf_reader.pages[page_num].extract_text()
    return pdf_text

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'

def process_form_data(employee_id, project_name, invoice_amount, currency_unit, pdf_invoice):
    # Get the current date
    submission_date = datetime.now().strftime('%Y-%m-%d')

    # Extract text from the PDF
    pdf_text = extract_text_from_pdf(pdf_invoice)

    # Create a DataFrame to store the data
    data = {
        'Employee ID': [employee_id],
        'Project Name': [project_name],
        'Invoice Amount': [invoice_amount],
        'Currency Unit': [currency_unit],
        'Submission Date': [submission_date],
        'PDF Text': [pdf_text]
    }

    df = pd.DataFrame(data)

    # Do further processing or save to a database as needed
    # For simplicity, let's return the DataFrame for now
    print('Submission successful')
    return df

# Route for the main page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data from the submitted form
        employee_id = request.form['employee_id']
        project_name = request.form['project_name']
        invoice_amount = request.form['invoice_amount']
        currency_unit = request.form['currency_unit']
        pdf_invoice = request.files['pdf_invoice']

        # Process form data
        df = process_form_data(employee_id, project_name, invoice_amount, currency_unit, pdf_invoice)

        # Return the processed data (for demonstration purposes)
        return df.to_html()

    return render_template('index_withText.html')

if __name__ == '__main__':
    app.run(debug=True)