import csv

def swap_invoice_expense(infile, outfile):
    with open(infile, 'r', newline='', encoding='latin-1') as csvfile:
        reader = csv.reader(csvfile)
        lines = list(reader)

    for i, line in enumerate(lines):
        # Assuming the line format is consistent, split by ', ' from the back
        sections = line[0].rsplit(', ', maxsplit=5)

        # Swap invoice number and expense category
        invoice_number, expense_category = sections[1], sections[2]
        sections[1], sections[2] = expense_category, invoice_number

        # Join the sections back
        lines[i][0] = ', '.join(sections)

    # Use DictWriter to write the modified lines
    header = ["Line"]
    with open(outfile, 'w', newline='', encoding='latin-1') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        
        # Write each line as a dictionary with a single key 'Line'
        for line in lines:
            writer.writerow({'Line': line[0]})

# Replace 'toDelete.csv' and 'newFile.csv' with your file paths
swap_invoice_expense('toDelete.csv', 'newFile.csv')