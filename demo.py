
"""
python -i expenseNLP.py
"""
#Get names of features extracted from ifidf vectorizer
vocabulary = vectorizer.get_feature_names_out()
"""
Result:  array(['000', '01', '02', '03', '0301', '04', '040', '05', '050', '0501',
       '051', '06', '07', '0701', '08', '09', '10', '100', '11', '1101',
       '12', '15', '150', '16', '18', '180', '19', '20', '200', '2023',
       '2024', '2025', '2026', '2027', '2029', '2030', '2031', '2032',
       '2033', '2034', '2035', '2036', '25', '250', '30', '300', '315',
       '350', '400', '450', '460', '480', '500', '540', '550', '570',
       '580', '600', '650', '700', '740', '750', '800', '850', '900',
       '920', '950', '980', 'abc', 'academy', 'accommodation',
       'accommodations', 'acquired', 'additional', 'advanced',
       'advantage', 'airlines', 'annual', 'arrangements', 'art',
       'artistic', 'best', 'booked', 'books', 'bought', 'building',
       'business', 'card', 'care', 'catered', 'catering', 'celebration',
       'chairs', 'client', 'com', 'comcast', 'commute', 'company',
       'companywide', 'conference', 'connectx', 'contact', 'corp',
       'courses', 'cozy', 'creative', 'creativity', 'credit', 'culinary',
       'custom', 'davis', 'decor', 'delights', 'delivery', 'delta',
       'desks', 'development', 'devices', 'doordash', 'educational',
       'efficiency', 'electronic', 'email', 'employee', 'emporium',
       'endeavors', 'engaged', 'enhanced', 'enlisted', 'enrolled',
       'entertainment', 'environment', 'equipment', 'ergoessentials',
       'ergonomic', 'ergonomics', 'event', 'events', 'expansion',
       'expenses', 'express', 'fitness', 'foods', 'fresh', 'furniture',
       'gadgets', 'getaways', 'global', 'green', 'growth', 'harmony',
       'haven', 'health', 'high', 'home', 'hospitality', 'hosted',
       'hotels', 'hub', 'individual', 'initiatives', 'innovations',
       'innovative', 'installation', 'instruments', 'internet',
       'invested', 'invoice', 'issue', 'items', 'landscaping', 'learning',
       'lines', 'logistics', 'lyft', 'market', 'marriott', 'mastery',
       'materials', 'meals', 'meeting', 'meetings', 'monthly',
       'movemaster', 'musical', 'new', 'nights', 'number', 'office',
       'officemax', 'online', 'package', 'packages', 'party', 'payment',
       'pet', 'phone', 'plan', 'price', 'printer', 'pro', 'procured',
       'products', 'professional', 'program', 'programs', 'project',
       'provider', 'purchase', 'purchased', 'qty', 'recognition',
       'registration', 'relaxation', 'relocation', 'relocations',
       'retreat', 'retreats', 'seating', 'service', 'services', 'session',
       'setup', 'shipping', 'skills', 'smith', 'software', 'solutions',
       'specialized', 'spectrum', 'speed', 'staples', 'stationery',
       'subscription', 'summit', 'supplier', 'supplies', 'sustainable',
       'swift', 'team', 'tech', 'teletech', 'thirty', 'tools', 'training',
       'transaction', 'transportation', 'travel', 'travels', 'trends',
       'trip', 'turner', 'unique', 'upgrade', 'upgraded', 'vendor',
       'visits', 'wellbeing', 'wellness', 'workshop', 'workshops', 'xyz'],
      dtype=object)
"""
# IDF weights learned from the training data
idf_weights = vectorizer.idf_
"""
Result:  array([3.71841299, 2.36386733, 3.88831202, 3.62880083, 4.98692431,
       4.61919953, 5.39238942, 2.1937163 , 4.41156017, 4.13962645,
       5.39238942, 3.29232859, 3.57323098, 5.23823874, 3.85194438,
       2.68433922, 2.86666078, 4.29377713, 3.27212589, 4.78625362,
       3.02526581, 2.37196454, 4.98692431, 4.35093555, 5.39238942,
       5.23823874, 3.96527307, 3.62880083, 3.14109762, 5.10470735,
       4.98692431, 3.17681571, 5.23823874, 3.71841299, 3.35550749,
       3.85194438, 3.25232326, 3.27212589, 3.29232859, 3.29232859,
       2.80212226, 3.68764133, 3.05701451, 4.54509156, 5.10470735,
       3.96527307, 5.39238942, 4.54509156, 3.96527307, 4.98692431,
       5.39238942, 4.98692431, 3.1587972 , 5.39238942, 5.23823874,
       4.8815638 , 4.69924224, 3.65778837, 5.39238942, 4.18841662,
       5.39238942, 4.98692431, 3.05701451, 4.41156017, 4.23970991,
       5.10470735, 4.78625362, 5.23823874, 4.29377713, 5.39238942,
       5.23823874, 4.18841662, 4.54509156, 5.23823874, 4.41156017,
       5.10470735, 4.47609869, 4.69924224, 5.23823874, 4.47609869,
       4.41156017, 5.23823874, 3.71841299, 5.23823874, 5.23823874,
       3.65778837, 2.92146901, 5.23823874, 4.78625362, 4.18841662,
       4.61919953, 4.61919953, 4.18841662, 4.09310644, 3.78295151,
       5.23823874, 5.39238942, 3.75016169, 4.54509156, 5.23823874,
       4.47609869, 3.78295151, 4.8815638 , 5.23823874, 5.23823874,
       4.41156017, 5.10470735, 5.23823874, 4.69924224, 4.78625362,
       4.8815638 , 4.23970991, 5.10470735, 5.23823874, 5.23823874,
       4.98692431, 4.18841662, 4.78625362, 5.10470735, 4.8815638 ,
       4.78625362, 4.8815638 , 3.78295151, 2.28630909, 5.23823874,
       5.10470735, 4.8815638 , 4.13962645, 4.98692431, 4.78625362,
       4.98692431, 5.39238942, 3.75016169, 5.23823874, 4.35093555,
       5.23823874, 5.23823874, 5.10470735, 5.39238942, 4.29377713,
       4.8815638 , 4.8815638 , 5.23823874, 4.98692431, 3.60062995,
       4.98692431, 4.8815638 , 4.69924224, 4.8815638 , 5.39238942,
       4.98692431, 4.8815638 , 3.71841299, 5.10470735, 4.54509156,
       5.10470735, 4.8815638 , 4.78625362, 4.69924224, 4.61919953,
       5.39238942, 5.39238942, 5.10470735, 4.41156017, 4.78625362,
       3.47057682, 4.09310644, 2.1735136 , 3.25232326, 3.78295151,
       4.8815638 , 4.29377713, 5.39238942, 4.8815638 , 5.10470735,
       5.23823874, 5.23823874, 4.78625362, 3.96527307, 4.13962645,
       5.39238942, 4.00609506, 4.98692431, 4.8815638 , 4.78625362,
       4.47609869, 5.23823874, 3.78295151, 2.09655256, 5.23823874,
       5.23823874, 5.23823874, 4.35093555, 5.39238942, 3.78295151,
       4.78625362, 3.49526944, 5.39238942, 3.33400129, 5.23823874,
       4.69924224, 4.09310644, 4.8815638 , 4.04865468, 3.62880083,
       4.54509156, 4.00609506, 2.0721611 , 4.98692431, 3.25232326,
       4.35093555, 4.23970991, 5.23823874, 5.39238942, 4.54509156,
       4.69924224, 3.92605235, 4.61919953, 5.23823874, 2.77742964,
       2.53018854, 5.23823874, 4.98692431, 5.39238942, 4.78625362,
       4.98692431, 4.8815638 , 3.12370588, 5.10470735, 5.10470735,
       4.54509156, 4.8815638 , 4.98692431, 4.35093555, 4.98692431,
       2.20054227, 3.05701451, 5.10470735, 4.29377713, 2.86666078,
       3.71841299, 4.98692431, 5.23823874, 5.10470735, 3.78295151,
       2.6198007 , 4.23970991, 3.12370588, 4.54509156, 4.78625362,
       4.8815638 , 5.23823874, 5.23823874, 4.04865468, 4.29377713,
       2.00799916, 4.8815638 , 4.47609869, 3.42294878, 4.41156017,
       5.10470735, 4.23970991])
"""

"""
Risk comparison:
EmployeeID P000002 had reasonable claim amount compared to our prediction for the same 
expense category for same employee based on past claim records in the ReimbursementRequestRecords 
table in database. 


"""
# Partial Data in ReimbursementRequestRecords table
# for employee'E000435', , past reimbursement request for'Phone & Internet', before last invoice date '2021-03-19', 'B9620'
    #were all around 165.   If we have a new invoice amount significantly greater than this amount, the risk score should raise up. 
    
dbcon.execute('''
    INSERT INTO ReimbursementRequestRecords 
    (RecordID, EmployeeID, ProjectID, ExpenseCategory, SubmissionDate, InvoiceDate, InvoiceID, InvoiceAmount)
    VALUES 
    ('R000006', 'E000002', 'P000002', 'Phone & Internet', '2020-06-20','2020-05-16', 'B92399','159.00'),
    ('R000007', 'E000435', 'P000005', 'Phone & Internet', '2020-06-21','2020-05-19', 'B92300','159.00'),
    ('R000008', 'E000435', 'P000005', 'Phone & Internet', '2020-07-21','2020-06-19', 'B92310','159.00'),
    ('R000009', 'E000435', 'P000005', 'Phone & Internet', '2020-08-25','2020-07-19', 'B92314','159.00'),
    ('R000010', 'E000435', 'P000005', 'Phone & Internet', '2020-09-21','2020-08-19', 'B92318','159.00'),
    ('R000011', 'E000435', 'P000005', 'Phone & Internet', '2020-10-20','2020-09-19', 'B92350','159.00'),
    ('R000012', 'E000435', 'P000005', 'Phone * Internet', '2020-11-20', '2020-10-19', 'B9600',
    '159.00'),
    ('R000013', 'E000435', 'P000005', 'Phone * Internet', '2021-01-20', '2020-11-19', 'B9605',
    '168.00'),
    ('R000014', 'E000435', 'P000005', 'Phone & Internet', '2021-02-21', '2020-12-19', 'B9610',
    '169.00'),
    ('R000015', 'E000435', 'P000006', 'Phone & Internet', '2021-03-21', '2021-01-19', 'B9611',
    '169.00'),
    ('R000016', 'E000435', 'P000006', 'Phone & Internet', '2021-04-21', '2021-02-19', 'B9612',
    '169.00'),
    ('R000017', 'E000435', 'P000006', 'Phone & Internet', '2021-05-21', '2021-03-19', 'B9614',
    '170.00'),
    ('R000018', 'E000435', 'P000006', 'Phone & Internet', '2021-04-21', '2021-03-19', 'B9615',
    '171.00'),
   