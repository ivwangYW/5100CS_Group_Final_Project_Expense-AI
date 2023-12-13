
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
    
'''
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
   '''



# Invoice Amount Forcast Examples  
# In reimbursementRequestRecords
"""
EmployeeID	ProjectID	ExpenseCategory	SubmissionDate	InvoiceDate	InvoiceID	InvoiceAmount
E000001	P000001	Travel	1/20/2020	1/15/2020	INV001	500
E000002	P000002	Phone & Internet	1/22/2020	1/16/2020	B00DG2	150
E000002	P000002	Phone & Internet	2/22/2020	2/16/2020	B94859	150
E000002	P000002	Phone & Internet	3/25/2020	3/16/2020	B92359	150.56
E000002	P000002	Phone & Internet	6/20/2020	4/16/2020	B92450	159
E000002	P000002	Phone & Internet	6/20/2020	5/16/2020	B92399	159
E000435	P000005	Phone & Internet	6/21/2020	5/19/2020	B92300	159
E000435	P000005	Phone & Internet	7/21/2020	6/19/2020	B92310	159
E000435	P000005	Phone & Internet	8/25/2020	7/19/2020	B92314	159
E000435	P000005	Phone & Internet	9/21/2020	8/19/2020	B92318	159
E000435	P000005	Phone & Internet	10/20/2020	9/19/2020	B92350	159
E000435	P000005	Phone & Internet	11/20/2020	10/19/2020	B9600	159
E000435	P000005	Phone & Internet	1/20/2021	11/19/2020	B9605	168
E000435	P000005	Travel	6/20/2020	2/16/2020	TRAIN319	135
E000435	P000005	Travel	6/20/2020	3/16/2020	TRAIN320	135
E000435	P000005	Travel	6/21/2020	4/16/2020	TRAIN321	135
E000435	P000005	Travel	7/21/2020	5/16/2020	TRAIN322	135
E000435	P000005	Travel	8/25/2020	5/19/2020	TRAIN323	135
E000435	P000005	Travel	9/21/2020	6/19/2020	TRAIN324	135
E000435	P000005	Travel	10/20/2020	7/19/2020	TRAIN325	135
E000435	P000005	Travel	11/20/2020	8/19/2020	TRAIN326	135
E000435	P000005	Travel	1/20/2021	9/19/2020	TRAIN327	135
E000435	P000005	Travel	1/21/2021	10/19/2020	TRAIN328	135
E000435	P000005	Travel	1/22/2021	11/19/2020	TRAIN329	140
E000435	P000005	Travel	1/23/2021	12/19/2020	TRAIN330	140
E000435	P000005	Travel	1/24/2021	1/19/2021	TRAIN331	140
E000435	P000005	Travel	2/19/2021	2/19/2021	TRAIN332	140
E000435	P000005	Travel	3/19/2021	3/19/2021	TRAIN333	140
E000435	P000005	Travel	4/19/2021	4/19/2021	TRAIN334	140
E000435	P000005	Travel	5/19/2021	5/19/2021	TRAIN335	140
E000435	P000005	Travel	6/19/2021	6/19/2021	TRAIN336	140
E000435	P000005	Travel	7/19/2021	7/19/2021	TRAIN337	140
E000435	P000005	Travel	8/19/2021	8/19/2021	TRAIN338	140
E000435	P000005	Travel	9/19/2021	9/19/2021	TRAIN339	140
E000435	P000005	Travel	10/19/2021	10/19/2021	TRAIN340	140
E000435	P000005	Travel	11/19/2021	11/19/2021	TRAIN341	140
E000435	P000005	Travel	12/19/2021	12/19/2021	TRAIN342	140
P000006	E000006	Health and Wellness	6/20/2020	2/16/2020	PSY3234	100
P000006	E000006	Health and Wellness	6/20/2020	3/16/2020	PSY3235	130
P000006	E000006	Health and Wellness	6/21/2020	4/16/2020	PSY3236	100
P000006	E000006	Health and Wellness	7/21/2020	5/16/2020	PSY3237	116
P000006	E000006	Health and Wellness	8/25/2020	5/19/2020	PSY3238	130
P000006	E000006	Health and Wellness	9/21/2020	6/19/2020	PSY3239	100
P000006	E000006	Health and Wellness	10/20/2020	7/19/2020	PSY3240	145
P000006	E000006	Health and Wellness	11/20/2020	8/19/2020	PSY3241	100
P000006	E000006	Health and Wellness	1/20/2021	9/19/2020	PSY3242	120
P000006	E000006	Health and Wellness	1/21/2021	10/19/2020	PSY3243	130
P000006	E000006	Health and Wellness	1/22/2021	11/19/2020	PSY3244	120
P000006	E000006	Health and Wellness	1/23/2021	12/19/2020	PSY3245	128
P000006	E000006	Health and Wellness	1/24/2021	1/19/2021	PSY3246	160
P000006	E000006	Health and Wellness	2/19/2021	2/19/2021	PSY3247	200
P000006	E000006	Health and Wellness	3/19/2021	3/19/2021	PSY3248	190
P000006	E000006	Health and Wellness	4/19/2021	4/19/2021	PSY3249	249
P000006	E000006	Health and Wellness	5/19/2021	5/19/2021	PSY3250	213
P000006	E000006	Health and Wellness	6/19/2021	6/19/2021	PSY3251	180
P000006	E000006	Health and Wellness	7/19/2021	7/19/2021	PSY3252	170
P000006	E000006	Health and Wellness	8/19/2021	8/19/2021	PSY3253	150
P000006	E000006	Health and Wellness	9/19/2021	9/19/2021	PSY3254	139
P000006	E000006	Health and Wellness	10/19/2021	10/19/2021	PSY3255	139
P000006	E000006	Health and Wellness	11/19/2021	11/19/2021	PSY3256	159
P000006	E000006	Health and Wellness	12/19/2021	12/19/2021	PSY3257	190
P000006	E000006	Health and Wellness	1/19/2022	1/19/2022	PSY3258	170
P000006	E000006	Health and Wellness	2/19/2022	2/19/2022	PSY3259	240
P000006	E000006	Health and Wellness	3/19/2022	3/19/2022	PSY3260	210
P000006	E000006	Health and Wellness	4/19/2022	4/19/2022	PSY3261	180
P000006	E000006	Health and Wellness	5/19/2022	5/19/2022	PSY3262	190
P000006	E000006	Health and Wellness	6/19/2022	6/19/2022	PSY3263	183
P000006	E000006	Health and Wellness	7/19/2022	7/19/2022	PSY3264	160
P000006	E000006	Health and Wellness	8/19/2022	8/19/2022	PSY3265	210
P000007	E000007	Health and Wellness	6/20/2020	2/16/2020	PSY3266	500
P000007	E000007	Health and Wellness	6/20/2020	3/16/2020	PSY3267	480
P000007	E000007	Health and Wellness	6/21/2020	4/16/2020	PSY3268	520
P000007	E000007	Health and Wellness	7/21/2020	5/16/2020	PSY3269	470
P000007	E000007	Health and Wellness	8/25/2020	5/19/2020	PSY3270	489
P000007	E000007	Health and Wellness	9/21/2020	6/19/2020	PSY3271	470
P000007	E000007	Health and Wellness	10/20/2020	7/19/2020	PSY3272	501
P000007	E000007	Health and Wellness	11/20/2020	8/19/2020	PSY3273	500
P000007	E000007	Health and Wellness	1/20/2021	9/19/2020	PSY3274	505
P000007	E000007	Health and Wellness	1/21/2021	10/19/2020	PSY3275	506
P000007	E000007	Health and Wellness	1/22/2021	11/19/2020	PSY3276	500
P000007	E000007	Health and Wellness	1/23/2021	12/19/2020	PSY3277	502
P000007	E000007	Health and Wellness	1/24/2021	1/19/2021	PSY3278	489
P000007	E000007	Health and Wellness	2/19/2021	2/19/2021	PSY3279	450
P000007	E000007	Health and Wellness	3/19/2021	3/19/2021	PSY3280	300
P000007	E000007	Health and Wellness	4/19/2021	4/19/2021	PSY3281	200
P000007	E000007	Health and Wellness	5/19/2021	5/19/2021	PSY3282	500
P000007	E000007	Health and Wellness	6/19/2021	6/19/2021	PSY3283	700
P000007	E000007	Health and Wellness	7/19/2021	7/19/2021	PSY3284	520
P000007	E000007	Health and Wellness	8/19/2021	8/19/2021	PSY3285	515
P000007	E000007	Health and Wellness	9/19/2021	9/19/2021	PSY3286	520
P000008	E000008	Travel	6/20/2020	2/16/2020	TAXI244	30
P000008	E000008	Travel	6/20/2020	3/16/2020	TAXI245	34
P000008	E000008	Travel	6/21/2020	4/16/2020	TAXI246	27
P000008	E000008	Travel	7/21/2020	5/16/2020	TAXI247	25
P000008	E000008	Travel	8/25/2020	5/19/2020	TAXI248	34
P000008	E000008	Travel	9/21/2020	6/19/2020	TAXI249	30
P000008	E000008	Travel	10/20/2020	7/19/2020	TAXI250	25
P000008	E000008	Travel	11/20/2020	8/19/2020	TAXI251	29
P000008	E000008	Travel	1/20/2021	9/19/2020	TAXI252	30
P000008	E000008	Travel	1/21/2021	10/19/2020	TAXI253	25
P000008	E000008	Travel	1/22/2021	11/19/2020	TAXI254	28
P000008	E000008	Travel	1/23/2021	12/19/2020	TAXI255	36
P000008	E000008	Travel	1/24/2021	1/19/2021	TAXI256	38
P000008	E000008	Travel	2/19/2021	2/19/2021	TAXI257	37
P000008	E000008	Travel	3/19/2021	3/19/2021	TAXI258	39
P000008	E000008	Travel	4/19/2021	4/19/2021	TAXI259	39
P000008	E000008	Travel	5/19/2021	5/19/2021	TAXI260	41
P000008	E000008	Travel	6/19/2021	6/19/2021	TAXI261	40
P000008	E000008	Travel	7/19/2021	7/19/2021	TAXI262	36
P000008	E000008	Travel	8/19/2021	8/19/2021	TAXI263	10
P000008	E000008	Travel	9/19/2021	9/19/2021	TAXI264	37
P000008	E000008	Travel	10/19/2021	10/19/2021	TAXI265	36
P000008	E000008	Travel	11/19/2021	11/19/2021	TAXI266	35
P000008	E000008	Travel	12/19/2021	12/19/2021	TAXI267	37
P000008	E000008	Travel	1/19/2022	1/19/2022	TAXI268	38
P000008	E000008	Travel	2/19/2022	2/19/2022	TAXI269	36
P000008	E000008	Travel	3/19/2022	3/19/2022	TAXI270	37
P000008	E000008	Travel	4/19/2022	4/19/2022	AIRLINE271	383

"""

