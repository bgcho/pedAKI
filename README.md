# pedAKI
Work flow of pediatric AKI predictor

1. Get design matrix X and label vector y
1-1. ISM
	a. Run pedAKI_get_itemdf_ism.ipynb: This script queries the ISM db and extracts the features for X (Only includes the encounters that have creatinine measurement). This reduces time for the following steps since we do not need to query the remote db anymore.
	b. Run pedAKI_ism_gen_iobatch.ipynb: This script filters the patients by age, length of stay, etc and assigns AKI stage and reference time from the creatinine measurement. Then, I/O dataframe for each feature is queried for given prediction window and observation window.
	c. Run pedAKI_add_composite_variables_ism.ipynb: This script adds the composite variables (SI, OSI, OI) to the I/O dataframe in step b.
	d. Run pedAKI_plausibility_filter.ipynb: This script filters the plausible values of each feature. Set the db_name variable as 'ism' before running the script.
	e-1. Run pedAKI_prep_train_test.ipynb: This script reforms the I/O dataframe to a form that is compatible for train/test. This script provides the final X and y for train/test.
	e-2. Run pedAKI_prep_train_test_nofill.ipynb: This script does the same thing as pedAKI_prep_train_test.ipynb, but doesn't fill the NaN values. Set the db_name variable as 'ism' before running the script.
1-2 STM
	* Note: Before running the pedAKI_get_itemdf_stm.ipynb, you will need access to the postgreSQL server in ICCADEV1. Follow the instructions below to get access.
					1) Access ICCADEV1 server either remotely or locally. For remote access, the ip address of ICCADEV1 server is 130.140.52.181, Username is Iccadev1\Administrator, and Password is hirba4u
					2) Open "C:\Program Files\PostgreSQL\9.5\data\pg_hba.conf"
					3) Add a line "host	 all	 all	 YOUR_IP_ADDRESS/32	 md5" where YOUR_IP_ADDRESS is your local machine's ip address. For example, If you would like to remotely access to postgreSQL server in ICCADEV1, from your local machine with ip address 130.140.57.17, then you should add "host	 all	 all	 130.140.57.17/32	 md5" at the end of pg_hba.conf
	a. Run pedAKI_get_itemdf_stm.ipynb: This script queries the STM db and extracts the features for X and y.
	b. Run pedAKI_gen_scr_io_batch.ipynb: This script filters the patients by age, length of stay, etc and assigns AKI stage and reference time from the creatinine measurement. Then, I/O dataframe for each feature is queried for given prediction window and observation window.
	c. Run pedAKI_add_composite_variables_stm.ipynb: This script adds the composite variables (SI, OSI, OI) to the I/O dataframe in step b.
	d. Run pedAKI_plausibility_filter.ipynb: This script filters the plausible values of each feature. Set the db_name variable as 'stm' before running the script.
	e-1. Run pedAKI_prep_train_test.ipynb: This script reforms the I/O dataframe to a form that is compatible for train/test. This script provides the final X and y for train/test.
	e-2. Run pedAKI_prep_train_test_nofill.ipynb: This script does the same thing as pedAKI_prep_train_test.ipynb, but doesn't fill the NaN values. Set the db_name variable as 'stm' before running the script.
1-3 Banner
	* Note: Before running pedAKI_import_banner.ipynb, follow the instruction below
					1) Install pyspark
					2) Install findspark
					3) Start pedAKI_import_banner.ipynb as usual jupyter notebook (Do not start with pyspark since we create the spark context in the script)
	a. Run pedAKI_import_banner.ipynb: This script converts parquet file formatted chartevent tables into pandas dataframe.
	b. Run pedAKI_get_itemdf_banner.ipynb: This script queries the Banner chartevent tables and extracts the features for X.
	c. Run pedAKI_filter_Banner.ipynb: This script filters the patients by age, length of stay, etc and assigns AKI stage and reference time from the creatinine measurement.
	d. Run pedAKI_convertUrine_banner.ipynb: This script converts urine output in cc to Urine rate per weight.
	e. Run pedAKI_gen_IOmat_banner.ipynb: This script creates I/O dataframe given a prediction window and observation window.
	f. Run pedAKI_add_composite_variables_banner.ipynb: This script adds the composite variables (SI, OSI, OI) to the I/O dataframe in step e.
	g. Run pedAKI_plausibility_filter.ipynb: This script filters the plausible values of each feature. Set the db_name variable as 'banner' before running the script.
	h-1. Run pedAKI_prep_train_test.ipynb: This script reforms the I/O dataframe to a form that is compatible for train/test. This script provides the final X and y for train/test.
	h-2. Run pedAKI_prep_train_test_nofill.ipynb: This script does the same thing as pedAKI_prep_train_test.ipynb, but doesn't fill the NaN values. Set the db_name variable as 'banner' before running the script.


2. Train/Test Logistic regression
	Run pedAKI_logistic.ipynb


3. Train/Test Adaboost
	Run autorun_adaboost.m
	* Note: boostedHII_cv.m, boostedHII_train.m are modified to manage imbalance between institution.
	* Note: It seems that there should be at least two values in each group (y=0, y=1) to run boostedHII_train.m without error.
					For example, wbc count in Banner only has one value for y=1 group. For this case, boostedHII_train.m returns an error related to the decision stump.


4. Test Adaboost for age groups
	Run autorun_adaboost_by_age.m	


5. libraries to run the .ipynb files and .m files
	a. ism_utilities_Ben.py
	b. pedAKI_utilities.py
	c. stm_utilities.py
	d. stm_tabledef.py
	e. stm_dbinit.py
	f. pedAKI_predictor.py
	g. boosted-hii_stableVer
	h. get_classifier.m
	i. splitByAge.m
