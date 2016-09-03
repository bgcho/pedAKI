function boostedHII_saveClassifier(clf,features)
rz = load('../../Desktop/HIRBA/results_boostedHII/logistic_stump_comparison_new/decisionstump_new_stacking_method_results_over_rounds_partitioned_by_hospital_T100.mat');

features = {'Arterial BP Mean',...
            'NBP Mean',...
            'AST',...
            'Albumin',...
            'Arterial pH',...
            'BUN',...
            'Calcium',...
            'Carbon Dioxide',...
            'Chloride',...
            'Creatinine',...
            'HCO3',...
            'Hematocrit',...
            'Hemoglobin',...
            'PT',...
            'Platelets',...
            'Potassium',...
            'Sodium',...
            'WBC - Leukocytes',...
            'PF Ratio',...
            'Arterial BP Systolic',...
            'Heart Rate',...
            'NBP Systolic',...
            'Shock Index - HR/ASBP',...
            'Shock Index - HR/SBP',...
            'CVP',...
            'FiO2 Set',...
            'Mean Airway Pressure',...
            'Peak Insp Pressure',...
            'Respiratory Rate',...
            'SpO2',...
            'Temperature C',...
            'ALT',...
            'Alk Phosphate',...
            'Amylase',...
            'Arterial Base Excess',...
            'Arterial PaCO2',...
            'Arterial PaO2',...
            'CPK',...
            'CPK MB',...
            'Glucose',...
            'INR',...
            'Ionized Calcium',...
            'LDH',...
            'Lactic Acid',...
            'Magnesium',...
            'PTT',...
            'RBC',...
            'SaO2',...
            'Total Bili',...
            'Total Protein',...
            'Triglyceride',...
            'Arterial BP Diastolic',...
            'NBP Diastolic',...
            'Eos',...
            'Polys',...
            'Bands',...
            'Monos',...
            'Basos',...
            'Lymphs',...
            'Neutrophil to Lymphocyte ratio',...
            'Age'};
        
        