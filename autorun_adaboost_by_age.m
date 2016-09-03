% This script creates a auc performance matrix as a function of prediction window and age group

% autorun adaboost_by_age
path2add = {fullfile(pwd, 'boosted-hii_stableVer')};
path_list_cell = regexp(path, pathsep, 'split');

for ii = 1:length(path2add)
    if ~any(ismember(path2add{ii}, path_list_cell))
        disp('added path')
        addpath(path2add{ii});
    end
end



tlag = [6:24];
% ism_tt_dir = 'train_test_ism7_nofill_only_level';
% stm_tt_dir = 'train_test_ism7_nofill_only_level';
% ism_tt_dir = 'train_test_ism6_nofill';
test_dir = 'train_test_banner_nofill';
test_db = 'banner';
train_method = 'across';
load('results_boosted_hii8/boosted_hii_train_across_test_stm_last_manual018_train0.mat')
cvres = model(1).predictor;
predictors_interest = model(1).feature_all;

manual_fts = {'age', 'nsbp_last', 'ndbp_last', 'hr_last', 'spo2_last', ...
              'hemoglobin_last', 'temperature_last', 'wbc_last', 'platelet_last', ...
              'albumin_last', 'urine_last', 'potassium_last', 'glucose_last', ...
              'creatinine_last', 'lactic_acid_last', 'osi_last', 'si_last', 'oi_last'};

ex_ft = {'ph', 'ratio_pao2_flo2','bun', 'bilirubin', 'calcium'};
suffices = {'min', 'max', 'mean', 'median', 'last'};
count=1;
for jj = 1:length(ex_ft)
    for kk = 1:length(suffices)
        ex_ft_full{count} = [ex_ft{jj} '_' suffices{kk}];
        count = count+1;
    end                
end


% age_group = [1/12 ,1/2 ; 1/2, 1 ; 1, 4 ; 4, 7 ; 7, 9 ; 9, 21];
age_group = [0 ,22];

only_manual = true;
train_by_age = false;
test_by_age = true;


for mm = 1:length(tlag)
	load(fullfile(pwd, test_dir, sprintf('%s_onset_tt_tlag%03d_twin006.mat', test_db, tlag(mm))));
	switch train_method
	case 'across'
		if tlag(mm)~=6			
			X_test = [X_train ; X_test];
			y_test = [y_train(:) ; y_test(:)];			
		end		
	case 'ism'
		X_test = [X_test ; X_train];
		y_test = [y_test(:) ; y_train(:)];
	end
	predictors_test = cellstr(predictors);
	mask_test = ismember(predictors_test, predictors_interest);
	X_test = X_test(:, mask_test);
	predictors_test = predictors_test(mask_test);

	idx_order = [];
	for jj = 1:length(predictors_interest)
	    idx_order(end+1) = find(ismember(predictors_test, predictors_interest{jj}));
	end
	X_test = X_test(:, idx_order);
	predictors_test = predictors_test(idx_order);
	w_test = 1/length(y_test)*ones(length(y_test),1);	
	idx_age = find(strcmp(predictors_test, 'age'));
	X_test(:,idx_age) = X_test(:,idx_age)./365;	

	[X_grouped, y_grouped, ~] = splitByAge(X_test, y_test, w_test, predictors_test, age_group);
	% keyboard

	AUC_by_age = [];
	for ii = 1:length(y_grouped)
	    X_test_age = X_grouped{ii};
	    y_test_age = y_grouped{ii};
	    y_pred_age  = boostedHII_predict(X_test_age,cvres.clf_full,'probability');	
			[~,~,~,AUC_test] = perfcurve(y_test_age,y_pred_age,1);			
	    AUC_by_age(end+1) = AUC_test;
	end

	
	AUC_by_pw_age(mm,:) = AUC_by_age;
end


pred_win = tlag - 6;
age_group = {'1 - 6 months' ; '6 months - 1 yr' ; '1 - 4 yrs' ; '4 - 7 yrs' ; '7 - 9 yrs' ; '9 - 21 yrs'};
group_id = [1:length(age_group)];

load('jet_bg.mat')
figure;
figSize = [286         103        1095         655];
imagesc(group_id, pred_win, AUC_by_pw_age)
set(gca, 'xticklabel', age_group)
title_str{1} = 'AUC as a function of Age Group and Prediction Window:';
title_str{2} = 'Train CHLA & Test CHLA';
set(gca,'xticklabel',age_group)
title(title_str)
xlabel('Age Group')
ylabel('Prediction Window (hours)')
set(gca, 'fontsize',15,'fontweight','bold')
cbh = colorbar;
ylabel(cbh, 'AUC')
colormap(cmap)
set(gcf, 'position', figSize)
set(gcf, 'paperpositionmode', 'auto')
