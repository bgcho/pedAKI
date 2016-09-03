path2add = {fullfile(pwd, 'boosted-hii_stableVer')};
path_list_cell = regexp(path, pathsep, 'split');

for ii = 1:length(path2add)
    if ~any(ismember(path2add{ii}, path_list_cell))
        disp('added path')
        addpath(path2add{ii});
    end
end


if only_manual
    manual_tag = sprintf('_manual%03d', length(manual_fts));
else
    manual_tag = '';
end

if only_sim
    sim_tag = '_sim0001';
else
    sim_tag = '';
end

if only_top
    top_tag = '_top10';
else
    top_tag = '';
end

if only_last
    last_tag = '_last';
else
    last_tag = '_all';
end

if train_on_time0
    t0_tag = '_train0';
else
    t0_tag = '';
end

if project
    proj_tag = '_prj';
else
    proj_tag = '';
end

if train_by_age
    tr_by_age_tag = '_trByAge';
else
    tr_by_age_tag = '';
end

if test_by_age
    te_by_age_tag = '_teByAge';
else
    te_by_age_tag = '';
end


tlag_all = [6:24];
twin_all = [6,12];
combination = combvec(tlag_all, twin_all).';
combination = combination(combination(:,1)>=combination(:,2),:);

auc_all = [];


banner_fts = {'lactic_acid', 'oi', 'hemoglobin', 'platelet', ...
           'nsbp', 'potassium', 'temperature', ...
           'urine', 'glucose', 'ndbp', 'creatinine', 'spo2', 'osi', ...
           'albumin', 'hr', 'wbc', 'si', 'age', 'sex'};

stm_fts = {'lactic_acid', 'oi', 'ph', 'hemoglobin', 'platelet', ...
           'nsbp', 'potassium', 'temperature', 'ratio_pao2_flo2', ...
           'urine', 'glucose', 'ndbp', 'creatinine', 'spo2', 'osi', ...
           'albumin', 'hr', 'wbc', 'si', 'age', 'sex'};
ism_fts = {'lactic_acid', 'oi', 'ph', 'hemoglobin', 'platelet', 'bun', ...
           'nsbp', 'potassium', 'temperature', 'ratio_pao2_flo2', ...
           'urine', 'glucose', 'ndbp', 'creatinine', 'spo2', 'osi', ...
           'calcium', 'albumin', 'bilirubin', 'hr', 'wbc', 'si', ...
           'age', 'sex'};

for ii = 1:length(banner_fts)
    banner_ft_last{ii} = [banner_fts{ii} '_last'];
end

for ii = 1:length(stm_fts)
    stm_ft_last{ii} = [stm_fts{ii} '_last'];
end

for ii = 1:length(ism_fts)
    ism_ft_last{ii} = [ism_fts{ii} '_last'];    
end

top_fts = {'lactic_acid', 'oi', 'hemoglobin', 'platelet', 'nsbp', ...
           'potassium', 'temperature', 'urine', 'glucose', 'ndbp'};

suffices = {'min', 'max', 'mean', 'median', 'last'};
count = 1;
for ii = 1:length(top_fts)
    for jj = 1:length(suffices)
        top_fts_full{count} = [top_fts{ii} '_' suffices{jj}];
        count = count+1;
    end
end

% p-value>=0.001
sm_fts = {'urine_mean', 'spo2_median', 'temperature_last', 'temperature_median', ...
          'temperature_max', 'temperature_mean', 'urine_median', 'urine_last', ...
          'spo2_mean', 'spo2_last', 'potassium_median', ...
          'potassium_mean', 'potassium_last', 'temperature_min', 'wbc_last', ...
          'wbc_max', 'wbc_mean', 'wbc_median', 'wbc_min', 'spo2_max', ...
          'creatinine_max', 'creatinine_last', 'creatinine_median', 'creatinine_mean', ...
          'creatinine_min', 'osi_max', 'spo2_min', 'oi_max'};

% % p-value>=0.01
% sm_fts = {'urine_mean', 'spo2_median', 'temperature_last', 'temperature_median', ...
%           'temperature_max', 'temperature_mean', 'urine_median', 'urine_last', ...
%           'spo2_mean', 'spo2_last', 'potassium_median', 'potassium_mean', ...
%           'potassium_last', 'temperature_min', 'wbc_last', 'wbc_max', ...
%           'wbc_mean', 'wbc_median', 'wbc_min'};

ex_ft = {'ph', 'ratio_pao2_flo2','bun', 'bilirubin', 'calcium'};
suffices = {'min', 'max', 'mean', 'median', 'last'};
count=1;
for jj = 1:length(ex_ft)
    for kk = 1:length(suffices)
        ex_ft_full{count} = [ex_ft{jj} '_' suffices{kk}];
        count = count+1;
    end                
end


load(fullfile(pwd, ism_tt_dir, 'ism_onset_tt_tlag006_twin006.mat'));
X_train_ism006 = X_train;
y_train_ism006 = y_train(:);
w_train_ism006 = 1/length(y_train_ism006)*ones(length(y_train_ism006),1);
X_test_ism006 = X_test;
y_test_ism006 = y_test(:);

load(fullfile(pwd, ism_tt_dir, 'ism_onset_tt_tlag012_twin012.mat'));
X_train_ism012 = X_train;
y_train_ism012 = y_train(:);
w_train_ism012 = 1/length(y_train_ism012)*ones(length(y_train_ism012),1);
X_test_ism012 = X_test;
y_test_ism012 = y_test(:);


predictors_ism = cellstr(predictors);
mask_sex_ism = ~ismember(predictors_ism, 'sex_M');
mask_age_ism = ismember(predictors_ism, 'age');
if only_sim
    mask_sim_ism = ismember(predictors_ism, sm_fts);
else
    mask_sim_ism = ismember(predictors_ism, predictors_ism);
end

if only_top
    mask_top_ism = ismember(predictors_ism, top_fts_full);
else
    mask_top_ism = ismember(predictors_ism, predictors_ism);
end

if only_last
    mask_last_ism = ismember(predictors_ism, ism_ft_last);
else
    mask_last_ism = ismember(predictors_ism, predictors_ism);
end

if only_manual
    mask_manual_ism = ismember(predictors_ism, manual_fts);
else
    mask_manual_ism = ismember(predictors_ism, predictors_ism);
end

mask_ex_ism = ~ismember(predictors_ism, ex_ft_full); 

mask_ism = mask_sex_ism &  mask_sim_ism & mask_ex_ism & mask_top_ism & mask_last_ism;
mask_ism = mask_ism | mask_age_ism;
mask_ism = mask_ism & mask_manual_ism;



%%%%% St.Mary's hospital %%%%%%%
load(fullfile(pwd, stm_tt_dir, 'stm_onset_tt_tlag006_twin006.mat'));
X_train_stm006 = X_train;
y_train_stm006 = y_train(:);
w_train_stm006 = 1/length(y_train_stm006)*ones(length(y_train_stm006),1);
X_test_stm006 = X_test;
y_test_stm006 = y_test(:);

load(fullfile(pwd, stm_tt_dir, 'stm_onset_tt_tlag012_twin012.mat'));
X_train_stm012 = X_train;
y_train_stm012 = y_train(:);
w_train_stm012 = 1/length(y_train_stm012)*ones(length(y_train_stm012),1);
X_test_stm012 = X_test;
y_test_stm012 = y_test(:);

clear X_train;
clear y_train;
clear X_test;
clear y_test;


predictors_stm = cellstr(predictors);
mask_sex_stm = ~ismember(predictors_stm, 'sex_M');
mask_age_stm = ismember(predictors_stm, 'age');
if only_sim
    mask_sim_stm = ismember(predictors_stm, sm_fts);
else
    mask_sim_stm = ismember(predictors_stm, predictors_stm);
end

if only_top
    mask_top_stm = ismember(predictors_stm, top_fts_full);
else
    mask_top_stm = ismember(predictors_stm, predictors_stm);
end

if only_last
    mask_last_stm = ismember(predictors_stm, stm_ft_last);
else
    mask_last_stm = ismember(predictors_stm, predictors_stm);
end

mask_ex_stm = ~ismember(predictors_stm, ex_ft_full);

if only_manual
    mask_manual_stm = ismember(predictors_stm, manual_fts);
else
    mask_manual_stm = ismember(predictors_stm, predictors_stm);
end
mask_stm = mask_sex_stm & mask_sim_stm & mask_ex_stm & mask_top_stm & mask_last_stm;
mask_stm = mask_stm | mask_age_stm;
mask_stm = mask_stm & mask_manual_stm;



%%%%% Banner hospital %%%%%%%
load(fullfile(pwd, banner_tt_dir, 'banner_onset_tt_tlag006_twin006.mat'));
X_train_banner006 = X_train;
y_train_banner006 = y_train(:);
w_train_banner006 = 1/length(y_train_banner006)*ones(length(y_train_banner006),1);
X_test_banner006 = X_test;
y_test_banner006 = y_test(:);

load(fullfile(pwd, banner_tt_dir, 'banner_onset_tt_tlag012_twin012.mat'));
X_train_banner012 = X_train;
y_train_banner012 = y_train(:);
w_train_banner012 = 1/length(y_train_banner012)*ones(length(y_train_banner012),1);
X_test_banner012 = X_test;
y_test_banner012 = y_test(:);

clear X_train;
clear y_train;
clear X_test;
clear y_test;


predictors_banner = cellstr(predictors);
mask_sex_banner = ~ismember(predictors_banner, 'sex_M');
mask_age_banner = ismember(predictors_banner, 'age');
if only_sim
    mask_sim_banner = ismember(predictors_banner, sm_fts);
else
    mask_sim_banner = ismember(predictors_banner, predictors_banner);
end

if only_top
    mask_top_banner = ismember(predictors_banner, top_fts_full);
else
    mask_top_banner = ismember(predictors_banner, predictors_banner);
end

if only_last
    mask_last_banner = ismember(predictors_banner, banner_ft_last);
else
    mask_last_banner = ismember(predictors_banner, predictors_banner);
end

mask_ex_banner = ~ismember(predictors_banner, ex_ft_full);

if only_manual
    mask_manual_banner = ismember(predictors_banner, manual_fts);
else
    mask_manual_banner = ismember(predictors_banner, predictors_banner);
end
mask_banner = mask_sex_banner & mask_sim_banner & mask_ex_banner & mask_top_banner & mask_last_banner;
mask_banner = mask_banner | mask_age_banner;
mask_banner = mask_banner & mask_manual_banner;




%%%%% ISM
X_train_ism006 = X_train_ism006(:, mask_ism);
X_test_ism006 = X_test_ism006(:, mask_ism);
X_train_ism012 = X_train_ism012(:, mask_ism);
X_test_ism012 = X_test_ism012(:, mask_ism);
predictors_ism = predictors_ism(mask_ism);

%%%%% STM
X_train_stm006 = X_train_stm006(:, mask_stm);
X_test_stm006 = X_test_stm006(:, mask_stm);
X_train_stm012 = X_train_stm012(:, mask_stm);
X_test_stm012 = X_test_stm012(:, mask_stm);
predictors_stm = predictors_stm(mask_stm);

idx_order_stm = [];
for jj = 1:length(predictors_ism)
    idx_order_stm(end+1) = find(ismember(predictors_stm, predictors_ism{jj}));
end
X_train_stm006 = X_train_stm006(:, idx_order_stm);
X_test_stm006 = X_test_stm006(:, idx_order_stm);
X_train_stm012 = X_train_stm012(:, idx_order_stm);
X_test_stm012 = X_test_stm012(:, idx_order_stm);
predictors_stm = predictors_stm(idx_order_stm);


%%%%% Banner
X_train_banner006 = X_train_banner006(:, mask_banner);
X_test_banner006 = X_test_banner006(:, mask_banner);
X_train_banner012 = X_train_banner012(:, mask_banner);
X_test_banner012 = X_test_banner012(:, mask_banner);
predictors_banner = predictors_banner(mask_banner);

idx_order_banner = [];
for jj = 1:length(predictors_ism)
    idx_order_banner(end+1) = find(ismember(predictors_banner, predictors_ism{jj}));
end
X_train_banner006 = X_train_banner006(:, idx_order_banner);
X_test_banner006 = X_test_banner006(:, idx_order_banner);
X_train_banner012 = X_train_banner012(:, idx_order_banner);
X_test_banner012 = X_test_banner012(:, idx_order_banner);
predictors_banner = predictors_banner(idx_order_banner);


for mm = 1:length(mode_all)
    mode = mode_all{mm}
    
    for ii = 1:size(combination,1)

        tlag = combination(ii,1);
        twin = combination(ii,2);
        disp(['timelag: ' num2str(tlag) ', timewindow: ' num2str(twin)])
        switch mode
            case 'ism_only'

                load(fullfile(pwd, ism_tt_dir, sprintf('ism_onset_tt_tlag%03d_twin%03d.mat', tlag, twin)));
                X_train1 = X_train;
                X_test1 = X_test;
                y_train1 = y_train(:);
                y_test1 = y_test(:);
                
                if train_on_time0
                    X_train = [eval(sprintf('X_train_ism%03d',twin)) ; eval(sprintf('X_test_ism%03d',twin))];
                    y_train = [eval(sprintf('y_train_ism%03d',twin)) ; eval(sprintf('y_test_ism%03d',twin))];
                    X_test = [X_train1 ; X_test1];
                    X_test = X_test(:, mask_ism);
                    y_test = [y_train1 ; y_test1];

                    % if tlag==twin
                    %     X_test = X_train;
                    %     y_test = y_train;
                    % else
                    %     X_test = X_test(:, mask_ism);
                    %     y_test = y_test(:);
                    % end
                else
                    X_train = X_train(:, mask_ism);
                    y_train = y_train(:);
                    X_test = X_test(:, mask_ism);
                    y_test = y_test(:);
                end
                n = length(y_train);
                w_train = 1/n*ones(n,1);

                predictors = predictors_ism;

            case 'stm_only'

                load(fullfile(pwd, stm_tt_dir, sprintf('stm_onset_tt_tlag%03d_twin%03d.mat', tlag, twin)));
                X_train1 = X_train;
                X_test1 = X_test;
                y_train1 = y_train(:);
                y_test1 = y_test(:);
                
                if train_on_time0
                    X_train = [eval(sprintf('X_train_stm%03d',twin)) ; eval(sprintf('X_test_stm%03d',twin))];
                    y_train = [eval(sprintf('y_train_stm%03d',twin)) ; eval(sprintf('y_test_stm%03d',twin))];
                    X_test = [X_train1 ; X_test1];
                    X_test = X_test(:, mask_stm);
                    X_test = X_test(:, idx_order_stm);
                    y_test = [y_train1 ; y_test1];
                    % if tlag==twin
                    %     X_test = X_train;
                    %     y_test = y_train;
                    % else
                    %     X_test = X_test(:, mask_stm);
                    %     X_test = X_test(:, idx_order);
                    %     y_test = y_test(:);
                    % end
                else
                    X_train = X_train(:, mask_stm);
                    X_train = X_train(:, idx_order_stm);
                    y_train = y_train(:);
                    X_test = X_test(:, mask_stm);
                    X_test = X_test(:, idx_order_stm);
                    y_test = y_test(:);
                end
                n = length(y_train);
                w_train = 1/n*ones(n,1);

                predictors = predictors_ism;

            case 'banner_only'

                load(fullfile(pwd, banner_tt_dir, sprintf('banner_onset_tt_tlag%03d_twin%03d.mat', tlag, twin)));
                X_train1 = X_train;
                X_test1 = X_test;
                y_train1 = y_train(:);
                y_test1 = y_test(:);
                
                if train_on_time0
                    X_train = [eval(sprintf('X_train_banner%03d',twin)) ; eval(sprintf('X_test_banner%03d',twin))];
                    y_train = [eval(sprintf('y_train_banner%03d',twin)) ; eval(sprintf('y_test_banner%03d',twin))];
                    X_test = [X_train1 ; X_test1];
                    X_test = X_test(:, mask_banner);
                    X_test = X_test(:, idx_order_banner);
                    y_test = [y_train1 ; y_test1];
                    % if tlag==twin
                    %     X_test = X_train;
                    %     y_test = y_train;
                    % else
                    %     X_test = X_test(:, mask_stm);
                    %     X_test = X_test(:, idx_order);
                    %     y_test = y_test(:);
                    % end
                else
                    X_train = X_train(:, mask_banner);
                    X_train = X_train(:, idx_order_banner);
                    y_train = y_train(:);
                    X_test = X_test(:, mask_banner);
                    X_test = X_test(:, idx_order_banner);
                    y_test = y_test(:);
                end
                n = length(y_train);
                w_train = 1/n*ones(n,1);

                predictors = predictors_ism;

                
            case 'train_ism_test_stm'                

                if train_on_time0
                    X_train1 = [eval(sprintf('X_train_ism%03d',twin)) ; eval(sprintf('X_test_ism%03d',twin))];
                    y_train1 = [eval(sprintf('y_train_ism%03d',twin)) ; eval(sprintf('y_test_ism%03d',twin))];
                else
                    load(fullfile(pwd, ism_tt_dir, sprintf('ism_onset_tt_tlag%03d_twin%03d.mat', tlag, twin)));
                    X_train1 = [X_train ; X_test];
                    X_train1 = X_train(:, mask_ism);
                    y_train1 = [y_train(:) ; y_test(:)];
                end
                n = length(y_train1);
                w_train = 1/n*ones(n,1);

                load(fullfile(pwd, stm_tt_dir, sprintf('stm_onset_tt_tlag%03d_twin%03d.mat', tlag, twin)));
                
                X_test = [X_train ; X_test];
                y_test = [y_train(:) ; y_test(:)];
                X_test = X_test(:, mask_stm);
                X_test = X_test(:, idx_order_stm);
                
                X_train = X_train1;
                % X_train = X_train(:, idx_order);
                y_train = y_train1;

                predictors = predictors_ism;

            case 'train_stm_test_ism'

                if train_on_time0
                    X_train1 = [eval(sprintf('X_train_stm%03d',twin)) ; eval(sprintf('X_test_stm%03d',twin))];
                    y_train1 = [eval(sprintf('y_train_stm%03d',twin)) ; eval(sprintf('y_test_stm%03d',twin))];
                else
                    load(fullfile(pwd, stm_tt_dir, sprintf('stm_onset_tt_tlag%03d_twin%03d.mat', tlag, twin)));
                    X_train1 = [X_train ; X_test];
                    X_train1 = X_train(:, mask_stm);
                    X_train1 = X_train1(:, idx_order_stm);
                    y_train1 = [y_train(:) ; y_test(:)];
                end
                n = length(y_train1);
                w_train = 1/n*ones(n,1);

                load(fullfile(pwd, ism_tt_dir, sprintf('ism_onset_tt_tlag%03d_twin%03d.mat', tlag, twin)));
                
                X_test = [X_train ; X_test];
                y_test = [y_train(:) ; y_test(:)];
                X_test = X_test(:, mask_ism);
                
                X_train = X_train1;
                y_train = y_train1;

                predictors = predictors_ism;

            case {'train_across_test_ism', 'train_across_test_stm', 'train_across_test_across'}
                load(fullfile(pwd, ism_tt_dir, sprintf('ism_onset_tt_tlag%03d_twin%03d.mat', tlag, twin)));
                X_train_ism = X_train(:, mask_ism);
                y_train_ism = y_train(:);
                w_train_ism = 1/length(y_train_ism)*ones(length(y_train_ism), 1);
                X_test_ism = X_test(:, mask_ism);
                y_test_ism = y_test(:);
                load(fullfile(pwd, stm_tt_dir, sprintf('stm_onset_tt_tlag%03d_twin%03d.mat', tlag, twin)));
                X_train_stm = X_train(:, mask_stm);
                X_train_stm = X_train_stm(:, idx_order_stm);
                y_train_stm = y_train(:);
                w_train_stm = 1/length(y_train_stm)*ones(length(y_train_stm), 1);
                X_test_stm = X_test(:, mask_stm);
                X_test_stm = X_test_stm(:, idx_order_stm);
                y_test_stm = y_test(:);

                if train_on_time0
                    X_train = [eval(sprintf('X_train_ism%03d', twin)) ; eval(sprintf('X_train_stm%03d', twin))];
                    y_train = [eval(sprintf('y_train_ism%03d', twin)) ; eval(sprintf('y_train_stm%03d', twin))];
                    w_train = [eval(sprintf('w_train_ism%03d', twin)) ; eval(sprintf('w_train_stm%03d', twin))];
                else
                    X_train = [X_train_ism ; X_train_stm];
                    y_train = [y_train_ism ; y_train_stm];
                    w_train = [w_train_ism ; w_train_stm];
                end
                

                switch mode
                case 'train_across_test_ism'
                    X_test = [X_train_ism ; X_test_ism];
                    y_test = [y_train_ism ; y_test_ism];
                case 'train_across_test_stm'
                    X_test = [X_train_stm ; X_test_stm];
                    y_test = [y_train_stm ; y_test_stm];
                case 'train_across_test_across'
                    X_test = [X_test_ism ; X_test_stm];
                    y_test = [y_test_ism ; y_test_stm];
                end
                predictors = predictors_ism;
            
        end

        age_mask = ismember(predictors, 'age');
        X_train(:,age_mask) = X_train(:,age_mask)./365;
        X_test(:,age_mask) = X_test(:,age_mask)./365;
        y_train = double(y_train);
        y_test = double(y_test);   

        % keyboard     

        %% run AdaBoost
        nfolds = 10;
        cvfolds = cvpartition(y_train,'Kfold',nfolds);
        trainFullModel = 1;
        useParallel    = 0;

        missingDataHandler = struct('method','none');
        method = 'Boosting';

        T = 100;
        boostingOpts = boostedHII_setOpts(T);
        boostingOpts.missingDataOpts = struct('method','abstain');
        missingDataHandler.method = boostingOpts.missingDataOpts.method;
        if train_on_time0 & tlag==twin
            cvres_full = boostedHII_cv(X_train, y_train, w_train, boostingOpts, cvfolds, trainFullModel, useParallel);

        elseif train_on_time0 & tlag~=twin
            cvres_full = cvres_full;
            % display(['train on time 0: ' num2str(tlag) ', ' num2str(twin)])
        else
            cvres_full = boostedHII_cv(X_train, y_train, w_train, boostingOpts, cvfolds, trainFullModel, useParallel);
        end

        prb_test = boostedHII_predict(X_test,cvres_full.clf_full,'probability');
        
        prb_to = boostedHII_predict(X_train,cvres_full.clf_full,'probability');
        if tlag==twin
            prb_from = boostedHII_predict(X_test,cvres_full.clf_full,'probability');
        end
        prb_test_proj = project_prb(prb_to, prb_from, prb_test);


        if tlag==twin & (strcmp(mode,'ism_only') | strcmp(mode,'stm_only'))
            fpr = 1- cvres_full.cv_results.specificities;
            fpr = fpr(:);
            tpr = cvres_full.cv_results.sensitivities;
            tpr = tpr(:);
            thr = cvres_full.cv_results.decision_thresholds;
            thr = thr(:);
            auc = cvres_full.cv_results.AUC;
            % keyboard
        else
            if project
                [fpr,tpr,thr,auc] = perfcurve(y_test, prb_test_proj, 1);
            else
                [fpr,tpr,thr,auc] = perfcurve(y_test, prb_test, 1);
            end
        end

        roc = sortrows([fpr tpr, thr],1);

        [feat, index] = get_picked_features(cvres_full, predictors);
        auc_all = [auc_all auc];
        model(ii).time_lag = tlag;
        model(ii).time_window = twin;
        model(ii).auc = auc;
        model(ii).feature_select = feat;
        model(ii).feature_all = predictors;
        model(ii).idx = index;
        model(ii).X_train = X_train;
        model(ii).X_test = X_test;
        model(ii).y_train = y_train;
        model(ii).y_test = y_test;
        model(ii).w_train = w_train;
        model(ii).roc = roc;
        model(ii).predictor = cvres_full;
    end

    fname = ['boosted_hii_' mode sim_tag top_tag last_tag manual_tag t0_tag proj_tag tr_by_age_tag te_by_age_tag '.mat'];
    save(fullfile(pwd, results_dir, fname), 'model')
end
