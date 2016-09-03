addpath('.\boosted-hii_stableVer')

clc; clear all; close all;
tlag_all = [6:24];
twin_all = [6,12];
combination = combvec(tlag_all, twin_all).';
combination = combination(combination(:,1)>=combination(:,2),:);

auc = [];

for ii = 1:size(combination,1)
    fname = ['..\..\raw_data_onset_tlag' sprintf('%03d',tlag_all(ii)) '_twin' sprintf('%03d', twin_all(ii)) '.mat'];
    load(fname)

    idx= [3,6:25,27:size(data,2)];



    tmp.x = nan(size(data,1),length(idx));

    for ft = 1:length(idx)
        tmp.x(:,ft) = cell2mat(data(:,idx(ft)));
    end

    tmp.y = cell2mat(data(:,5));

    tmp.col = {'age', 'nsbp_min', 'nsbp_mean', 'ndbp_min', 'ndbp_mean', 'hr_max', ...
        'hr_mean', 'spo2_min', 'spo2_mean', 'ratio_pao2_flo2_min', ...
        'hemoglobin_min', 'temperature_max', 'wbc_max', 'platelet_min', ...
        'bilirubin_max', 'albumin_min', 'ph_min', 'urine_mean', ...
        'potassium_max', 'calcium_min', 'glucose_max', ...
        'creatinine_min', 'creatinine_max', 'creatinine_mean', ...
        'lactic_acid_min', 'lactic_acid_max', 'lactic_acid_mean', 'bun_min', ...
        'bun_max', 'bun_mean'};


    clear data
    tmp.y(tmp.y~=0) = 1;
    X = tmp.x; 
    y = tmp.y;

    %X = zscore(X);
    %idx = [1,3,5,7,9,11,12,13,14,15,16,17,18,19,20,21,27];
    idx = [1:21,25:27];
    tmp.col = tmp.col(idx);
    X(:,1) = X(:,1)./365;
    X = X(:,idx);
    % X = X(:,[1,3,5,7,9,11,12,13,14,15:21,24,27]);

    %% run AdaBoost
    nfolds = 10;
    cvfolds = cvpartition(y,'Kfold',nfolds);
    trainFullModel = 1;
    useParallel    = 0;

    missingDataHandler = struct('method','none');
    method = 'Boosting';

    [n,p] = size(X);

    T = 100;
    boostingOpts = boostedHII_setOpts(T);
    boostingOpts.missingDataOpts = struct('method','abstain');
    missingDataHandler.method = boostingOpts.missingDataOpts.method;
    cvres_full = boostedHII_cv(X,y,boostingOpts,cvfolds,trainFullModel,useParallel);


    [feat,index] = get_picked_features(cvres_full,tmp.col);
    auc = [auc cvres_full.cv_results.AUC]
    model(ii).time_lag = tlag_all(ii);
    model(ii).time_window = twin_all(ii);
    model(ii).auc = cvres_full.cv_results.AUC;
    model(ii).feature = feat;
end
