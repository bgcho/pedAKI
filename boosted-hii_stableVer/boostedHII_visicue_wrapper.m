function [cvres,hourly_results] = boostedHII_visicue_wrapper(r,cv)
%r = load('../../Desktop/HIRBA/data/ffm51_full.mat');


% Whether or not to partition by hospital when performing cross-validation
% Setting this value to 1 makes the results closer to reality (applying
% learned model to a new hospital)
partition_by_hospital = 1;

saveOutPrefix = '../../Desktop/HIRBA/results_boostedHII/logistic_stump_comparison_new/decisionstump_new_stacking_method_results_over_rounds_partitioned_by_hospital_no_shock_index';%decisionstump_new_stacking_method_results_over_rounds_partitioned_by_hospital';

if isfield(r,'age')
    r.feat_list(end+1) = 300007;
    for j=1:numel(r.X)
        r.X{j}(end+1,:) = r.age(j);
    end
end

labelMapping = [];
missingValFn = 'NaN';
featsToUse = []; % Grab all 51 features

% Use the 1-hour before intervention data to train the model
hrsback = 1;
[X,y,r_index] = generateClassificationDataset(r,hrsback,labelMapping,featsToUse,missingValFn);
[n,p] = size(X);

% Blank out the 14th, 29th, 30th, and 37th features (PT,RespRate,SpO2,PaO2)
blankOutNodes = [14,23,24,29,30,37];
keepNodes = setdiff(1:p,blankOutNodes);
X(:,blankOutNodes) = NaN;

% Generate a 10-fold cross-validation partition
if ~exist('cv','var')
    if partition_by_hospital
        cv = generate_hosp_cv(r,r_index,10);
    else
        cv = cvpartition(y,'Kfold',10);
    end
end

runLR = 0;
runMultiImpute = 0;
runSVM = 0;


if runMultiImpute == 1
    clf = cell(cv.NumTestSets,1);
    y_test = zeros(n,1);
    for foldNum = 8%1:cv.NumTestSets  
        fprintf('Working on fold #%d of %d\n',foldNum,cv.NumTestSets);pause(1e-5);
        if foldNum == 8
            foldNumToLoad = 7;
        else
            foldNumToLoad = foldNum;
        end
        X_imputed = csvread(sprintf('../../Desktop/X/X_cv_%d_imputed1.csv',foldNumToLoad),1,1);
        X_imputed2 = zeros(size(X_imputed,1),p);
        X_imputed2(:,keepNodes) = X_imputed;
        X_imputed = X_imputed2; clear X_imputed2;
        
        mnv_mu = mean(X_imputed);
        covMat = (1/size(X_imputed,1))*(X_imputed - repmat(mnv_mu,size(X_imputed,1),1))'*(X_imputed - repmat(mnv_mu,size(X_imputed,1),1));
        
        mnv_mu(blankOutNodes) = NaN;
        
        T = 100;
        boostingOpts = boostedHII_setOpts(T);
        boostingOpts.missingDataOpts = struct('method','multivariate_normal_model',...
                                              'means',mnv_mu',...
                                              'covMat',covMat);
        boostingOpts.stackingOpts.use = 0;
        
        clf{foldNum} = boostedHII_train(X(cv.training(foldNum),:),y(cv.training(foldNum)),boostingOpts,1,0);
        y_test(cv.test(foldNum)) = boostedHII_predict(X(cv.test(foldNum),:),clf{foldNum});
    end
end

if runLR == 1
    lr_impute = 1;
    
    fprintf(1,'Working on full model...\n');pause(1e-5);
    lr_clf_full = lr_train(X,y,lr_impute);
    lr_clf = cell(cv.NumTestSets,1);
    lr_y_test = zeros(n,2);
    for foldNum = 1:cv.NumTestSets
        fprintf(1,'Working on fold #%d of %d\n',foldNum,cv.NumTestSets);pause(1e-5);
        lr_clf{foldNum} = lr_train(X(cv.training(foldNum),:),y(cv.training(foldNum)),lr_impute);
        lr_y_test(cv.test(foldNum),:) = lr_predict(X(cv.test(foldNum),:),lr_clf{foldNum});
    end
    cvres = [];
    cvres.clf_full = lr_clf_full;
    cvres.clf = lr_clf;
    cvres.cv_results=prediction_results(y,lr_y_test(:,2),1,0.05:0.025:0.95);
    cvres.y_test = lr_y_test(:,2);
    save('../../Desktop/HIRBA/results_boostedHII/logistic_stump_comparison_new/logistic_regression_results_imputed.mat','X','y','cv','cvres');


    lr_impute = 0;
    
    fprintf(1,'Working on full model...\n');pause(1e-5);
    lr_clf_full = lr_train(X,y,lr_impute);
    lr_clf = cell(cv.NumTestSets,1);
    lr_y_test = zeros(n,2);
    for foldNum = 1:cv.NumTestSets
        fprintf(1,'Working on fold #%d of %d\n',foldNum,cv.NumTestSets);pause(1e-5);
        lr_clf{foldNum} = lr_train(X(cv.training(foldNum),:),y(cv.training(foldNum)),lr_impute);
        lr_y_test(cv.test(foldNum),:) = lr_predict(X(cv.test(foldNum),:),lr_clf{foldNum});
    end
    cvres = [];
    cvres.clf_full = lr_clf_full;
    cvres.clf = lr_clf;
    cvres.cv_results=prediction_results(y,lr_y_test(:,2),1,0.05:0.025:0.95);
    cvres.y_test = lr_y_test(:,2);
    save('../../Desktop/HIRBA/results_boostedHII/logistic_stump_comparison_new/logistic_regression_results_abstain.mat','X','y','cv','cvres');
end



%Ts = 500:-100:300;
Ts = 100;
if 0
    for j=1:numel(Ts)
        T = Ts(j);
        boostingOpts = boostedHII_setOpts(T);
        cvres = boostedHII_cv(X,y,boostingOpts,cv,1,1);
        [hrsbacks,hourly_results] = getHourlyResults(r,r_index,cvres,cv,labelMapping,featsToUse,missingValFn);
        save(sprintf('%s_T%d.mat',saveOutPrefix,T),'X','cv','cvres','r_index','y','hrsbacks','hourly_results');    
    end    
else
boostingOpts = boostedHII_setOpts(Ts(1));
cvres = boostedHII_cv(X,y,boostingOpts,cv,1,0);
[hrsbacks,hourly_results] = getHourlyResults(r,r_index,cvres,cv,labelMapping,featsToUse,missingValFn);
save(sprintf('%s_T%d.mat',saveOutPrefix,Ts(1)),'X','cv','cvres','r_index','y','hrsbacks','hourly_results');    
cvres_full = cvres;

for j=2:numel(Ts)
    T = Ts(j);
    
    cvres.clf_full = boostedHII_shortenRounds(X,y,cvres_full.clf_full,T);
    y_test = zeros(n,1);
    for foldNum = 1:cv.NumTestSets
        tr = cv.training(foldNum);
        te = cv.test(foldNum);
        cvres.clf{foldNum} = boostedHII_shortenRounds(X(tr,:),y(tr),cvres_full.clf{foldNum},T);
        y_test(te) = boostedHII_predict(X(te,:),cvres.clf{foldNum});
    end    
    cvres.cv_results = prediction_results(y,y_test,1,0.05:0.05:0.95);
    cvres.y_test = y_test;
    [hrsbacks,hourly_results] = getHourlyResults(r,r_index,cvres,cv,labelMapping,featsToUse,missingValFn);
    
    save(sprintf('%s_T%d.mat',saveOutPrefix,T),'X','cv','cvres','r_index','y','hrsbacks','hourly_results');    
end    
end

end

function clf = lr_train(X,y,impute)
    if ~exist('impute','var')
        impute = 1;
    end
    
    % First, need to fill in missing values
    [n,p] = size(X);
    impute_vals = zeros(p,1);
    remove_feats = [];
    XM = double(~isnan(X));
    for j=1:p
        impute_vals(j) = mean(X(~isnan(X(:,j)),j));
        if isnan(impute_vals(j)) 
            impute_vals(j) = 0; 
            remove_feats(end+1) = j;
        end
        if ~impute
            impute_vals(j) = 0;
        end
        X(isnan(X(:,j)),j) = impute_vals(j);
    end

    X(:,remove_feats) = [];
    XM(:,remove_feats) = [];
    if impute
        [B,~,~] = mnrfit(X,y+1);
    else
        [B,~,~] = mnrfit([X,XM],y+1);
    end
    
    clf = struct('impute',impute,...
                 'impute_vals',impute_vals,...
                 'remove_feats',remove_feats,...
                 'B',B);
end

function y = lr_predict(X,clf)
    [n,p] = size(X);
    
    XM = double(~isnan(X));
    for j=1:p
        X(isnan(X(:,j)),j) = clf.impute_vals(j);
    end
    
    X(:,clf.remove_feats) = [];
    XM(:,clf.remove_feats) = [];
    if clf.impute
        y = mnrval(clf.B,X);
    else
        y = mnrval(clf.B,[X,XM]);
    end
end

function svm_train(X,y)
    
end

function svm_predict(X,clf)

end

function [hrsbacks,hourly_results] = getHourlyResults(r,r_index,cvres,cv,labelMapping,featsToUse,missingValFn)
% Now test the classifier on other hours before intervention
hrsbacks = [1,2,4,6];
hourly_results = cell(numel(hrsbacks),1);
% We've already tested on 1-hour before
hourly_results{1} = cvres.cv_results;
% Apply the learned 1-hour before intervention model to 2,4,6 hours before
% intervention
for j=2:numel(hrsbacks)
    hrsback = hrsbacks(j);
    [X2,y2,r2_index] = generateClassificationDataset(r,hrsback,labelMapping,featsToUse,missingValFn);
    
    y_test = zeros(size(X2,1),1);
    for foldNum = 1:cv.NumTestSets
        [~,test_inds] = intersect(r2_index,r_index(cv.test(foldNum)));
        y_test(test_inds) = boostedHII_predict(X2(test_inds,:),cvres.clf{foldNum});
    end
    
    hourly_results{j} = prediction_results(y2,y_test);
    hourly_results{j}.y_test = y_test;
    hourly_results{j}.r_index = r2_index;
end
end

function cv = generate_hosp_cv(r,r_index,nfolds)

% Partition by hospital
[uh,~,ih] = unique(r.hospid_list);
if nfolds > numel(uh)
    error('');
end
tmpvar1 = reshape(repmat((1:nfolds)',1,ceil(numel(uh)/nfolds)),nfolds*ceil(numel(uh)/nfolds),1);
tmpvar1 = tmpvar1(1:numel(uh));
hospTestSets = tmpvar1(randperm(numel(tmpvar1)));

TRAINING = ones(nfolds,numel(r_index));
TEST = zeros(nfolds,numel(r_index));
for j=1:numel(r_index)
    ind = find(uh==r.hospid_list(r_index(j)));
    TRAINING(hospTestSets(ind),j) = 0;
    TEST(hospTestSets(ind),j) = 1;
end

clearvars -except nfolds TRAINING TEST;
cv = [];
cv.NumTestSets = nfolds;
cv.training = @(i) logical(TRAINING(i,:));
cv.test = @(i) logical(TEST(i,:));

end