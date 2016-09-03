function cvres = boostedHII_cv(X,y,w_example,boostingOpts,cv,trainFullModel,useParallel)


[n,p] = size(X);
if ~exist('cv','var'); cv = cvpartition(y,'Kfold',10); end;
if ~exist('trainFullModel','var'); trainFullModel = 1; end;
if ~exist('useParallel','var'); useParallel = 1; end;
if ~exist('w_example', 'var'); w_example = (1/n)*ones(n,1); end;

if useParallel
    if matlabpool('size') > 0
        matlabpool('close');
    end
    myCluster = parcluster('local');
    if myCluster.NumWorkers < 8
        myCluster.NumWorkers = 8;
        saveProfile(myCluster);
    end
    matlabpool('open',8);
end


nfolds = cv.NumTestSets;
clf = cell(nfolds+trainFullModel,1);
y_test_container = cell(nfolds+trainFullModel,1);
if useParallel
    parfor foldNum = 1:(nfolds+trainFullModel)
        if foldNum == nfolds + 1
            fprintf(1,'Working on full model\n');
            clf{foldNum} = boostedHII_train(X,y,w_example, boostingOpts);
        else    
            fprintf(1,'Working on fold #%d of %d\n',foldNum,nfolds);
            tr = cv.training(foldNum);
            te = cv.test(foldNum);
    
            clf{foldNum} = boostedHII_train(X(tr,:),y(tr),w_example(tr),boostingOpts);
            y_test_container{foldNum} = boostedHII_predict(X(te,:),clf{foldNum});
        end            
    end
else
    for foldNum = 1:(nfolds+trainFullModel)
        if foldNum == nfolds + 1
            fprintf(1,'Working on full model\n');
            clf{foldNum} = boostedHII_train(X,y,w_example, boostingOpts);
        else    
            fprintf(1,'Working on fold #%d of %d\n',foldNum,nfolds);
            tr = cv.training(foldNum);
            te = cv.test(foldNum);           
    
            clf{foldNum} = boostedHII_train(X(tr,:),y(tr),w_example(tr),boostingOpts);
            y_test_container{foldNum} = boostedHII_predict(X(te,:),clf{foldNum},'');
            %obj_func = sum(exp(-y.*clo));
        end            
    end
end    

if trainFullModel
    clf_full = clf{end};
    clf = clf(1:end-1);
end
y_test = zeros(n,1);

for foldNum = 1:nfolds
    y_test(cv.test(foldNum)) = y_test_container{foldNum};
end
clear y_test_container;

thr = linspace(min(y_test),max(y_test),1000);
cv_results = prediction_results(y,y_test,1,thr);
%cv_results = prediction_results(y,y_test,1,0.05:0.05:0.95);

cvres.y_test = y_test;
cvres.clf = clf;
cvres.cv_results = cv_results;
if trainFullModel
    cvres.clf_full = clf_full;
end

if useParallel
    if matlabpool('size') > 0
        matlabpool('close');
    end
end
end