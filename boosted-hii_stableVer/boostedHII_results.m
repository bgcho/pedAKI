function boostMissingData_results

suffix1 = '_ver2';
suffixes = {'','_imputed'};
colors = {'r','b'};
legendStrs = {'Abstained','Imputed'};

if 0
for k=1:numel(suffixes)
    suffix = suffixes{k};
    
    hrsbacks = [1,2,4,6];
    AUC = zeros(numel(hrsbacks),1);
    sensitivity = zeros(numel(hrsbacks),1);
    specificity = zeros(numel(hrsbacks),1);
    ppv = zeros(numel(hrsbacks),1);
    npv = zeros(numel(hrsbacks),1);
    for j = 1:numel(hrsbacks)
        hrsback = hrsbacks(j);
        load(sprintf('../../Desktop/HIRBA/boostMissingData%s%s_%dhr_cv.mat',suffix1,suffix,hrsback));

        T = numel(decision_stumps{1});
        nfolds = cv.NumTestSets;
        pred = zeros(size(X,1),1);
        for foldNum = 1:nfolds    
            Xtest = X(cv.test(foldNum),:);
            pred(cv.test(foldNum)) = evaluateClassifier(Xtest,decision_stumps{foldNum});
        end

        locs_pos = find(y==1);
        locs_neg = find(y==-1);

        [~,~,~,AUC(j)] = perfcurve(y,pred,1);
        sensitivity(j) = sum(pred(locs_pos) > 0)/numel(locs_pos);
        specificity(j) = sum(pred(locs_neg) < 0)/numel(locs_neg);
        ppv(j) = sum(pred(locs_pos) > 0) / sum(pred > 0);
        npv(j) = sum(pred(locs_neg) < 0) / sum(pred < 0);
    end

    figure(1); if k == 1; clf; else; hold on; end;
    scatter(hrsbacks,AUC,150,colors{k},'filled');
    title('AUC');
    xlabel('Hours before intervention');
    set(gca,'XTick',hrsbacks);
    set(gca,'XDir','reverse');
    if k==2; grid minor; end;

    figure(2); if k == 1; clf; else; hold on; end;
    scatter(hrsbacks,sensitivity,150,colors{k},'filled');
    title('Sensitivity');
    xlabel('Hours before intervention');
    set(gca,'XTick',hrsbacks);
    set(gca,'XDir','reverse');
    if k==2; grid minor; end;

    figure(3); if k == 1; clf; else; hold on; end;
    scatter(hrsbacks,specificity,150,colors{k},'filled');
    title('Specificity');
    xlabel('Hours before intervention');
    set(gca,'XTick',hrsbacks);
    set(gca,'XDir','reverse');
    if k==2; grid minor; end;

    figure(4); if k == 1; clf; else; hold on; end;
    scatter(hrsbacks,ppv,150,colors{k},'filled');
    title('PPV');
    xlabel('Hours before intervention');
    set(gca,'XTick',hrsbacks);
    set(gca,'XDir','reverse');
    if k==2; grid minor; end;

    figure(5); if k == 1; clf; else; hold on; end;
    scatter(hrsbacks,npv,150,colors{k},'filled');
    title('NPV');
    xlabel('Hours before intervention');
    set(gca,'XTick',hrsbacks);
    set(gca,'XDir','reverse');
    if k==2; grid minor; end;


end
for j=1:5; figure(j); legend(legendStrs,'Location','Best'); end;
end


% Now provide plots of robustness to missing data
if 1
missingDataFracts = [0.1,0.15,0.2,0.25,0.4,0.5,0.6,0.75];
for k=1:1%numel(suffixes)
    suffix = suffixes{k};
    
    hrsback = 1;
    load(sprintf('../../Desktop/HIRBA/boostMissingData%s%s_%dhr_cv.mat',suffix1,suffix,hrsback));
    Xsave = X;
    medvals = zeros(size(X,2),1);
    for j=1:size(X,2)
        medvals(j) = median(X(~isnan(X(:,j)),j));
    end
        
    AUC = zeros(numel(missingDataFracts),1);
    sensitivity = zeros(numel(missingDataFracts),1);
    specificity = zeros(numel(missingDataFracts),1);
    ppv = zeros(numel(missingDataFracts),1);
    npv = zeros(numel(missingDataFracts),1);
    for j=1:numel(missingDataFracts)
        missingDataFract = missingDataFracts(j);
        q = randperm(numel(X));
        q = q(1:round(missingDataFract*numel(X)));
        M = ones(size(X));
        M(q) = NaN;
        
        X = Xsave.*M;
        if strcmp(suffix,'')
            % Do nothing
        elseif strcmp(suffix,'_imputed')
            % Replace NaN with median values
            for m = 1:size(X,2)
                X(isnan(X(:,m)),m) = medvals(m);
            end
        else
            error('What?');
        end
        
        T = numel(decision_stumps{1});
        nfolds = cv.NumTestSets;
        pred = zeros(size(X,1),1);
        for foldNum = 1:nfolds    
            Xtest = X(cv.test(foldNum),:);
            pred(cv.test(foldNum)) = evaluateClassifier(Xtest,decision_stumps{foldNum});
        end

        locs_pos = find(y==1);
        locs_neg = find(y==-1);

        [g1,g2,g3,AUC(j)] = perfcurve(y,pred,1);
        sensitivity(j) = sum(pred(locs_pos) > 0)/numel(locs_pos);
        specificity(j) = sum(pred(locs_neg) < 0)/numel(locs_neg);
        ppv(j) = sum(pred(locs_pos) > 0) / sum(pred > 0);
        npv(j) = sum(pred(locs_neg) < 0) / sum(pred < 0);
    end

    figure(1); if k == 1; clf; else; hold on; end;
    scatter(100*missingDataFracts,AUC,150,colors{k},'filled');
    title('AUC');
    xlabel('Missing Data Percentage (%)');
    set(gca,'XTick',100*missingDataFracts);
    if k==2; grid minor; end;

    figure(2); if k == 1; clf; else; hold on; end;
    scatter(100*missingDataFracts,sensitivity,150,colors{k},'filled');
    title('Sensitivity');
    xlabel('Missing Data Percentage (%)');
    set(gca,'XTick',100*missingDataFracts);
    if k==2; grid minor; end;

    figure(3); if k == 1; clf; else; hold on; end;
    scatter(100*missingDataFracts,specificity,150,colors{k},'filled');
    title('Specificity');
    xlabel('Missing Data Percentage (%)');
    set(gca,'XTick',100*missingDataFracts);
    if k==2; grid minor; end;

    figure(4); if k == 1; clf; else; hold on; end;
    scatter(100*missingDataFracts,ppv,150,colors{k},'filled');
    title('PPV');
    xlabel('Missing Data Percentage (%)');
    set(gca,'XTick',100*missingDataFracts);
    if k==2; grid minor; end;

    figure(5); if k == 1; clf; else; hold on; end;
    scatter(100*missingDataFracts,npv,150,colors{k},'filled');
    title('NPV');
    xlabel('Missing Data Percentage (%)');
    set(gca,'XTick',100*missingDataFracts);
    if k==2; grid minor; end;
    
end
for j=1:5; figure(j); legend(legendStrs,'Location','Best'); end;
end

end

function y = evaluateClassifier(X,decision_stumps)
    n = size(X,1);
    
    y = zeros(n,1);
    for j=1:numel(decision_stumps)
        ds = decision_stumps{j};
        locs = find(~isnan(X(:,ds.i)));
        if isfield(ds,'alpha_bias')
            y(locs) = y(locs) + ds.alpha_bias;
        end
        
        % First assume it's positive and then double correct the negatives
        y(locs) = y(locs) + ds.alpha;
        neg_locs = find(X(locs,ds.i) < ds.b);
        y(locs(neg_locs)) = y(locs(neg_locs)) - 2*ds.alpha;
    end
end
