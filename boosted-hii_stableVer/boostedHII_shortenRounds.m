function clf = boostedHII_shortenRounds(X,y,clf,max_round)

% First, we need to throw away weak learners that exceed max_round
for j=1:numel(clf.weak_learners)
    max_ind = numel(clf.weak_learners{j});
    for k=1:numel(clf.weak_learners{j})
        if clf.weak_learners{j}{k}.boosting_round > max_round
            max_ind = k-1;
            break;
        end
    end
    clf.weak_learners{j} = clf.weak_learners{j}(1:max_ind);
end

% Also check the clf.missingDataHandler
if strcmp(clf.missingDataHandler.method,'abstain')
    % Then we are good to go
elseif strcmp(clf.missingDataHandler.method,'mean')    
    if ~isfield(clf.boostingOpts,'missingDataFilterPct')
        missingDataFilterPct = 0;
    else
        missingDataFilterPct = clf.boostingOpts.missingDataFilterPct;
    end
        
    for j=1:size(X,2)
        locs = find(isnan(X(:,j)));
        if 100*(size(X,1)-numel(locs))/size(X,1) < missingDataFilterPct
            X(:,j) = 0;
            continue;
        end
        X(locs,j) = clf.missingDataHandler.imputedVals(j);
    end
else
    error('Unknown missingDataMethod');
end

% Now retrain the stacking
clf.stacking = boostedHII_trainStack2(X,y,clf.weak_learners,clf.boostingOpts);
clf.boostingOpts.T = max_round;
