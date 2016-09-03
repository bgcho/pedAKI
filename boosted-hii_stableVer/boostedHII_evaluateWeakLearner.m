function y = boostedHII_evaluateWeakLearner(X,weak_learners,feat_ind,stump_inds)
    if ~exist('stump_inds','var'); stump_inds = []; end;
    
    if ~numel(stump_inds); stump_inds = 1:numel(weak_learners{feat_ind}); end;
    
    n = size(X,1);
    y = zeros(n,1);
    
    if size(X,2) > 1
        x = X(:,feat_ind);
    else
        x = X;
    end
    
    for j=1:numel(stump_inds)
        stump_ind = stump_inds(j);
        ds = weak_learners{feat_ind}{stump_ind};
        locs = find(~isnan(x));
        ycurr = zeros(n,1);
        if isfield(ds,'bias')
            ycurr(locs) = ds.bias;
        end
        
        switch ds.type
            case 'constant'
                % There's nothing left to do
            case 'logistic'
                ycurr(locs) = ycurr(locs) + ds.alpha*(2./(1+exp(-(ds.a*x(locs)+ds.b))) - 1);
            case 'stump'
                % First assume it's positive and then double correct the negatives
                ycurr(locs) = ycurr(locs) + ds.alpha;
                neg_locs = find(x(locs) < ds.threshold);
                ycurr(locs(neg_locs)) = ycurr(locs(neg_locs)) - 2*ds.alpha;
            otherwise
                error('Unknown weak learner type');
        end
        
        y = y + ycurr;
    end
end