function clf = boostedHII_train(X,y,w_example,boostingOpts,verbosity,useParallel)    
% FUNCTION boostedHII_train(X,y,boostingOpts)
% *** INPUTS ***
%   X:            n x p feature data matrix.  
%   y:            n x 1 label vector.
%   boostingOpts:
%   verbosity:
%
% *** OUTPUTS ***
%   clf:          a structure 
[n,p] = size(X);

if ~exist('verbosity','var'); verbosity = []; end;
if ~exist('useParallel','var'); useParallel = 0; end;
if ~exist('w_example', 'var'); w = (1/n)*ones(n,1); else w = w_example; end;

if ~numel(verbosity); verbosity = 1; end;

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

T = boostingOpts.T;

% Add a bias feature
X = [X,ones(n,1)];
p = p + 1;

missingDataHandler = boostingOpts.missingDataOpts;
if strcmp(missingDataHandler.method,'abstain')
    % Then we are fine -- nothing to do
elseif strcmp(missingDataHandler.method,'mean')
    if ~isfield(boostingOpts,'missingDataFilterPct')
        boostingOpts.missingDataFilterPct = 0;
    end
        
    missingDataHandler.imputedVals = zeros(p,1);    
    missingDataHandler.missingDataFilterPct = boostingOpts.missingDataFilterPct;
    for j=1:p
        locs = find(isnan(X(:,j)));
        if 100*(n-numel(locs))/n < boostingOpts.missingDataFilterPct
            X(:,j) = 0;
            continue;
        end
        missingDataHandler.imputedVals(j) = mean(X(~isnan(X(:,j)),j));
        X(locs,j) = missingDataHandler.imputedVals(j);
    end
elseif strcmp(missingDataHandler.method,'multivariate_normal_model')
    % Then use the multivariate normal to impute values on each trial
    % First, get all of the distinct missing patterns
    [MP,~,pattern_inds] = unique(~isnan(X(:,1:end-1)),'rows');
    [pattern_index,pattern_inds] = sort(pattern_inds,'ascend');
    j = 0;
    while j < n
        fprintf('%d\n',j);pause(1e-5);
        j = j + 1;
        jStart = j;
        index = pattern_index(j);
        while j <= n && pattern_index(j) == index
            j = j + 1;
        end
        j = j - 1;
        
        inds = pattern_inds(jStart:j);
        
        exists_locs = find(MP(index,:));
        missing_locs = setdiff(1:p-1,exists_locs);
        
        tmpCov = missingDataHandler.covMat;
        tmpCov = tmpCov([missing_locs,exists_locs],:);
        tmpCov = tmpCov(:,[missing_locs,exists_locs]);
        
        S12 = tmpCov(1:numel(missing_locs),(numel(missing_locs)+1):end);
        S22 = tmpCov((numel(missing_locs)+1):end,(numel(missing_locs)+1):end);
        
        impute_vals = repmat(missingDataHandler.means(missing_locs),1,numel(inds)) + S12*(S22\(X(inds,exists_locs)' - repmat(missingDataHandler.means(exists_locs),1,numel(inds))));
        
        X(inds,missing_locs) = impute_vals';
    end  
        
else
    error('Unknown missingDataMethod');
end


% Check the label vector
uy = unique(y);
if numel(uy) ~= 2
    error('Label vector y must contain 2 classes.');
end
if uy(1) == -1 && uy(2) == 1
    % Then we are all good
elseif uy(1) == 0 && uy(2) == 1
    % Convert from {0, 1} -> {-1, +1}
    y = 2*y - 1;
else
    error('Unknown label vector classes.');
end

% The initial classifier output (all zeros)
clo = zeros(n,1);

printmsg('Initial objective function value: %.2f\n',n);

% Initialize the decision stumps
weak_learners = cell(p,1);
for j=1:p
    weak_learners{j} = [];
end

% Set the bias
weak_learners{p}{1} = struct('type','constant',...
                             'boosting_round',1,...
                             'bias',resetBias(y,w));
                           
clo = clo + boostedHII_evaluateWeakLearner(X,weak_learners,p,1);
% Update the weights
% After this update, we'll have sum(w(y==1)) == sum(w(y==-1))
w = w.*exp(-y.*clo); w = w/sum(w);

printmsg('Objective function after round #1 (bias computation): %.2f\n',objective_function(y,clo));
printmsg('Training error after round #1 (bias computation): %.2f\n',training_error(y,clo));
obj_prev = inf;
for t=2:T       
    % Check each feature for the best decision stump
    % The feature with minimum Z will be chosen
    if useParallel
        dsi = cell(p-1,1);
        Zi = zeros(p-1,1);
        parfor ii = 1:(p-1)
            [dsi{ii},Zi(ii)] = logreg1d(X(:,ii),y,w);
        end
        [Z,i] = min(Zi);
        ds = dsi{i};
        save(sprintf('../../Desktop/HIRBA/results_boostedHII/logistic_stump_comparison/tmp_results_round_%d.mat',t),'w','dsi','Zi');
    else
        i = 0;Z = inf;
        for ii = 1:(p-1)
%            [dsi,Zi] = logreg1d(X(:,ii),y,w);
%            if Zi <= Z
%                i = ii;
%                Z = Zi;
%                ds = dsi;
%            end
        
            [dsi,Zi] = decisionStump(X(:,ii),y,w);        
            if Zi <= Z
                i = ii;
                Z = Zi;
                ds = dsi;
            end
        end
    end
    s = numel(weak_learners{i}) + 1;
    weak_learners{i}{s} = ds;
    weak_learners{i}{s}.boosting_round = t;
                                   
    % Now update the weights
    clo_t = boostedHII_evaluateWeakLearner(X,weak_learners,i,s);
    w = w.*exp(-y.*clo_t); w = w/sum(w);
    
    % Now correct the bias
    s = numel(weak_learners{p}) + 1;
    weak_learners{p}{s} = struct('type','constant',...
                                 'boosting_round',t,...
                                 'bias',resetBias(y,w));
    
	clo_bias = boostedHII_evaluateWeakLearner(X,weak_learners,p,s);
	% Again, after this update, we should have sum(w(y==1)) == sum(w(y==-1))
	w = w.*exp(-y.*clo_bias); w = w/sum(w);
    
    clo = clo + clo_t + clo_bias;
    obj_fn = objective_function(y,clo);
    if obj_fn > obj_prev
        disp('hi');
    end
    obj_prev = obj_fn;
    printmsg('Objective function at round #%d:  %.2f\n',t,objective_function(y,clo));
    printmsg('Training error at round #%d:  %.2f\n',t,training_error(y,clo));
end


clf.weak_learners = weak_learners;
clf.stacking = boostedHII_trainStack2(X,y,weak_learners,boostingOpts);
%clf.stacking = train_stack2(X,y,weak_learners,boostingOpts.stackingOpts);
%clf.stacking = train_stack(X,y,weak_learners,boostingOpts.stackingOpts);
clf.boostingOpts = boostingOpts;
clf.missingDataHandler = missingDataHandler;

if useParallel
    if matlabpool('size') > 0
        matlabpool('close');
    end
end

end

    function printmsg(msg,varargin)
        %if verbosity >= 1
            fprintf(1,msg,varargin{:});
        %end
    end

    function obj_func = objective_function(y,clo)
        obj_func = sum(exp(-y.*clo));
    end

    function train_err = training_error(y,clo)
        train_err = sum(sign(clo)~=y)/numel(y);
    end

function alpha = resetBias(y,w,b)
% Picks the appropriate weighting on a constant output
% so that after updating weights, we have sum(w(y==1)) == sum(w(y==-1))
% b is the threshold
    if ~exist('b','var')
        b = 0;
    end
    
    frac_neg1 = sum(w(y==-1));
    frac_pos1 = sum(w(y==1));
        
    if b < 1
        alpha = 0.5*log(frac_pos1/frac_neg1);
    elseif b > 1
        alpha = 0.5*log(frac_neg1/frac_pos1);
    else
        error('b cannot be 1');
    end
end

    

function [wts,mus] = logreg1d_ver2(x,y,w)
    
    max_slope = 0.25;
    % Use this to initialize the search
    [ds,Z] = decisionStump(x,y,w);
    a = sign(ds.alpha)*4*max_slope;
    b = -1*a*ds.threshold;
    wts0 = [a;b];
    n = size(x,1);
    
    z = w.*y;
    X = [x,ones(n,1)];
    warning off all;
    wts = nlinfit(X,z,@(wts,X) (wts(3)*(2./(1+exp(-X*wts(1:2)))-1)),[wts0;1]);
    warning on all;

%    opts=optimoptions('fmincon','Display','off');
%    fnmin = @(wts) sum((z - wts(3)*(2./(1+exp(-X*wts(1:2)))-1)).^2);
%    wts = fmincon(fnmin, [wts0;1],[],[],[],[],[-inf,-inf,-inf],[inf,inf,inf],[],opts);

    if abs(wts(3)) < 1e-10
        wts = [0;0];
    else
        wts = sign(wts(3))*wts(1:2);
    end
    
    mus = 1./(1+exp(-X*wts));
end

function [ds,Z] = logreg1d(x,y,w)
    abstain_locs = isnan(x);
    
    W0 = sum(w(abstain_locs));
    
    if sum(abstain_locs) == numel(x)
        Z = W0;
        ds = struct('type','constant',...
                    'bias',0);
        return;
    end
    
    x(abstain_locs) = [];
    y(abstain_locs) = [];
    w(abstain_locs) = [];
    
    % First, compute a bias term so that sum(w(y==1)) == sum(w(y==-1))
    alpha_bias = resetBias(y,w);
    w = w.*exp(-y.*alpha_bias);
    
    n = numel(x);
    
    if 0
        wts = [0;0];
        X = [x,ones(n,1)];
        while 1
            wtsprev = wts;
            etas = X*wts;
            mus = 1./(1+exp(-etas));
            R = w.*mus.*(1-mus);
        
            l = X'*(R.*etas + w.*(0.5*(y+1) - mus));
            wts = (X'*bsxfun(@times,R,X)) \ l;
            if max(abs(wts - wtsprev)) < 1e-5
                break;
            end                
        end               
    else
        [wts,mus] = logreg1d_ver2(x,y,w);
    end
    
    % Now compute alpha
    % Need to use binary search for this
    alpha = alpha_linesearch(w,y,2*mus-1);
    
    % Now compute Z for this setting
    Z = W0 + sum(w.*exp(-alpha*u));
    ds = struct('type','logistic',...
                'a',wts(1),...
                'b',wts(2),...
                'alpha',alpha,...
                'bias',alpha_bias);
end
    

function alpha = alpha_linesearch(w,y,z)
    % Now compute alpha
    % Need to use binary search for this
    % We are looking for a root of the derivative
    % d/d_alpha = -sum(w(i)*y(i)*z(i)*exp(-y(i)*z(i)*alpha))
    u = y.*z;
    alpha_lb = -inf;
    alpha_ub = inf; alpha = 0;
    while 1
        Zd = -1*sum(w.*u.*exp(-alpha*u));
        if abs(Zd) < 1e-8
            break;
        end
        if Zd > 0
            alpha_ub = alpha;
            if isinf(alpha_lb)
                alpha = alpha - 1;
            else
                alpha = 0.5*(alpha_lb + alpha_ub);
            end
        else
            alpha_lb = alpha;
            if isinf(alpha_ub)
                alpha = alpha + 1;
            else
                alpha = 0.5*(alpha_lb + alpha_ub);
            end
        end
    end
end


function [ds,Z] = decisionStump(x,y,w)
    abstain_locs = isnan(x);

    W0 = sum(w(abstain_locs));

    if sum(abstain_locs) == numel(x)
        Z = W0;    
        ds = struct('type','constant',...
                    'bias',0);
        return;
    end
    
    x(abstain_locs) = [];
    y(abstain_locs) = [];
    w(abstain_locs) = [];
    
    % First, compute a bias term so that sum(w(y==1)) == sum(w(y==-1))
    alpha_bias = resetBias(y,w);
    w = w.*exp(-y.*alpha_bias);
    
    n = numel(x);
    
    % unique returns xu in sorted order
    [xu,~,iu] = unique(x);
    nu = numel(xu);
    wu = zeros(2,nu);
    % Old way that is slow
    %for j=1:nu
    %    wu(1,j) = sum(w(iu==j & y==-1));
    %    wu(2,j) = sum(w(iu==j & y==1));
    %end
    for j=1:n
        if y(j) == -1
            wu(1,iu(j)) = wu(1,iu(j)) + w(j);
        else
            wu(2,iu(j)) = wu(2,iu(j)) + w(j);
        end
    end
    % Sanity check -- sum(wu(:)) should equal sum(w)
    if abs(sum(wu(:)) - sum(w)) > 1e-10
        error('Something went wrong here!');
    end
    
    % This is the weighting if the decision boundary is less than the
    % smallest point
    Winc = sum(w(y==-1));
    Wcor = sum(w(y==1));
    best_i = 0;
    best_Z = Winc*Wcor;
    best_Winc = Winc;
    best_Wcor = Wcor;
    for i = 1:nu
        % 1:i are the points classified as -1
        % (i+1):nu are the points classified as +1
        % wu(1,i) are weights of points labeled y==-1
        % wu(2,i) are weights of points labeled y==1
        delta = wu(1,i) - wu(2,i);
        Wcor = Wcor + delta;
        Winc = Winc - delta;
        
        Z = Winc*Wcor;
        if Z < best_Z
            best_i = i;
            best_Z = Z;
            best_Winc = Winc;
            best_Wcor = Wcor;
        end
    end
    
    
    if best_i == 0 || best_i == nu
        if numel(xu) <= 1
            mdxu = 1;
        else
            mdxu = median(diff(xu));
        end
        
        if best_i == 0
            b = xu(1) - mdxu;
        else
            b = xu(nu) + mdxu;
        end
    else
        b = 0.5*(xu(best_i) + xu(best_i+1));
    end
   
    alpha = 0.5*log(best_Wcor/best_Winc);
    Z = W0 + 2*sqrt(best_Z);
    
    ds = struct('type','stump',...
                'threshold',b,...
                'alpha',alpha,...
                'bias',alpha_bias);
end


function stacking = train_stack2(X,y,weak_learners,stackingOpts)

if ~stackingOpts.use
    stacking = [];
    return;
end

[n,p] = size(X);

feats_picked = zeros(p,1);
for j=1:numel(weak_learners)
    if numel(weak_learners{j})
        feats_picked(j) = 1;
    end
end
feats_picked = find(feats_picked);
feats_notpicked = setdiff(1:p,feats_picked);

M = double(isnan(X));

clos = zeros(n,p);
for j=1:p
    clos(:,j) = boostedHII_evaluateWeakLearner_orig(X,weak_learners,j);
end
clo = sum(clos,2);

w = exp(-y.*clo); w = w/sum(w);

mf_coeffs = cell(p,1);
for j=1:p
    mf_coeffs{j} = [1,zeros(1,p)];
end

stacking_order = [];
for t=1:200
    best_j = 0;
    best_k = 0;
    best_c = [];
    best_innerprod = 0;
    for j=1:p
        for k=0:p
            if k == 0
                c = clos(:,j);
            else
                c = clos(:,j).*M(:,k);
            end
            innerprod = abs(sum(w.*y.*(c./sqrt(sum(c.^2)))));
            if innerprod > best_innerprod
                best_c = c;
                best_j = j;
                best_k = k;
                best_innerprod = innerprod;
            end
        end
    end
    
    alpha = alpha_linesearch(w,y,best_c);
    clo_curr = alpha*best_c;
    w = w.*exp(-y.*clo_curr); w = w/sum(w);

    
    stacking_order(t).j = best_j;
    stacking_order(t).k = best_k;
    stacking_order(t).alpha = alpha;
    
    if k == 0
        mf_coeffs{j}(1) = mf_coeffs{j}(1) + alpha;
    else
        mf_coeffs{j}(1) = mf_coeffs{j}(1) + alpha;
        mf_coeffs{j}(k+1) = mf_coeffs{j}(k+1) - alpha;
    end
    
    clo = clo + clo_curr;
    printmsg('Objective function at stacking round #%d:  %.2f\n',t,objective_function(y,clo));
    printmsg('Training error at stacking round #%d:  %.2f\n',t,training_error(y,clo));
end
stacking.stacking_order = stacking_order;
stacking.mf_coeffs = mf_coeffs;
end

function stacking = train_stack(X,y,weak_learners,stackingOpts)
    
if ~stackingOpts.use
    stacking = [];
    return;
end
frac = stackingOpts.energyFrac;
[n,p] = size(X);

feats_picked = zeros(p,1);
for j=1:numel(weak_learners)
    if numel(weak_learners{j})
        feats_picked(j) = 1;
    end
end
feats_picked = find(feats_picked);
feats_notpicked = setdiff(1:p,feats_picked);

M = double(~isnan(X));
if ~numel(find(M==0)) || frac == 0
    V = zeros(p,0);
    MF = ones(n,1);
else
    M(:,feats_notpicked) = 0;
    C = M'*M;
    [V,D] = eig(C);
    d = diag(D);
    [d,ds] = sort(d,'descend');
    d_frac = cumsum(d)/sum(d);
    ind_frac = find(d_frac >= frac,1,'first');
    if numel(ind_frac)
        d = d(1:ind_frac);
        V = V(:,ds(1:ind_frac));
    end
    % Now generate the meta-features
    MF = [ones(n,1),M*V];
end

% Now generate all of the new features
Z = zeros(n,numel(feats_picked)*size(MF,2));
feat_listing = zeros(1,numel(feats_picked));flc = 0;
eInd = 0;
for j=1:numel(feats_picked)
    feat_ind = feats_picked(j);
    
    clo = boostedHII_evaluateWeakLearner_orig(X,weak_learners,feat_ind);
    sInd = eInd + 1;
    eInd = sInd + size(MF,2) - 1;
    Z(:,sInd:eInd) = repmat(clo,1,size(MF,2)).*MF;
    
    flc = flc + 1;
    feat_listing(flc) = feat_ind;
end
Z = Z(:,1:eInd);
feat_listing = feat_listing(1:flc);

% Now let's build a logistic regression classifier
% Convert back to {0,1} from {-1,+1}
y = 0.5*(y+1);

etas_base = zeros(n,1);
%pluck_inds = 1:size(MF,2):eInd;
%etas_base = sum(Z(:,pluck_inds),2);
%Z(:,pluck_inds) = [];
w = zeros(size(Z,2),1);
lambda = 1e-6;
while 1
    wprev = w;
    etas = Z*w + etas_base;
    mus = 1./(1+exp(-etas));
    R = mus.*(1-mus);
    
    grad = -Z'*(y - mus);
    H = Z'*bsxfun(@times,R,Z);
    w = (H + lambda*speye(size(H,1))) \ (H*w - grad);
    if max(abs(w-wprev)) < 1e-5
        break;
    end
end
% Now recompute the weights taking V into account

w = transpose(reshape(w,size(MF,2),numel(feat_listing)));
%w = [ones(numel(feat_listing),1),transpose(reshape(w,size(MF,2)-1,numel(feat_listing)))];
wr = zeros(p,size(MF,2));
for j=1:size(w,1)
    wr(feat_listing(j),:) = w(j,:);
end
wr = [wr(:,1),transpose(V*wr(:,2:end)')];

% It's useful to compute an average stacking weight for each feature
stacking.mf_coeffs = cell(p,1);
stacking.expected_stacking_weights = zeros(p,1);
for j=1:p
    %stacking.mf_coeffs{j} = [sum(wr(j,:)),-wr(j,2:end)];
    stacking.mf_coeffs{j} = wr(j,:);
    locs = find(M(:,j));
    if ~numel(locs)
        continue;
    end
    mx = mean(M(locs,:))';
    stacking.expected_stacking_weights(j) = stacking.mf_coeffs{j}(1) + stacking.mf_coeffs{j}(2:end)*mx;
end

end
