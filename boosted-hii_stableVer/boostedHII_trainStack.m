function stacking = boostedHII_trainStack(X,y,weak_learners,stackingOpts)
    
if ~stackingOpts.use
    stacking = [];
    return;
end
frac = stackingOpts.energyFrac;
[n,p] = size(X);
if p == numel(weak_learners) - 1
    X = [X,ones(n,1)];
    p = p + 1;
end

feats_picked = zeros(p,1);
for j=1:numel(weak_learners)
    if numel(weak_learners{j})
        feats_picked(j) = 1;
    end
end
feats_picked = find(feats_picked);
feats_notpicked = setdiff(1:p,feats_picked);

M = double(isnan(X));
M(:,mean(M)<0.3) = 0;
if ~numel(find(M==1)) || frac == 0
    V = zeros(p,0);
    MF = ones(n,1);
else
    M(:,feats_notpicked) = 0;
%    if 1
%        V = speye(size(M,2));
%    else
    C = M'*M;
    [V,D] = eig(C);
    d = diag(D);
    [d,ds] = sort(d,'descend');
    d_frac = cumsum(d)/sum(d);
    ind_frac = find(d_frac >= frac,1,'first');
    if numel(ind_frac)
        if d_frac(ind_frac) < frac
            ind_frac = max(0,ind_frac-1);
        end
        d = d(1:ind_frac);
        V = V(:,ds(1:ind_frac));
    end
%    end
    % Now generate the meta-features
    MF = [ones(n,1),M*V];
end

% Now generate all of the new features
Z = zeros(n,numel(feats_picked)*size(MF,2));
feat_listing = zeros(1,numel(feats_picked));flc = 0;
eInd = 0;
for j=1:numel(feats_picked)
    feat_ind = feats_picked(j);
    
    clo = boostedHII_evaluateWeakLearner(X,weak_learners,feat_ind);
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
uy = sort(unique(y),'ascend');
if uy(1) == -1 && uy(2) == 1
    y = 0.5*(y+1);
elseif uy(1) == 0 && uy(2) == 1
    % Then we are OK
else
    error('Unknown label type');
end

etas_base = zeros(n,1);
pluck_inds = 1:size(MF,2):eInd;
etas_base = sum(Z(:,pluck_inds),2);
Z(:,pluck_inds) = [];

%Z(:,mean(Z~=0)<0.3) = 0;
z2orig = size(Z,2);
locs = find(sum(Z));
nonlocs = find(sum(Z)==0);
Z(:,nonlocs) = [];
w = zeros(size(Z,2),1);
lambda = 1;
while 1
    wprev = w;
    etas = etas_base + Z*w;
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

w2 = zeros(z2orig,1);
w2(locs) = w;
w = w2;
w = [ones(numel(feat_listing),1),transpose(reshape(w,size(MF,2)-1,numel(feat_listing)))];
%w = transpose(reshape(w,size(MF,2),numel(feat_listing)));
wr = zeros(p,size(MF,2));
for j=1:size(w,1)
    wr(feat_listing(j),:) = w(j,:);
end
wr = [wr(:,1),transpose(V*wr(:,2:end)')];

% It's useful to compute an average stacking weight for each feature
stacking.mf_coeffs = cell(p,1);
stacking.expected_stacking_weights = zeros(p,1);
for j=1:p
    %stacking.mf_coeffs{j} = wr(j,:);
    stacking.mf_coeffs{j} = [sum(wr(j,:)),-wr(j,2:end)];
    locs = find(~M(:,j));
    if ~numel(locs)
        continue;
    end
    mx = mean(~M(locs,:))';
    stacking.expected_stacking_weights(j) = stacking.mf_coeffs{j}(1) + stacking.mf_coeffs{j}(2:end)*mx;
end