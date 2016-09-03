function out = boostedHII_example(X,clf)


p = numel(clf.weak_learners);
n = size(X,1);
if p == size(X,2) + 1
    X = [X,ones(n,1)];
end

% First, generate the univariate classifiers
f = zeros(p,n);
for j=1:numel(clf.weak_learners)
    f(j,:) = boostedHII_evaluateWeakLearner(X,clf.weak_learners,j);
end


% Now compute the mask
XM = ~isnan(X);

% Now compute the stacking coefficients
sc = zeros(p,1);
for j=1:p
    sc(j) = clf.stacking.mf_coeffs{j}(1) + XM'*clf.stacking.mf_coeffs{j}(2:end)';
end
    
disp('hi');

% Now, compute the mask and produce the stacking coefficients

