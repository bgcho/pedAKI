function [clf,feat_labels] = boostedHII_reorderFeatures(clf,feat_ordering,feat_labels)

feat_labels = feat_labels(feat_ordering);

% Make sure feat ordering is row vector
if size(feat_ordering,1) > 1
    feat_ordering = feat_ordering';
end

p = numel(clf.weak_learners);
feat_ordering = [feat_ordering,p];

% First, re-order the weak learners
clf.weak_learners = clf.weak_learners(feat_ordering);

% Now re-order the stacking coefficients
sc_ord = [1,(feat_ordering+1)];
for j=1:p
    clf.stacking.mf_coeffs{j} = clf.stacking.mf_coeffs{j}(sc_ord);
end

clf.stacking.mf_coeffs = clf.stacking.mf_coeffs(feat_ordering);
clf.stacking.expected_stacking_weights = clf.stacking.expected_stacking_weights(feat_ordering);


