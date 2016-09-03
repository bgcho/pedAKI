function boostedHII_missingDataPatterns(r)

[n,p] = size(r.X);
X = [r.X,ones(n,1)];
p = p + 1;

stacking_matrix = zeros(p+1,p);
for j=1:p
    stacking_matrix(:,j) = transpose(r.cvres.clf_full.stacking.mf_coeffs{j});
end

[patterns,~,row_inds] = unique(double(~isnan(X)),'rows');
noccurrences = zeros(size(patterns,1),1);
for j=1:numel(row_inds)
    noccurrences(row_inds(j)) = noccurrences(row_inds(j)) + 1;
end
[noccurrences,sort_ord] = sort(noccurrences,'descend');

patterns = patterns(sort_ord,:);

% Blank out coefficients if the feature is missing
% Do this by .* with patterns (which is {0,1}-valued)
pattern_coefficients = patterns.*([ones(size(patterns,1),1),patterns]*stacking_matrix);
disp('hi');

