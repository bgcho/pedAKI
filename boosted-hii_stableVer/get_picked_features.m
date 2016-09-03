function [feat,index] = get_picked_features(cvres,cols)

cvres.clf_full.weak_learners = cvres.clf_full.weak_learners(1:end-1);
first_rounds=inf*ones(numel(cvres.clf_full.weak_learners),1);
for j=1:numel(cvres.clf_full.weak_learners)
  if ~numel(cvres.clf_full.weak_learners{j});
      continue;
  end;
  first_rounds(j)=cvres.clf_full.weak_learners{j}{1}.boosting_round;
end;
locs = find(isfinite(first_rounds));
[vals,ord]=sort(first_rounds(locs),'ascend');
index = locs(ord);
feat = cols(index);