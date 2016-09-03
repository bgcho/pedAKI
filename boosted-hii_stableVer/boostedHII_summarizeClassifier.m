function boostedHII_summarizeClassifier(clf,feat_labels)

feat_labels{end+1} = 'Bias';

p = numel(clf.weak_learners);

first_round = -1*ones(p,1);
for j=1:p
    if ~numel(clf.weak_learners{j}); continue; end;
    first_round(j) = clf.weak_learners{j}{1}.boosting_round;
end

[~,feature_order] = sort(first_round,'ascend');
not_selected_buffer = '';
fprintf(1,'Feature order:\n');
feat_number = 0;
for j=1:p
    if first_round(feature_order(j)) == -1
        not_selected_buffer = [not_selected_buffer,'\n',num2str(j)];
        continue
    end
    fprintf(1,'%d) %s\n',j,feat_labels{feature_order(j)});
end


end

