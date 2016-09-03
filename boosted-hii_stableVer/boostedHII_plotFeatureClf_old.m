function boostedHII_plotFeatureClf(X,clf,feat_ind,feature_label)

if ~exist('feature_label','var'); feature_label = []; end;

if ~numel(clf.weak_learners{feat_ind})
    fprintf(1,'This feature was never selected.\n');
    return;
end

if ~numel(feature_label); feature_label = 'Feature'; end;

fprintf(1,'This feature was selected at the following rounds:\n');
for k = 1:numel(clf.weak_learners{feat_ind})
    fprintf(1,'\tBoosting round #%d\n',clf.weak_learners{feat_ind}{k}.boosting_round);
end

[h,x] = hist(X(~isnan(X(:,feat_ind)),feat_ind),100);
x = x';
    
figure;
bar(x,h/sum(h));
hold on;
haxes1 = gca;
haxes1_pos = get(haxes1,'Position');
haxes2 = axes('Position',haxes1_pos,...
              'XAxisLocation','top',...
              'YAxisLocation','right',...
              'Color','none');

y = boostedHII_evaluateWeakLearner(x,clf.weak_learners,feat_ind);
%y = y*clf.stacking.expected_stacking_weights(feat_ind);
          
line(x,y,'Parent',haxes2,'Color','r','LineWidth',3);
    
xlim(haxes2,xlim(haxes1));
%ylim(haxes2,[miny,maxy]);
title(feature_label);
xlabel(haxes1,feature_label);
set(gcf,'color','w');
    
end