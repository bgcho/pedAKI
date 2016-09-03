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

if isfield(clf.weak_learners{feat_ind}{1},'theta')
    locs = find(~isnan(X(:,feat_ind)));
    xf_lin = linspace(min(X(locs,feat_ind)),max(X(locs,feat_ind)),50);
    xage_lin = linspace(min(X(locs,67)),max(X(locs,67)));
    [xf,xage] = meshgrid(xf_lin,...
                         xage_lin);
    X2 = zeros(numel(xf),67);
    X2(:,feat_ind) = xf(:);
    X2(:,67) = xage(:);
    y = boostedHII_evaluateWeakLearner(X2,clf.weak_learners,feat_ind);
    % The line below reshapes so that the x-axis is age
    y = transpose(reshape(y,size(xf,1),size(xf,2)));
    figure;
    %subplot(1,2,1);
    imagesc(xage_lin,xf_lin,y);colorbar;
    set(gca,'YDir','normal');
    xlabel('Age');
    ylabel(feature_label);
    
    %subplot(1,2,2);
    
    %hold on;
    %scatter(X(locs,67),X(locs,feat_ind),50,'r');
else
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
y = y*clf.stacking.expected_stacking_weights(feat_ind);
          
line(x,y,'Parent',haxes2,'Color','r','LineWidth',3);
    
xlim(haxes2,xlim(haxes1));
%ylim(haxes2,[miny,maxy]);
title(feature_label);
xlabel(haxes1,feature_label);
set(gcf,'color','w');
end

end