function boostedHII_plotWeakLearner(X,clf,feat_ind)

[h,x1] = hist(X(~isnan(X(:,feat_ind)),feat_ind),1000);
c = clf.weak_learners{feat_ind};
x = linspace(min(X(:,feat_ind)),max(X(:,feat_ind)),10000)';

figure;
bar(x1,h/sum(h));
hold on;

haxes1 = gca;
haxes1_pos = get(haxes1,'Position');
haxes2 = axes('Position',haxes1_pos,...
              'XAxisLocation','bottom',...
              'YAxisLocation','left',...
              'Color','none');
xlim(haxes2,xlim(haxes1));
set(haxes1,'XTick',[]);
set(haxes1,'YTick',[]);
set(gcf,'color','white');
y = boostedHII_evaluateWeakLearner(x,clf.weak_learners,feat_ind);
line(x,y,'Parent',haxes2,'Color','r','LineWidth',3);
