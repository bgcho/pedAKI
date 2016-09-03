function boostedHII_compareStacking

labelMapping = [];
missingValFn = 'NaN';
featsToUse = []; % Grab all 51 features

r=load('../../Desktop/HIRBA/data/ffm51_full.mat');
load('../../Desktop/HIRBA/results_boostedHII/boostedHII_results_not_partitioned_by_hospital.mat');

nfolds = cv.NumTestSets;
hrsbacks = [1,2,4,6];
hourly_results_nostack = cell(numel(hrsbacks),1);
hourly_results_stack = cell(numel(hrsbacks),1);
for j=1:numel(hrsbacks)
    fprintf(1,'Working on %d of %d\n',j,numel(hrsbacks));pause(1e-5);
    hrsback = hrsbacks(j);
    
    [X2,y2,r2_index] = generateClassificationDataset(r,hrsback,labelMapping,featsToUse,missingValFn);
    y_test_nostack = zeros(size(X2,1),1);
    y_test_stack = zeros(size(X2,1),1);
    for foldNum = 1:nfolds
        [~,test_inds] = intersect(r2_index,r_index(cv.test(foldNum)));
        y_test_stack(test_inds) = boostedHII_predict(X2(test_inds,:),cvres.clf{foldNum});
        clf_nostack = cvres.clf{foldNum};
        clf_nostack.stacking = [];
        y_test_nostack(test_inds) = boostedHII_predict(X2(test_inds,:),clf_nostack);
    end
    
    hourly_results_nostack{j} = prediction_results(y2,y_test_nostack);
    hourly_results_stack{j} = prediction_results(y2,y_test_stack);    
end

disp('hi');

auc_stack = zeros(numel(hrsbacks),1);
auc_nostack = zeros(numel(hrsbacks),1);
sensitivity_stack = zeros(numel(hrsbacks),1);
sensitivity_nostack = zeros(numel(hrsbacks),1);
specificity_stack = zeros(numel(hrsbacks),1);
specificity_nostack = zeros(numel(hrsbacks),1);
for j=1:numel(hrsbacks)
    auc_stack(j) = hourly_results_stack{j}.AUC;
    auc_nostack(j) = hourly_results_nostack{j}.AUC;
    
    sensitivity_stack(j) = hourly_results_stack{j}.sensitivity;
    sensitivity_nostack(j) = hourly_results_nostack{j}.sensitivity;
    
    specificity_stack(j) = hourly_results_stack{j}.specificity;
    specificity_nostack(j) = hourly_results_nostack{j}.specificity;
end

figure;
scatter(hrsbacks,auc_stack,100,'red','filled');
hold on;
scatter(hrsbacks,auc_nostack,100,'blue','filled');
legend('Stacking','No stacking','Location','Best');
set(gca,'XDir','reverse');
grid on;
xlabel('Hours before intervention');
ylabel('AUC');

figure;
colors = [1,0,0;0,1,0;0,0,1;0,0,0];
for j=1:4
    scatter([1-specificity_stack(j);1-specificity_nostack(j)],[sensitivity_stack(j);sensitivity_nostack(j)],100,[colors(j,:);colors(j,:)],'filled');
    hold on;
end
xlabel('1 - Specificity');
ylabel('Sensitivity');
legend('1 hour before','2 hours before','4 hours before','6 hours before','Location','Best');
%hold on;
%scatter(1-specificity_nostack,sensitivity_nostack,100,colors,'filled');

%figure;
%scatter(hrsbacks,sensitivity_stack,100,'red','filled');
%hold on;
%scatter(hrsbacks,sensitivity_nostack,100,'blue','filled');
%legend('Stacking','No stacking','Location','Best');
%set(gca,'XDir','reverse');
%grid on;
%xlabel('Hours before intervention');
%ylabel('Sensitivity');

disp('hi');

return

for j=1:numel(hrsbacks)
    figure;
    plot(hourly_results_stack{j}.roc_fpr,hourly_results_stack{j}.roc_tpr,'red','LineWidth',2);
    hold on;
    
    plot(hourly_results_nostack{j}.roc_fpr,hourly_results_nostack{j}.roc_tpr,'blue','LineWidth',2);
    
    xlim([0,0.04]);
    xlabel('1 - Specificity');
    ylabel('Sensitivity');

    grid on;

    legend('Stacking','No stacking','Location','Best');
    scatter(1-hourly_results_stack{j}.specificity,hourly_results_stack{j}.sensitivity,100,'red','filled');
    scatter(1-hourly_results_nostack{j}.specificity,hourly_results_nostack{j}.sensitivity,100,'blue','filled');

    title(sprintf('%d hours before intervention',hrsbacks(j)));
end



disp('hi');
