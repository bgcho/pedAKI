function results = prediction_results(labels,preds,pos_class,decision_thresholds)

if ~exist('pos_class','var'); pos_class = 1; end;
if ~exist('decision_thresholds','var'); decision_thresholds = 0.5; end;

if numel(unique(labels)) ~= 2
    error('labels must be 2-class');
end

locs_pos = find(labels == pos_class);
locs_neg = find(labels ~= pos_class);

[roc_fpr,roc_tpr,~,AUC] = perfcurve(labels,preds,pos_class);

%preds = 1./(1+exp(-preds));

sensitivities = zeros(size(decision_thresholds));
specificities = zeros(size(decision_thresholds));
ppvs = zeros(size(decision_thresholds));
npvs = zeros(size(decision_thresholds));
for j=1:numel(decision_thresholds)
    sensitivities(j) = sum(preds(locs_pos) > decision_thresholds(j)) / numel(locs_pos);
    specificities(j) = sum(preds(locs_neg) < decision_thresholds(j)) / numel(locs_neg);
    ppvs(j) = sum(preds(locs_pos) > decision_thresholds(j)) / sum(preds > decision_thresholds(j));
    npvs(j) = sum(preds(locs_neg) < decision_thresholds(j)) / sum(preds < decision_thresholds(j));
end

results.AUC = AUC;
results.pAUC5 = trapz(roc_fpr(roc_fpr<=0.05),roc_tpr(roc_fpr<=0.05));
results.decision_thresholds = decision_thresholds;
results.sensitivities = sensitivities;
results.specificities = specificities;
results.ppvs = ppvs;
results.npvs = npvs;
results.roc_fpr = roc_fpr;
results.roc_tpr = roc_tpr;
