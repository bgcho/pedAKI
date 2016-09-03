% This script plots histogram of age groups:
% Age groups of AKI@CHLA, Stable@SHLA, AKI@STM, and Stable@STM

age_group = [1/12 ,1/2 ; 1/2, 1 ; 1, 4 ; 4, 7 ; 7, 9 ; 9, 21];

% Load the mat files
load('train_test_ism6_nofill/ism_onset_tt_tlag006_twin006.mat')
predictors = cellstr(predictors);
age_idx_ism = find(strcmp(predictors, 'age'));
X_ism = [X_test ; X_train];
y_ism = [y_test y_train];
y_ism = y_ism(:);
age_ism = X_ism(:, age_idx_ism)/365;
age_ism_aki = age_ism(y_ism==1);
age_ism_con = age_ism(y_ism==0);
nb_total_ism = length(age_ism);

load('train_test_stm6_nofill/stm_onset_tt_tlag006_twin006.mat')
predictors = cellstr(predictors);
age_idx_stm = find(strcmp(predictors, 'age'));
X_stm = [X_test ; X_train];
y_stm = [y_test y_train];
y_stm = y_stm(:);
age_stm = X_stm(:, age_idx_stm)/365;
age_stm_aki = age_stm(y_stm==1);
age_stm_con = age_stm(y_stm==0);
nb_total_stm = length(age_stm);


age_ism_aki_grouped = [];
age_ism_con_grouped = [];
age_stm_aki_grouped = [];
age_stm_con_grouped = [];

for ii = 1:size(age_group,1)
	mask_ism_aki = age_ism_aki>=age_group(ii,1) & age_ism_aki<age_group(ii,2);
	mask_ism_con = age_ism_con>=age_group(ii,1) & age_ism_con<age_group(ii,2);
	mask_stm_aki = age_stm_aki>=age_group(ii,1) & age_stm_aki<age_group(ii,2);
	mask_stm_con = age_stm_con>=age_group(ii,1) & age_stm_con<age_group(ii,2);
	if isempty(age_ism_aki_grouped)
		age_ism_aki_grouped = sum(mask_ism_aki);
		age_ism_con_grouped = sum(mask_ism_con);
		age_stm_aki_grouped = sum(mask_stm_aki);
		age_stm_con_grouped = sum(mask_stm_con);
	else
		age_ism_aki_grouped(end+1) = sum(mask_ism_aki);
		age_ism_con_grouped(end+1) = sum(mask_ism_con);
		age_stm_aki_grouped(end+1) = sum(mask_stm_aki);
		age_stm_con_grouped(end+1) = sum(mask_stm_con);
	end
end

age_ism_aki_grouped = age_ism_aki_grouped/nb_total_ism;
age_ism_con_grouped = age_ism_con_grouped/nb_total_ism;
age_stm_aki_grouped = age_stm_aki_grouped/nb_total_stm;
age_stm_con_grouped = age_stm_con_grouped/nb_total_stm;

data = [age_ism_aki_grouped(:) age_ism_con_grouped(:) age_stm_aki_grouped(:) age_stm_con_grouped(:)];

% cmap = [copper(2) ; hot(2)];
xval = 1:size(age_group,1);
figure(1)
hBar = bar(xval,data);                     % Plot Data, Get Handle
% set(hBar(1), 'FaceColor', cmap(1,:))       % Colour First Bar Set
% set(hBar(2), 'FaceColor', cmap(2,:))       % Colour First Bar Set
% set(hBar(3), 'FaceColor', cmap(3,:))       % Colour First Bar Set
% set(hBar(4), 'FaceColor', cmap(4,:))       % Colour First Bar Set
% set(hBar(5), 'FaceColor', cmap(5,:))       % Colour First Bar Set
% set(hBar(5), 'FaceColor', cmap(5,:))       % Colour First Bar Set
% set(gca, 'YLim', [870 1080])               % Set Y-Axis Limits
hold on
keyboard

lgh = legend('CHLA AKI', 'CHLA Stable', 'STM AKI', 'STM Stable');
title('Age groups')
xtick_str = {'1 month - 6 months' ; '6 months - 1 yr' ; '1 yr - 4 yrs' ; '4 yrs - 7 yrs' ; '7 yrs - 9 yrs' ; '9 yrs - 21 yrs'};
set(gca,'xticklabel', xtick_str)

% age_group = [1/12 ,1/2 ; 1/2, 1 ; 1, 4 ; 4, 7 ; 7, 9 ; 9, 21];