% This script plots the AUC curve as a function of prediction window

clear all; clc


% fdir_boost = fullfile(pwd, 'results_boosted_hii5');
% % fname_boost{1} = [fdir_boost '\boosted_hii_ism_only_last_train0.mat'];
% fname_boost{1} = [fdir_boost '\boosted_hii_ism_only_last_manual018_train0.mat'];
% % fname_boost{end+1} = [fdir_boost '\boosted_hii_stm_only_last_manual018_train0.mat'];
% % fname_boost{end+1} = [fdir_boost '\boosted_hii_ism_only_last_manual014_train0.mat'];
% fname_boost{end+1} = [fdir_boost '\boosted_hii_train_ism_test_stm_last_manual018_train0.mat'];
% % fname_boost{end+1} = [fdir_boost '\boosted_hii_train_ism_test_stm_last_manual018_train0_prj_bal.mat'];
% fname_boost{end+1} = [fdir_boost '\boosted_hii_train_across_test_ism_last_manual018_train0_bal.mat'];
% fname_boost{end+1} = [fdir_boost '\boosted_hii_train_across_test_stm_last_manual018_train0_bal.mat'];
% fname_boost{end+1} = [fdir_boost '\boosted_hii_train_across_test_across_last_manual018_train0_bal.mat'];
% % fname_boost{end+1} = [fdir_boost '\boosted_hii_train_ism_test_stm_last_manual014_train0.mat'];
% % fname_boost{end+1} = [fdir_boost '\boosted_hii_train_across_test_ism_last_manual014_train0.mat'];
% % fname_boost{end+1} = [fdir_boost '\boosted_hii_train_across_test_stm_last_manual014_train0.mat'];
% % fname_boost{end+1} = [fdir_boost '\boosted_hii_train_across_test_across_last_manual014_train0.mat'];

% fdir_boost = fullfile(pwd, 'results_boosted_hii7_only_level');
% % fname_boost{1} = [fdir_boost '\boosted_hii_ism_only_last_train0.mat'];
% fname_boost{1} = [fdir_boost '\boosted_hii_ism_only_last_manual018_train0.mat'];
% fname_boost{end+1} = [fdir_boost '\boosted_hii_stm_only_last_manual018_train0.mat'];
% fname_boost{end+1} = [fdir_boost '\boosted_hii_train_ism_test_stm_last_manual018_train0.mat'];
% % fname_boost{end+1} = [fdir_boost '\boosted_hii_train_stm_test_ism_last_manual018_train0.mat'];
% fname_boost{end+1} = [fdir_boost '\boosted_hii_train_across_test_ism_last_manual018_train0.mat'];
% fname_boost{end+1} = [fdir_boost '\boosted_hii_train_across_test_stm_last_manual018_train0.mat'];
% fname_boost{end+1} = [fdir_boost '\boosted_hii_train_across_test_across_last_manual018_train0.mat'];


fdir_boost = fullfile(pwd, 'results_boosted_hii8');
% fname_boost{1} = [fdir_boost '\boosted_hii_ism_only_last_train0.mat'];
fname_boost{1} = [fdir_boost '\boosted_hii_ism_only_last_manual018_train0.mat'];
fname_boost{end+1} = [fdir_boost '\boosted_hii_stm_only_last_manual018_train0.mat'];
fname_boost{end+1} = [fdir_boost '\boosted_hii_banner_only_last_manual012_train0.mat'];
% % fname_boost{end+1} = [fdir_boost '\boosted_hii_ism_only_last_manual014_train0.mat'];
fname_boost{end+1} = [fdir_boost '\boosted_hii_train_ism_test_stm_last_manual018_train0.mat'];
% % fname_boost{end+1} = [fdir_boost '\boosted_hii_train_ism_test_stm_last_manual018_train0_prj_bal.mat'];
fname_boost{end+1} = [fdir_boost '\boosted_hii_train_across_test_ism_last_manual018_train0.mat'];
fname_boost{end+1} = [fdir_boost '\boosted_hii_train_across_test_stm_last_manual018_train0.mat'];
fname_boost{end+1} = [fdir_boost '\boosted_hii_train_across_test_across_last_manual018_train0.mat'];
% % fname_boost{end+1} = [fdir_boost '\boosted_hii_train_ism_test_stm_last_manual014_train0.mat'];
% % fname_boost{end+1} = [fdir_boost '\boosted_hii_train_across_test_ism_last_manual014_train0.mat'];
% % fname_boost{end+1} = [fdir_boost '\boosted_hii_train_across_test_stm_last_manual014_train0.mat'];
% % fname_boost{end+1} = [fdir_boost '\boosted_hii_train_across_test_across_last_manual014_train0.mat'];



colors = distinguishable_colors(25);
% group = [0,3,7, length(fname_boost)];
% group = [0,3,length(fname_boost)];
group = [0,3,length(fname_boost)];
linstyle = {'-', '--', '-.'};
for ii = 1:length(fname_boost)
	idx = find(ii>group, 1, 'last');
	l_style{ii} = linstyle{idx};
end

h006 = figure; hold on
h012 = figure; hold on


for ii = 1:length(fname_boost)
	load(fname_boost{ii})
	disp(fname_boost{ii})
	count006 = 1;
	count012 = 1;
	for jj = 1:length(model)
		if model(jj).time_window==6
			lag_006(count006) = model(jj).time_lag;
			auc_006(count006) = model(jj).auc;
			count006 = count006+1;
		else
			lag_012(count012) = model(jj).time_lag;
			auc_012(count012) = model(jj).auc;
			count012 = count012+1;
		end
	end

	lag_auc_006 = [double(lag_006) ; auc_006].';
	lag_auc_006 = sortrows(lag_auc_006, 1);
	lag_auc_012 = [double(lag_012) ; auc_012].';
	lag_auc_012 = sortrows(lag_auc_012, 1);
	% keyboard


	figure(h006)
	% subplot(122)
	% plot(-(lag_auc_006(:,1)- 6), lag_auc_006(:,2),'color',colors(ii,:), 'linewidth',3, 'linestyle', '-')	
	[ax1, h11, h12] = plotyy(-(lag_auc_006(:,1)-6), lag_auc_006(:,2), -(lag_auc_006(:,1)-6), lag_auc_006(:,2), 'plot');
	set(h11, 'color', colors(ii,:), 'linewidth', 3, 'linestyle', l_style{ii});
	set(h12, 'color', colors(ii,:), 'linewidth', 3, 'linestyle', l_style{ii});
	set(ax1(1), 'YColor', 'k', 'XLim', [-18 0], 'YLim', [0.4 1], 'YTick', [0.4:0.1:1], ...
		  'fontsize', 13, 'fontweight', 'bold', ...
		  'position', [0.1300    0.1500    0.775    0.70]);
	set(ax1(2), 'YColor', 'k', 'XLim', [-18 0], 'YLim', [0.4 1], 'YTick', [0.4:0.1:1], ...
		'fontsize', 13, 'fontweight', 'bold');
	figure(h012)
	% subplot(122)
	% plot(-(lag_auc_012(:,1)-12), lag_auc_012(:,2),'color',colors(ii,:), 'linewidth',3, 'linestyle', '-')
	[ax2, h21, h22] = plotyy(-(lag_auc_012(:,1)-12), lag_auc_012(:,2), -(lag_auc_012(:,1)-12), lag_auc_012(:,2), 'plot');
	set(h21, 'color', colors(ii,:), 'linewidth', 3, 'linestyle', l_style{ii});
	set(h22, 'color', colors(ii,:), 'linewidth', 3, 'linestyle', l_style{ii});
	set(ax2(1), 'YColor', 'k', 'XLim', [-12 0], 'YLim', [0.4 1], 'YTick', [0.4:0.1:1], ...
		  'fontsize', 13, 'fontweight', 'bold', ...
		  'position', [0.1300    0.1500    0.775    0.70]);
	set(ax2(2), 'YColor', 'k', 'XLim', [-12 0], 'YLim', [0.4 1], 'YTick', [0.4:0.1:1], ...
		  'fontsize', 13, 'fontweight', 'bold', ...
		  'position', [0.1300    0.1500    0.775    0.70]);
	
end




% lr_mode = {'lr: across-in' ; 'lr: ism' ; 'lr: ism/stm'; 'lr: across-out'};
% bh_mode = {'bh: across-in' ; 'bh: ism' ; 'bh: ism/stm'};

% ab_str{1} = 'Train CHLA Test CHLA: 23 features';
ab_str{1} = 'Train CHLA Test CHLA';
ab_str{end+1} = 'Train STM Test STM';
ab_str{end+1} = 'Train Banner Test Banner';
% ab_str{end+1} = 'Train/Test in CHLA: 16 features';
% ab_str{end+1} = 'Train/Test in CHLA: 15 features';
% ab_str{end+1} = 'Train CHLA Test CHLA: 14 features';
% ab_str{end+1} = 'Train/Test in CHLA: 13 features';
% ab_str{end+1} = 'Train/Test in CHLA: 12 features';
% ab_str{end+1} = 'Train/Test in CHLA: 11 features';
% ab_str{end+1} = 'Train/Test in CHLA: 10 features';
% ab_str{end+1} = 'Train/Test in CHLA: 9 features';
% ab_str{end+1} = 'Train/Test in CHLA: 8 features';
% ab_str{end+1} = 'Train/Test in CHLA: 7 features';
% ab_str{end+1} = 'Train/Test in CHLA: vital features';
ab_str{end+1} = 'Train CHLA Test STM';
% ab_str{end+1} = 'Train STM Test CHLA';
ab_str{end+1} = 'Train CHLA+STM Test CHLA';
ab_str{end+1} = 'Train CHLA+STM Test STM';
ab_str{end+1} = 'Train CHLA+STM Test CHLA+STM';
% ab_str{end+1} = 'Train in CHLA, Test in St.Mary: 16 features';
% ab_str{end+1} = 'Train in CHLA, Test in St.Mary: 15 features';
% ab_str{end+1} = 'Train CHLA Test STM: 14 features';
% ab_str{end+1} = 'Train CHLA+STM Test CHLA: 14 features';
% ab_str{end+1} = 'Train CHLA+STM Test STM: 14 features';
% ab_str{end+1} = 'Train CHLA+STM Test CHLA+STM: 14 features';
% ab_str{end+1} = 'Train in CHLA, Test in St.Mary: 13 features';
% ab_str{end+1} = 'Train in CHLA, Test in St.Mary: 12 features';
% ab_str{end+1} = 'Train in CHLA, Test in St.Mary: 11 features';
% ab_str{end+1} = 'Train in CHLA, Test in St.Mary: 10 features';
% ab_str{end+1} = 'Train in CHLA, Test in St.Mary: 9 features';
% ab_str{end+1} = 'Train in CHLA, Test in St.Mary: 8 features';
% ab_str{end+1} = 'Train in CHLA, Test in St.Mary: 7 features';
% ab_str{end+1} = 'Train in CHLA, Test in St.Mary: vital features';


figure(h006)
lgh006 = legend(ab_str);
set(lgh006, 'fontsize',11)
set(gcf,'paperpositionmode', 'auto')
set(gca,'position')
% set(gcf, 'position', [8 301 1534 477])
xlabel('Time from AKI onset (hours)')
ylabel('AUC')
title_h006 = title({'AdaBoost: AUC vc Time lag' ; 'when the time window is 6 hours'});
set(title_h006, 'fontsize', 18, 'fontweight', 'bold');
grid on
ylim([0.4 1])

figure(h012)
lgh012 = legend(ab_str);
set(lgh012, 'fontsize', 11)
set(gcf, 'paperpositionmode', 'auto')
% set(gcf, 'position', [8 301 1534 477])
xlabel('Time from AKI onset (hours)')
ylabel('AUC')
title_h012 = title({'AdaBoost: AUC vc Time lag' ; 'when the time window is 12 hours'});
set(title_h012, 'fontsize', 18, 'fontweight', 'bold');
grid on