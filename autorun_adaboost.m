% This script is a wrapper that sets up variables before running get_classifier

clear all
ism_tt_dir = 'train_test_ism6_nofill';
stm_tt_dir = 'train_test_stm6_nofill';
banner_tt_dir = 'train_test_banner_nofill';
results_dir = 'results_boosted_hii8';

% ism_tt_dir = 'train_test_ism7_nofill_only_level';
% stm_tt_dir = 'train_test_stm7_nofill_only_level';
% results_dir = 'results_boosted_hii7_only_level';

% mode_all = {'ism_only', 'stm_only' 'train_ism_test_stm', 'train_across_test_ism', ...
%             'train_across_test_stm', 'train_across_test_across'};
% mode_all = {'train_across_test_ism', 'train_across_test_stm', 'train_across_test_across'};
% mode_all = {'train_across_test_ism'};
% mode_all = {'train_ism_test_stm'};
% mode_all = {'ism_only', 'stm_only'};
% mode_all = {'train_stm_test_ism'};
% mode_all = {'stm_only'};
mode_all = {'banner_only'};

% 18 features
% manual_fts = {'age', 'nsbp_last', 'ndbp_last', 'hr_last', 'spo2_last', ...
%               'hemoglobin_last', 'temperature_last', 'wbc_last', 'platelet_last', ...
%               'albumin_last', 'urine_last', 'potassium_last', 'glucose_last', ...
%               'creatinine_last', 'lactic_acid_last', 'osi_last', 'si_last', 'oi_last'};

manual_fts = {'age', 'nsbp_last', 'ndbp_last', 'hr_last', 'spo2_last', ...
              'temperature_last', ...
              'urine_last', 'potassium_last', 'glucose_last', ...
              'creatinine_last', 'lactic_acid_last', 'si_last'};
age_group = [1/12 ,1/2 ; 1/2, 1 ; 1, 4 ; 4, 7 ; 7, 9 ; 9, 21];

% 14 features
% manual_fts = {'age', 'nsbp_last', 'ndbp_last', 'hr_last', 'spo2_last', ...
%               'hemoglobin_last', 'temperature_last', ...
%               'albumin_last', 'urine_last', 'potassium_last', 'glucose_last', ...
%               'creatinine_last', 'si_last', 'oi_last'};


only_sim = false;
only_top = false;
only_last = true;
train_on_time0 = true;
only_manual = true;
project = false;
train_by_age = false;
test_by_age = false;
get_classifier

% clearvars -except ism_tt_dir stm_tt_dir results_dir mode_all

% only_sim = false;
% only_top = false;
% only_last = true;
% train_on_time0 = true;
% only_manual = true;
% get_classifier

