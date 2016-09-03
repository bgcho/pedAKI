"""
library for Early pediatric AKI data preparation
author: 310248864, Ben ByungGu Cho
last modified: 20160707
"""

import os
import numpy as np
import numpy.matlib
import pandas as pd
import pickle
import matplotlib
import matplotlib.pyplot as plt
import random

from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV

matplotlib.style.use('ggplot')
np.seterr(divide='ignore')

path2normscr = os.path.join(os.path.dirname("__file__"), "csv_files", "ped_normal_scr.csv")
norm_scr_lim = pd.DataFrame.from_csv(path2normscr)
age_yr_ll = np.array(norm_scr_lim.low_age)
age_yr_ul = np.array(norm_scr_lim.upp_age)

def filterEncPerPat(_df):
    """
    Function that maintains one encounter per patient with the longest los
    If the los is same for multiple encounters, maintain the earliest one
    and get rid of the later ones
    :param _df: Dataframe with multiple encounters per patient
    :return: dataframe with one encounter per patient
    """
    # initialize the return dataframe
    df = _df.copy(deep=True)

    to_drop_idx = np.array([])
    for patient in _df.patient_id.unique():
        pat_df = _df.loc[_df.patient_id == patient, :]
        if len(pat_df.encounter_id.unique()) > 1:
            # Leave the encounter that has longest los
            dtime_sec = (pat_df.outtime - pat_df.intime).astype('timedelta64[s]')
            if len(pat_df.encounter_id.unique()) == len(dtime_sec.unique()):
                to_drop_idx = np.concatenate((to_drop_idx,
                                              pat_df.index[dtime_sec < max(dtime_sec)]),
                                             axis=0)
            else:
                min_intime = pat_df.intime.unique().min()
                # max_outtime = pat_df.outtime.unique().max()
                to_drop_idx = np.concatenate((to_drop_idx,
                                              pat_df.index[pat_df.intime != min_intime]),
                                             axis=0)
                # to_drop_idx = np.concatenate((to_drop_idx,
                #                               pat_df.index[pat_df.outtime != max_outtime]),
                #                              axis=0)
    df.drop(to_drop_idx, inplace=True)
    return df

def exAKIAdm(_group, _aki_adm_hr):
    ### Optional exclusion
    # df_sub = _group.copy(deep=True)
    dtime_hr = (_group.loc[:, 'charttime'] \
                - _group.loc[_group.index[0], 'intime']).astype('timedelta64[m]') / 60
    # _group.loc[:, 'dtime_hr'] = dtime_hr
    admaki_mask = dtime_hr <= _aki_adm_hr

    if _group.loc[admaki_mask, 'AKI_stage'].sum() > 0:
        return False
    else:
        return True

def exAKILate(_group, _aki_late_hr):
    dtime_hr = (_group.loc[:, 'charttime'] \
                - _group.loc[_group.index[0], 'intime']).astype('timedelta64[m]') / 60
    eaki_mask = dtime_hr <= _aki_late_hr

    if _group.loc[eaki_mask, 'AKI_stage'].sum()<1:
        return False
    else:
        return True

def exAKINone(_group):
    # df_sub = _group.copy(deep=True)

    if _group.loc[:, 'AKI_stage'].sum() < 1:
        return False
    else:
        return True

def tagAKI(_group):
    # df_sub = _group.copy(deep=True)
    # print("group name: {}".format(_group.name))
    dtime_hr = (_group.loc[:, 'charttime'] \
        - _group.loc[_group.index[0], 'intime']).astype('timedelta64[m]') / 60

    # df_sub.loc[:, 'dtime_hr'] = dtime_hr

    col_nb = _group.valcol.unique()[0]
    for idx, row in _group.iterrows():
        scr_rate = np.NaN
        _group.loc[idx, 'scr_rate'] = scr_rate
        if row.dtime_hr>=48:
            time_mask = (dtime_hr >= row.dtime_hr - 48) & (dtime_hr <= row.dtime_hr)

            y = eval("np.array(_group.value{}[time_mask])".format(col_nb))
            x = np.array(_group.dtime_hr[time_mask])
            A = np.vstack([x, np.ones(len(x))]).T
            if A.shape[0]>1:
                slope, intercept = np.linalg.lstsq(A, y)[0]
                # stop
                scr_rate = slope * 48
                _group.loc[idx, 'scr_rate'] = scr_rate

        cur_val = eval("row.value{}".format(col_nb))

        # Calculate average rate of change of SCr within 48 hour time window
        # charttime +- 30h is used as the time window
        # time_mask = (dtime_hr > dtime_hr[idx] - 30) \
        #             & (dtime_hr <= dtime_hr[idx] + 30)
        # df_window = _group.loc[time_mask, :]
        # dtime_hr_window = dtime_hr[time_mask]
        # # delta_t = df_window.dtime_hr[df_window.index[-1]] - df_window.dtime_hr[df_window.index[0]]
        # delta_t = dtime_hr_window[dtime_hr_window.index[-1]] - dtime_hr_window[dtime_hr_window.index[0]]
        # value = np.empty(2)
        #
        # if df_window.valcol[df_window.index[0]] == 1:
        #     value[0] = df_window.value1[df_window.index[0]]
        # elif df_window.valcol[df_window.index[0]] == 2:
        #     value[0] = df_window.value2[df_window.index[0]]
        # if df_window.valcol[df_window.index[-1]] == 1:
        #     value[1] = df_window.value1[df_window.index[-1]]
        # elif df_window.valcol[df_window.index[-1]] == 2:
        #     value[1] = df_window.value2[df_window.index[-1]]
        # # stop
        # scr_rate = np.divide(value[-1] - value[0], delta_t) * 48
        # _group.loc[idx, 'scr_rate'] = scr_rate
        # # df_sub.loc[idx, 'scr_rate'] = scr_rate
        #
        # # Apply AKI criteria
        # if _group.valcol[idx] == 1:
        #     cur_val = _group.value1[idx]
        # elif _group.valcol[idx] == 2:
        #     cur_val = _group.value2[idx]

        if (cur_val >= 3.0 * _group.bs_scr[idx]) or (cur_val >= 4.0):
            AKI_stage = 3
        elif cur_val >= 2.0 * _group.bs_scr[idx]:
            AKI_stage = 2
        elif (cur_val >= 1.5 * _group.bs_scr[idx]) or (scr_rate >= 0.3):
            AKI_stage = 1
        else:
            AKI_stage = 0
        _group.loc[idx, 'AKI_stage'] = AKI_stage
        # df_sub.loc[idx, 'AKI_stage'] = AKI_stage
    return _group

# def tagBSCr(_group):
#
#     # df_sub = _group.copy(deep=True)
#     age_yr = np.array(_group.age / 365)
#     age_yr_mat, lage_yr_mat = np.meshgrid(age_yr, age_yr_ll, indexing='xy', sparse=False)
#     uage_yr_mat = np.matlib.repmat(age_yr_ul, len(age_yr), 1).transpose()
#
#     sex_pat = _group.sex.unique().astype(str)
#     sex_arr = norm_scr_lim.sex.as_matrix().astype(str)
#     sex_mat = np.matlib.repmat(sex_arr, len(age_yr), 1).transpose()
#
#     age_sex_mask = (age_yr_mat > lage_yr_mat) & (age_yr_mat <= uage_yr_mat) & \
#                    (np.char.find(sex_mat, sex_pat[0]) > -1)
#     # print(norm_scr_lim.upp_scr[age_sex_mask.nonzero()[0]])
#     # print('group name: {}'.format(_group.name))
#     _group['bs_scr'] = norm_scr_lim.upp_scr[age_sex_mask.nonzero()[0]].as_matrix()
#
#
#     return _group

def tagBSCr(_group):

    age_yr = _group.age.unique()[0] / 365.
    age_mask = (age_yr_ll < age_yr) & (age_yr_ul >= age_yr)

    sex = _group.sex.unique()[0]

    sex_arr = norm_scr_lim.sex.as_matrix().astype(str)
    sex_mask = [sex in normsex for normsex in sex_arr]
    _group['bs_scr'] = norm_scr_lim.upp_scr[age_mask & sex_mask].unique()[0]

    return _group


def tagDtimeHr(_group):
    dtime_hr = (_group.loc[:, 'charttime'] \
                - _group.loc[_group.index[0], 'intime']).astype('timedelta64[m]') / 60
    _group['dtime_hr'] = dtime_hr

    return _group


def cleanSCrDf2(_scrdataframe, ex_age=False, ex_los=False,
                ex_aki_adm=False, aki_adm_hr=12, ex_laki=False, aki_late_hr=72,
                ex_noaki=False, enc_per_pat=False):

    # Convert encounter_id, patient_id from float64 to int
    _scrdataframe['encounter_id'] = _scrdataframe['encounter_id'].astype('int')
    _scrdataframe['patient_id'] = _scrdataframe['patient_id'].astype('int')
    _scrdataframe['value2'] = _scrdataframe['value2'].astype('float64')

    if ex_age:
        ped_lim = np.array([1. / 12, 21.]) * 365
        _scrdataframe = _scrdataframe.loc[(_scrdataframe['age'] >= ped_lim[0]) &
                                          (_scrdataframe['age'] <= ped_lim[-1]), :]
    if ex_los:
        _scrdataframe = _scrdataframe.loc[_scrdataframe['los'] >= 24 * 60, :]

    ### Caculate bs_scr, scr_rate, AKI stage
    _scrdataframe = _scrdataframe.sort_values(by=['encounter_id', 'charttime'],
                                              ascending=[1, 1])

    print('tagging time from admission')
    if 'dtime_hr' not in _scrdataframe.columns:
        _scrdataframe = _scrdataframe.groupby('encounter_id').apply(tagDtimeHr)

    print('tagging baseline SCr level ...')
    if 'bs_scr' not in _scrdataframe.columns:
        _scrdataframe = _scrdataframe.groupby('encounter_id').apply(tagBSCr)
    print('tagging AKI stage ...')
    if 'AKI_stage' not in _scrdataframe.columns:
        _scrdataframe = _scrdataframe.groupby('encounter_id').apply(tagAKI)
    print('tagging reference time ...')
    if 'reftime' not in _scrdataframe.columns:
        _scrdataframe = _scrdataframe.groupby('encounter_id').apply(fillReftime2, aki_adm_hr)


    if ex_aki_adm:
        print('Excluding patients who had AKI at admission (within {} hours from admission) ... '.format(int(aki_adm_hr)))
        _scrdataframe = _scrdataframe.groupby('encounter_id').\
            filter(lambda _group: _group.loc[_group.dtime_hr<=aki_adm_hr, 'AKI_stage'].sum()==0)
    if ex_laki:
        print('Excluding patients who developed AKI after {} hours ... '.format(int(aki_late_hr)))
        _scrdataframe = _scrdataframe.groupby('encounter_id').\
            filter(lambda _group: _group.loc[_group.dtime_hr<=aki_late_hr, 'AKI_stage'].sum()>0)
    if ex_noaki:
        print('Excluding patients who never developed AKI ... ')
        _scrdataframe = _scrdataframe.groupby('encounter_id').\
            filter(lambda _group: _group.loc[:, 'AKI_stage'].sum()>0)
    if enc_per_pat:
        print('Assigning single encounter per patient ...')
        _scrdataframe = filterEncPerPat(_scrdataframe)

    # _scrdataframe = _scrdataframe.drop('dtime_hr', axis=1)

    return _scrdataframe


def cleanSCrDf(_scrdataframe, ex_age=False, ex_los=False,
               ex_aki_adm=False, aki_adm_hr=12, ex_laki=False, aki_late_hr=72,
               ex_noaki=False, enc_per_pat=False):
    """
    Clean up scr dataframe, queried from the ism database.
    :param _scrdataframe: SCr Dataframe to be cleaned. _scrdataframe is a returned dataframe
      from ism_utilities_Ben.queryISM::getItemData()
    :param _path2normscr: String. Path to ped_normal_scr.csv file
      (Currently saved at C:\Users\310248864\pedAKI\src\pedAKI_predictor\csv_files)
    :param ex_age: (Optional) Boolean. Flag for age filtering. Include encounters 1month<=age<=21year
    :param ex_los: (Optional) Boolean. Flag for length of stay filtering. Include encounters los>=24hours
    :param ex_aki_adm: (Optional) Boolean. Flag to filter out AKI at admission.
      Exclude encounters with AKI_stage>0 within 12hours(default) from admission
    :param aki_adm_hr: (Optional) hours from admission that defines AKI at admission.
      Used for excluding AKI patients at admission
    :param ex_laki: (Optional) Boolean. Flag to filter out AKI lately.
      Exclude encounters with AKI_stage>0 after 72hours from admission
    :param ex_noaki: (Optional) Boolean. Flag to filter out no AKI encounters.
      Exclude encounters with no AKI_stage during the stay
    :param enc_per_pat: (Optional) Boolean. Flag to leave only one encounter per patient with the longest los.
    :return: Dataframe. Filtered dataframe with various options listed above.

    Note:
    Added columns:
    1) age: age in days
    2) bs_scr: baseline Scr (mg/dl)
    3) scr_rate: rate of change of SCr (mg/dl/48hr)
    4) AKI_stage: AKI stage according to KDIGO criteria
    The exclusion criteria is:
    1) Exclude encounters without charttime OR dob OR sex
    2) (Optional) Only include encounters with 1 mnth <= age <= 21 yr
    3) (Optional) Only include encounters with los >= 24 h
    4) (Optional) Exclude encounters that already have AKI at admission
       (within 12 hours (default) from admission)
    5) (Optional) Only include encounters that have Early AKI
    6) (Optional) Exclude encounters that never had AKI
    """

    # Convert encounter_id, patient_id from float64 to int
    _scrdataframe['encounter_id'] = _scrdataframe['encounter_id'].astype('int')
    _scrdataframe['patient_id'] = _scrdataframe['patient_id'].astype('int')
    _scrdataframe['value2'] = _scrdataframe['value2'].astype('float64')

    # ### Exclude patients with no 'charttime'
    # _scrdataframe = _scrdataframe.dropna(subset=['charttime'])
    #
    # ### Fill in missing 'dob' and 'sex' if there are available record with same 'patient_id'
    # ### If not, exclude patients with no 'dob' OR 'sex'
    # missing_key = ['sex', 'dob']
    # for key in missing_key:
    #     nullkey_patient = _scrdataframe.loc[pd.isnull(_scrdataframe[key]), key]. \
    #         groupby(_scrdataframe.patient_id).unique().index
    #
    #     for patient in nullkey_patient:
    #         tmp_key = _scrdataframe.loc[_scrdataframe.patient_id == patient, key].unique()
    #         real_key = tmp_key[~pd.isnull(tmp_key)]
    #
    #         if real_key:
    #             try:
    #                 _scrdataframe.loc[_scrdataframe.patient_id == patient, key] = real_key
    #             except:
    #                 _scrdataframe = _scrdataframe.loc[_scrdataframe.patient_id != patient, :]
    #         else:
    #             _scrdataframe = _scrdataframe.loc[_scrdataframe.patient_id != patient, :]

    # ### Add age in days
    # _scrdataframe['age'] = (_scrdataframe['intime'] - _scrdataframe['dob']).\
    #     astype('timedelta64[h]') / 24
    # # _scrdataframe['age'] = (_scrdataframe['intime'] - _scrdataframe['dob']).days
    # _scrdataframe['age'] = _scrdataframe['age'].astype('int')
    ### Only include patients with 1 month <= age <= 21 yr
    if ex_age:
        ped_lim = np.array([1./12, 21.]) * 365
        _scrdataframe = _scrdataframe.loc[(_scrdataframe['age'] >= ped_lim[0]) &
                                          (_scrdataframe['age'] <= ped_lim[-1]), :]
    if ex_los:
        _scrdataframe = _scrdataframe.loc[_scrdataframe['los'] >= 24*60, :]

    ### Caculate bs_scr, scr_rate, AKI stage
    _scrdataframe = _scrdataframe.sort_values(by=['encounter_id', 'charttime'],
                                              ascending=[1, 1])
    encounter_group = _scrdataframe.groupby('encounter_id')['patient_id'].count()
    encounter_cnt = encounter_group.values
    encounter_idx = encounter_group.index
    cum_cnt = np.cumsum(encounter_cnt)

    # Load the normal Scr table for age/sex
    # ped_norm_scr = pd.DataFrame.from_csv(_path2normscr)

    prev_count = 0
    # lage_yr = np.array(ped_norm_scr.low_age)
    # uage_yr = np.array(ped_norm_scr.upp_age)
    to_drop_indx = np.array([])


    for encounter, curr_count in zip(encounter_idx, cum_cnt):

        df_sub = _scrdataframe.loc[_scrdataframe.index[prev_count:curr_count],
                 ['encounter_id', 'charttime', 'intime', 'sex', 'age',
                  'valcol', 'value1', 'value2']]

        dtime_hr = (df_sub.loc[:, 'charttime']
                    - df_sub.loc[df_sub.index[0], 'intime']).astype('timedelta64[m]')/60
        df_sub.loc[:, 'dtime_hr'] = dtime_hr

        # Cacluate baseline SCr
        # Get the index in the ped_norm_scr table for each row in unique encounter
        age_yr = np.array(df_sub.age / 365)
        age_yr_mat, lage_yr_mat = np.meshgrid(age_yr, age_yr_ll, indexing='xy', sparse=False)
        uage_yr_mat = np.matlib.repmat(age_yr_ul, len(age_yr), 1).transpose()

        sex_pat = df_sub.sex.unique().astype(str)
        sex_arr = norm_scr_lim.sex.as_matrix().astype(str)
        sex_mat = np.matlib.repmat(sex_arr, len(age_yr), 1).transpose()

        age_sex_mask = (age_yr_mat > lage_yr_mat) & (age_yr_mat <= uage_yr_mat) & (
        np.char.find(sex_mat, sex_pat[0]) > -1)

        df_sub.loc[:, 'norm_scr_idx'] = age_sex_mask.nonzero()[0]

        for idx, row in df_sub.iterrows():
            bs_scr = norm_scr_lim.upp_scr[row.norm_scr_idx]
            _scrdataframe.loc[idx, 'bs_scr'] = bs_scr
            df_sub.loc[idx, 'bs_scr'] = bs_scr

            # Calculate average rate of change of SCr within 48 hour time window
            # charttime +- 30h is used as the time window
            time_mask = (df_sub.dtime_hr > df_sub.dtime_hr[idx] - 30) \
                        & (df_sub.dtime_hr <= df_sub.dtime_hr[idx] + 30)
            df_window = df_sub.loc[time_mask, :]
            delta_t = df_window.dtime_hr[df_window.index[-1]] - df_window.dtime_hr[df_window.index[0]]
            value = np.empty(2)
            if df_window.valcol[df_window.index[0]] == 1:
                value[0] = df_window.value1[df_window.index[0]]
            elif df_window.valcol[df_window.index[0]] == 2:
                value[0] = df_window.value2[df_window.index[0]]
            if df_window.valcol[df_window.index[-1]] == 1:
                value[1] = df_window.value1[df_window.index[-1]]
            elif df_window.valcol[df_window.index[-1]] == 2:
                value[1] = df_window.value2[df_window.index[-1]]
            # stop
            scr_rate = np.divide(value[-1] - value[0], delta_t) * 48
            _scrdataframe.loc[idx, 'scr_rate'] = scr_rate
            df_sub.loc[idx, 'scr_rate'] = scr_rate

            # Apply AKI criteria
            if df_sub.valcol[idx] == 1:
                cur_val = df_sub.value1[idx]
            elif df_sub.valcol[idx] == 2:
                cur_val = df_sub.value2[idx]
            
            if (cur_val >= 3.0 * df_sub.bs_scr[idx]) or (scr_rate >= 4.0):
                AKI_stage = 3
            elif cur_val >= 2.0 * df_sub.bs_scr[idx]:
                AKI_stage = 2
            elif (cur_val >= 1.5 * df_sub.bs_scr[idx]) or (scr_rate >= 0.3):
                AKI_stage = 1
            else:
                AKI_stage = 0
            _scrdataframe.loc[idx, 'AKI_stage'] = AKI_stage
            df_sub.loc[idx, 'AKI_stage'] = AKI_stage


        ### Optional exclusion
        admaki_mask = df_sub.dtime_hr <= aki_adm_hr

        if ex_aki_adm and (df_sub.loc[admaki_mask, 'AKI_stage'].sum()>0):
            to_drop_indx = np.concatenate((to_drop_indx,
                                           _scrdataframe.index[prev_count:curr_count]),
                                          axis=0)

            print('encounter_id = {} removed for having AKI at admission'.format(encounter))

        eaki_mask = df_sub.dtime_hr <= aki_late_hr

        if ex_laki and (df_sub.loc[eaki_mask, 'AKI_stage'].sum()<1):
            to_drop_indx = np.concatenate((to_drop_indx,
                                            _scrdataframe.index[prev_count:curr_count]),
                                           axis=0)
            print('encounter_id = {} removed for not having Early AKI'.format(encounter))

        if ex_noaki and (df_sub.loc[:, 'AKI_stage'].sum() < 1):
            to_drop_indx = np.concatenate((to_drop_indx,
                                            _scrdataframe.index[prev_count:curr_count]),
                                           axis=0)
            print('encounter_id = {} removed for not having AKI'.format(encounter))


        prev_count = curr_count

    to_drop_indx = to_drop_indx.astype('int')
    to_drop_indx = np.unique(to_drop_indx)
    _scrdataframe.drop(to_drop_indx, inplace=True)

    if enc_per_pat:
        _scrdataframe = filterEncPerPat(_scrdataframe)


    return _scrdataframe


def randomDate(_start, _end, _prop):
    rtime = _start + _prop * (_end - _start)
    return rtime


def fillReftime(_group, ref_mode="onset", lag=None, scr_df=None):
# def fillReftime(group, _ref_mode, _lag, _scr_df):
    """
    Function that calculates the reference time for each encounter_id
    :param group: pandas groupby object groupbed by encounter_id
    :param ref_mode: (Optional) Reference time mode. Should be either "onset" OR "random"
    In "onset" mode, the reference time is the earliest time when AKI is detected.
    In "random" mode, the reference time is randomly chosen between intime+abs(lag)
    and outtime
    :param lag: (Optional) Only used for ref_mode="random"
    :param scr_df: (Optional) Only used for ref_mode="onset"
    :return:
    """
    if ref_mode == "onset":
        tmp_df = scr_df.loc[scr_df.encounter_id==_group.encounter_id.unique()[0],
                            ['charttime', 'AKI_stage']]
        _group['reftime'] = tmp_df.charttime[tmp_df.AKI_stage>0].min()
        # group['reftime'] = group.charttime[group.AKI_stage > 0].min()
    elif ref_mode == "random":
        stime = _group.intime.unique()[0] + np.timedelta64(int(abs(lag)*60), 'm')
        etime = _group.outtime.unique()[0]
        rand_reftime = randomDate(stime, etime, random.random())
        _group['reftime'] = rand_reftime
    return _group

def fillReftime2(_group, lag=None):
    """
    Function that calculates the reference time for each encounter_id
    :param group: pandas groupby object groupbed by encounter_id
    :param ref_mode: (Optional) Reference time mode. Should be either "onset" OR "random"
    In "onset" mode, the reference time is the earliest time when AKI is detected.
    In "random" mode, the reference time is randomly chosen between intime+abs(lag)
    and outtime
    :param lag: (Optional) Only used for ref_mode="random"
    :param scr_df: (Optional) Only used for ref_mode="onset"
    :return:
    """
    if _group.AKI_stage.sum()>0:
        _group['reftime'] = _group.charttime[_group.AKI_stage>0].min()
        # group['reftime'] = group.charttime[group.AKI_stage > 0].min()
    else:
        stime = _group.intime.unique()[0] + np.timedelta64(int(abs(lag)*60), 'm')
        etime = _group.outtime.unique()[0]
        rand_reftime = randomDate(stime, etime, random.random())
        _group['reftime'] = rand_reftime
    return _group





def getLastVal(_series):
    """
    Function that returns the last non-null value of a series
    :param _series: series data
    :return: last non-null value of the series
    """
    tmp_series = _series[~pd.isnull(_series)]
    try:
        return tmp_series.values[-1]
    except:
        return np.float64(np.nan)


def getBandwidth(_x):
    """
    Function that calculates the best bandwidth for gaussian kernel density estimation
    of pdf of given data
    :param _x: data
    :return: best gaussian kernel bandwidth
    """
    x = _x[np.isfinite(_x)]
    std_x = np.std(x)
    med_x = np.median(x)
    x_fit = x[abs(x - med_x) < 3 * std_x]
    grid = GridSearchCV(KernelDensity(),
                        {'bandwidth': std_x*np.logspace(-1,0,10)},
                        cv=5)
    grid.fit(x_fit[:,None])
    print("best bandwidth: {}".format(grid.best_estimator_.bandwidth/std_x))
    return grid.best_estimator_.bandwidth


def kde_sklearn(_x, _x_grid, bandwidth=0.1):
    """
    Kernel Density Estimation with Scikit-learn
    :param _x: data
    :param _x_grid: pdf grid
    :param bandwidth: bandwidth of the gaussian kernel
    :return: pdf
    """
    x = _x[np.isfinite(_x)]
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(x[:,None])
    pdf = np.exp(kde.score_samples(_x_grid[:, None]))
    return pdf



def getDistIOmat(_io_mat, _reftype, age_lim=None, sex=None, dbstring='ism',
                 xlim=None, ylim=None, dump_fig=False):
    """
    Function that gets the distribution for each feature from the I/O matrix
    :param _io_mat: predictor/response variable matrix
    :param age_lim: list of tuples that defines the upper/lower age limit [(llim, ulim), ...]
    :param sex: (ToDo) distribution for different sex
    :param xlim: x-axis limit for each feature, list of tuples
    :param ylim: y-axis limit for each feature, list of tuples
    :param dump_fig: boolean, if True, save the figures
    :return:
    """
    path2chartitem_csv = os.path.join(os.path.dirname("__file__"), "csv_files", "chartitem.csv")
    chartitemdf = pd.DataFrame.from_csv(path2chartitem_csv)
    col_all = _io_mat.columns

    col_all = col_all.drop(['patient_id','encounter_id','age','sex','AKI_stage'])
    if xlim is None:
        xlim = [(None, None)]*len(col_all)
    if ylim is None:
        ylim = [(None, None)] * len(col_all)

    for col, val_lim, pdf_lim in zip(col_all, xlim, ylim):
        # stop
        itemsdf = _io_mat.loc[:,['patient_id', 'encounter_id',
                                'age', 'sex', 'AKI_stage', col]]
        itemsdf = itemsdf.loc[~pd.isnull(itemsdf[col]),:]
        pattern = col[:col.find("_")]
        # p_mask = [label.find(pattern)>-1 for label, uom in chartitemdf.label]
        # _uom = chartitemdf.uom_proc[p_mask].unique()[0]
        try:
            _uom = chartitemdf.uom_proc[chartitemdf.label==pattern].unique()[0]
        except:
            _uom = None

        pdf_grid = np.linspace(itemsdf[col].min(), itemsdf[col].max(), 100)
        bw = getBandwidth(np.array(itemsdf[col]))

        if age_lim is not None:

            fig, axs = plt.subplots(nrows=int(np.ceil(len(age_lim) / 2.)),
                                    ncols=2, figsize=(13, 7))

            col_series_all = list()
            n_pat_all = list()
            pdf_all = np.empty((0,pdf_grid.size))
            for age, caxis in zip(age_lim, axs.reshape(-1)):
                agedf = itemsdf.loc[(itemsdf.age >= age[0] * 365.) &
                                    (itemsdf.age < age[-1] * 365.), :]
                n_pat = len(agedf.patient_id.unique())
                n_pat_all.append(n_pat)
                plt.sca(caxis)
                # stop

                try:
                    col_series = agedf[col]
                    col_series_all.append(col_series)
                    data = np.array(col_series)
                    pdf = kde_sklearn(data, pdf_grid, bandwidth=bw)
                    pdf_all = np.append(pdf_all, [pdf], axis=0)
                    plt.plot(pdf_grid, pdf, linewidth=3)
                    plt.hold(True)
                    # stop
                    plt.hist(data, bins=np.arange(min(data), max(data)+bw, bw),
                             normed=True)
                    # col_series.plot.kde()
                    plt.title("{} Distribution ({} patients) \n Age between {:.2f} - {:.2f} (yr)". \
                              format(col, n_pat, age[0], age[-1]))
                    plt.ylabel('Distribution')
                    plt.xlabel('{} ({})'.format(col, _uom))
                    if val_lim is not None:
                        plt.xlim(val_lim)
                    if pdf_lim is not None:
                        plt.ylim(pdf_lim)
                except:
                    pass
            fig.tight_layout()
            plt.show()

            # adj_lim = raw_input("Adjust X/Y limits? (y/n)")
            adj_lim = 'n'
            if adj_lim.lower() == 'y':
                fig, axs = plt.subplots(nrows=int(np.ceil(len(age_lim) / 2.)),
                                        ncols=2, figsize=(13, 7))
                _xlim = raw_input("x-axis limit? (llim, hlim)")
                _ylim = raw_input("y-axis limit? (llim, hlim)")
                for caxis, age, n_pat, col_series, pdf \
                        in zip(axs.reshape(-1), age_lim, n_pat_all, col_series_all, pdf_all):
                    plt.sca(caxis)
                    plt.plot(pdf_grid, pdf, linewidth=3)
                    plt.hold(True)
                    # data = np.array(itemsdf[col])
                    plt.hist(col_series, bins=np.arange(min(col_series), max(col_series)+bw, bw),
                             normed=True)
                    # col_series.plot.kde()
                    plt.title("{} Distribution ({} patients) \n Age between {:.2f} - {:.2f} (yr)". \
                              format(col, n_pat, age[0], age[-1]))
                    plt.ylabel('Distribution')
                    plt.xlabel('{} ({})'.format(col, _uom))
                    eval("plt.xlim({})".format(_xlim))
                    eval("plt.ylim({})".format(_ylim))
            fig.tight_layout()

        else:

            n_pat = len(itemsdf.patient_id.unique())
            fig = plt.figure()
            data = np.array(itemsdf[col])
            pdf_all = kde_sklearn(data, pdf_grid, bandwidth=bw)
            plt.plot(pdf_grid, pdf_all, linewidth=3)
            plt.hold(True)
            plt.hist(data, bins=np.arange(min(data), max(data)+bw, bw),
                     normed=True)
            # itemsdf[col].plot.kde()
            plt.title("{} Distribution ({} patients)".format(col, n_pat))
            plt.xlabel("{} ({})".format(col, _uom))
            plt.ylabel('Distribution')
            if val_lim is not None:
                plt.xlim(val_lim)
            if pdf_lim is not None:
                plt.ylim(pdf_lim)
            fig.tight_layout()
            plt.show()

            # adj_lim = raw_input("Adjust X/Y limits? (y/n)")
            adj_lim = 'n'
            if adj_lim.lower() == 'y':
                fig = plt.figure()
                # plt.rand_pick.plot.kde()
                plt.plot(pdf_grid, pdf_all, linewidth=3)
                plt.hold(True)
                plt.hist(data, bins=np.arange(min(data), max(data) + bw, bw),
                         normed=True)
                plt.title("{} Distribution ({} patients)".format(col, n_pat))
                plt.xlabel("{} ({})".format(col, _uom))
                plt.ylabel('Distribution')
                _xlim = raw_input("x-axis limit? (llim, hlim)")
                _ylim = raw_input("y-axis limit? (llim, hlim)")
                eval("plt.xlim({})".format(_xlim))
                eval("plt.ylim({})".format(_ylim))
            fig.tight_layout()
            plt.show()

        if dump_fig:
            fig_dir = os.path.join(os.path.dirname("__file__"), "distribution_io_{}_{}".format(_reftype, dbstring))
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)
            fig_fname = os.path.join(fig_dir, "{}.png".format(col))
            fig.savefig(fig_fname)

        pkl_fname = os.path.join(fig_dir, "{}.pkl".format(col))
        with open(pkl_fname, "wb") as f:
            if age_lim is not None:
                pickle.dump([col, age_lim, pdf_grid, pdf_all], f)
            else:
                pickle.dump([col, pdf_grid, pdf_all], f)

def pickFromPdf(_pdf_grid, _pdf, num_pick=1):
    """
    Function that randomly generates number according to a given pdf
    :param _pdf_grid: possible values
    :param _pdf: probability density function values at _pdf_grid
    :param num_pick: number of random pick
    :return: randomly picked number(s) from given pdf
    """

    dx = np.diff(_pdf_grid)[0]
    cdf = np.cumsum(_pdf)*dx
    cdf = cdf/cdf[-1]

    pick = [random.random() for count in range(num_pick)]

    pick_val = np.interp(pick, cdf, _pdf_grid)
    return pick_val

def meanFromPdf(_pdf_grid, _pdf):
    """
    Function that calculates the mean from pdf
    :param _pdf_grid: possible values
    :param _pdf: probability density function values at _pdf_grid
    :return: mean of the pdf
    """

    dx = np.diff(_pdf_grid)[0]
    norm_factor = np.sum(_pdf)*dx
    pdf = _pdf/norm_factor

    return np.sum(np.multiply(_pdf_grid, pdf))*dx

def pickFromAge(age_list, _pdf_grid, _pdf_all, _age_lim, mode='mean'):
    """
    Function that picks values for a given age (age_list) according to
    the corresponding pdf at the queried age
    :param age_list: age list of patients with missing value. (query age)
    :param _pdf_grid: possible values
    :param _pdf_all: list of pdf for each age limit
    :param _age_lim: age limit for each pdf in pdf_all
    :param mode: picking mode 'mean' or 'random'
    :return: list of values picked according to pdf. Same length with age_list.
    """

    fill_val = np.array([])
    age_lim = np.array(_age_lim).transpose()
    for age in age_list:
        age_mask = (age/365.>=age_lim[0]) & (age/365.<age_lim[1])
        # print(age)
        # print(np.where(age_mask))
        idx = np.where(age_mask)[0][0]
        pdf = _pdf_all[idx,:]
        if mode=='random':
            ran_pick = pickFromPdf(_pdf_grid, pdf)
            fill_val = np.append(fill_val, ran_pick)
        elif mode=='mean':
            mean_val = meanFromPdf(_pdf_grid, pdf)
            fill_val = np.append(fill_val, mean_val)
    return fill_val




def fillMissing(_io_mat, **kwargs):
    """
    Function that fills the missing values of predictor/response variable matrix
    :param _io_mat:
    :param kwargs: {mode: 'mean' or 'median' or 'mode' or 'age_mean' or 'age_rand'
                    ref_time: 'onset' or 'admit'}
    :return: Filled predictor/response variable matrix
    """

    io_mat_filled = _io_mat.copy(deep=True)

    for col in io_mat_filled.columns:
        if (pd.isnull(io_mat_filled[col])==True).sum()>0:
            miss_idx = pd.isnull(io_mat_filled[col])
            aval_idx = ~miss_idx
            miss_age = io_mat_filled.age[miss_idx]
            mode = kwargs['mode']
            if mode=='mean':
                io_mat_filled.loc[miss_idx, col] = io_mat_filled.loc[aval_idx, col].mean()
            elif mode == 'median':
                io_mat_filled.loc[miss_idx, col] = io_mat_filled.loc[aval_idx, col].median()
            elif mode =='mode':
                mode_pool = io_mat_filled.loc[aval_idx, col].mode()
                mode_pick = [random.choice(mode_pool.values) for count in len(miss_idx)]
                io_mat_filled.loc[miss_idx, col] = mode_pick
            elif mode=='age_mean':

                pdfdir = os.path.join(os.path.dirname("__file__"),
                                      "distribution_io_{}".format(kwargs['ref_time']))
                pkl_fname = os.path.join(pdfdir, col + '.pkl')
                label, age_lim, pdf_grid, pdf_all = pickle.load(open(pkl_fname, 'rb'))
                io_mat_filled.loc[miss_idx, col] = pickFromAge(miss_age,
                                                         pdf_grid, pdf_all, age_lim, mode='mean')
            elif mode=='age_rand':
                pdfdir = os.path.join(os.path.dirname("__file__"),
                                      "distribution_io_{}".format(kwargs['ref_time']))
                pkl_fname = os.path.join(pdfdir, col + '.pkl')
                label, age_lim, pdf_grid, pdf_all = pickle.load(open(pkl_fname, 'rb'))

                io_mat_filled.loc[miss_idx, col] = pickFromAge(miss_age,
                                                         pdf_grid, pdf_all, age_lim, mode='random')
            else:
                pass



    # print(_io_mat.nsbp_min[0])
    return io_mat_filled


def calAge(_group):
    _group['age'] = (_group.intime - _group.dob).astype('timedelta64[h]') / 24
    _group['age'] = _group.age.astype('int')
    return _group



