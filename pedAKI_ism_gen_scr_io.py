"""
update:
07/30/2016: Used to create scr_ism2. scr_ism2 uses a new criteria when scr_rate is calculated.
-48 hour time window and least square linear fit is used to calculate the scr_rate.
Used to create io_ism3. io_ism3 uses scr_ism2. Only includes variables directly queried from the ismdb.
Does not include composite variables such as si, osi, oi. These composite variabels will be included later
 in io_ism4
"""


import numpy as np
import os
import pandas as pd
import pickle

np.seterr(divide='ignore')

import ism_utilities_Ben as ism
# import ism_utilities as ism
import pedAKI_utilities as paki



def genIO_onset(timelag, timewin, stable_time):
    fileDir = os.path.dirname("__file__")
    if not os.path.exists(os.path.join(fileDir, 'scr_ism')):
        os.makedirs(os.path.join(fileDir, 'scr_ism'))

    if not os.path.exists(os.path.join(fileDir, 'io_ism')):
        os.makedirs(os.path.join(fileDir, 'io_ism'))

    path2normscr = os.path.join(fileDir, "csv_files", "ped_normal_scr.csv")
    path2chartitem_csv = os.path.join(fileDir, "csv_files", "chartitem.csv")
    # list_suff = pickle.load(open(os.path.join(fileDir, 'pickle_files', 'feature_stats.pkl'), 'rb'))
    list_suff = pickle.load(open(os.path.join(fileDir, 'pickle_files', 'feature_stats.pkl'), 'r'))

    # list_suff = pickle.load(f)

    ismdb = ism.queryISM()
    # ismdb = ism.queryDB()

    # ism_scr = ismdb.getItemData(1461, 1, 'mg/dl', unique_census_time=True)
    ism_scr = pd.read_pickle(os.path.join(fileDir, 'item_df_ism', 'ism_creatinine_df.pkl'))
    if 'AKI_stage' not in ism_scr.columns:
        ism_scr = paki.cleanSCrDf2(ism_scr, ex_age=True, ex_los=True, ex_aki_adm=True, aki_adm_hr=12, enc_per_pat=True)
        ism_scr.to_pickle(os.path.join(fileDir, 'item_df_ism', 'ism_creatinine_df.pkl'))

    # Clean the SCr dataframe and get the AKI group and Control group
    fname_ism_scr_tot = "ism_onset_scr_tlag{:03d}_stime{:03d}_tot.pkl".\
        format(int(abs(timelag)), int(stable_time))
    fname_ism_scr_tot = os.path.join(fileDir, 'scr_ism', fname_ism_scr_tot)
    fname_ism_scr_aki = "ism_onset_scr_tlag{:03d}_stime{:03d}_aki.pkl".\
        format(int(abs(timelag)), int(stable_time))
    fname_ism_scr_aki = os.path.join(fileDir, 'scr_ism', fname_ism_scr_aki)
    fname_ism_scr_con = "ism_onset_scr_tlag{:03d}_stime{:03d}_con.pkl".\
        format(int(abs(timelag)), int(stable_time))
    fname_ism_scr_con = os.path.join(fileDir, 'scr_ism', fname_ism_scr_con)



    try:
        ism_scr_tot = pd.read_pickle(fname_ism_scr_tot)
    except:
        ism_scr_tot = paki.cleanSCrDf2(ism_scr, ex_age=True, ex_los=True, ex_aki_adm=True,
                                       aki_adm_hr=np.max([stable_time, abs(timelag)]), enc_per_pat=True)
        ism_scr_tot.to_pickle(fname_ism_scr_tot)

    try:
        ism_scr_aki = pd.read_pickle(fname_ism_scr_aki)
    except:
        ism_scr_aki = paki.cleanSCrDf2(ism_scr_tot, ex_age=True, ex_los=True,
                                       ex_aki_adm=True, aki_adm_hr=np.max([stable_time, abs(timelag)]),
                                       ex_noaki=True, enc_per_pat=True)
        ism_scr_aki.to_pickle(fname_ism_scr_aki)

    try:
        ism_scr_con = pd.read_pickle(fname_ism_scr_con)
    except:
        ism_scr_con = ism_scr_tot.loc[~np.in1d(ism_scr_tot.patient_id, ism_scr_aki.patient_id), :]
        ism_scr_con.to_pickle(fname_ism_scr_con)

    fname_ism_onset_io_aki = "ism_onset_io_tlag{:03d}_twin{:03d}_aki.pkl".format(int(abs(timelag)), timewin)
    fname_ism_onset_io_aki = os.path.join(fileDir, "io_ism", fname_ism_onset_io_aki)
    fname_ism_onset_io_con = "ism_onset_io_tlag{:03d}_twin{:03d}_con.pkl".format(int(abs(timelag)), timewin)
    fname_ism_onset_io_con = os.path.join(fileDir, "io_ism", fname_ism_onset_io_con)

    if os.path.isfile(fname_ism_onset_io_aki):
        print("AKI group io dataframe already exists..")
    else:
        print('Creating AKI group io dataframe')
        ism_onset_io_aki = ismdb.getIOMatrix(ism_scr_aki, path2chartitem_csv, list_suff, timelag,
                                             twindow=timewin)
        ism_onset_io_aki.to_pickle(fname_ism_onset_io_aki)

    if os.path.isfile(fname_ism_onset_io_con):
        print("AKI group io dataframe already exists..")
    else:
        print('Creating control group io dataframe')
        ism_onset_io_con = ismdb.getIOMatrix(ism_scr_con, path2chartitem_csv, list_suff, timelag,
                                             twindow=timewin)
        ism_onset_io_con.to_pickle(fname_ism_onset_io_con)









