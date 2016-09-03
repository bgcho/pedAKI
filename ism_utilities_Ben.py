# -*- coding: utf-8 -*-
"""@package docstring
Created on Sun Jan 04 16:55:05 2015

@author: 310050083
last modified: 20160707 by 310248864, Ben ByungGu Cho
"""

import os
import numpy as np
import pedAKI_utilities as paki

from sqlalchemy.orm import create_session
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, DateTime, Float
from sqlalchemy import create_engine, MetaData, Table, func

import pandas as pd
import pickle
import matplotlib
import matplotlib.pyplot as plt
import random
import re


matplotlib.style.use('ggplot')
np.seterr(divide='ignore')


fileDir = os.path.dirname("__file__")

ped_norm_scr = pd.DataFrame.from_csv(os.path.join(fileDir, 'csv_files',
                                                  'ped_normal_scr.csv'))


###############################################################################
# connection string to ISM3 databse
db_conn_str = 'mssql+pyodbc://HirbaDbUser:hirba4u@130.140.52.181/ISM3_new?driver=SQL+Server+Native+Client+11.0'
#db_conn_str = 'mssql+pyodbc://HirbaDbUser:hirba4u@130.140.52.181/ISM3_new?driver=FreeTDS'
# db_conn_str = 'mssql+pymssql://HirbaDbUser:hirba4u@130.140.52.181/ISM3_new'
#db_conn_str = 'mssql://HirbaDbUser:hirba4u@130.140.52.181/ISM3_new'
Base = declarative_base()
engine = create_engine(db_conn_str)
metadata = MetaData(bind=engine)
#Create a session to use the tables
session = create_session(bind=engine)

###############################################################################
# Define tables from ISM3
###############################################################################
class CensusEvents(Base):
    __table__ = Table('CENSUSEVENTS', metadata,
                      Column("CEID", Float, primary_key = True),
                      Column("INTIME", DateTime, primary_key = True),
                      autoload = True)

    def __repr__(self):
        return "CensusEvents(subject_id={}, encounter_id={}, intime={}, \
                             outtime={}, los={})".\
                             format(self.PIDN, self.CEID, self.INTIME, \
                             self.OUTTIME, self.LOS)
###############################################################################
class Patients(Base):
    __table__ = Table('D_PATIENTS', metadata,
                      Column("PIDN", Integer, primary_key = True),
                      Column("DOB", DateTime, primary_key = True),
                      autoload = True)
    def __repr__(self):
        return "Patients(subject_id={}, dob={}, sex={})".\
                    format(self.PIDN, self.DOB, self.SEX)
###############################################################################
class ChartEvents(Base):
    __table__ = Table('CHARTEVENTS', metadata,
                      Column("PIDN", Integer, primary_key = True),
                      Column("ITEMID", Integer, primary_key = True),
                      Column("CHARTTIME", Integer, primary_key = True),
                      Column("REALTIME", Integer, primary_key = True),
                      autoload=True)
    def __repr__(self):
        return "ChartEvents(subject_id={}, item_id={}, chart_time={}, \
                real_time{})".format(self.PIDN, self.ITEMID, self.CHARTTIME, \
                self.REALTIME)
###############################################################################
class ChartItems(Base):
    __table__ = Table('D_CHARTITEMS', metadata,
                      Column("ITEMID", Float, primary_key=True),
                      autoload=True)

    def __repr__(self):
        return "ChartItems(item_id={}, label={}". \
            format(self.ITEMID, self.LABEL)

###############################################################################
# Define queryISM class
###############################################################################
class queryISM:
    
    def __init__(self):
        self.session = session
        self.engine = engine
        self.ChartEvents = ChartEvents
        self.ChartItems = ChartItems
        # self.MedEvents = MedEvents
        self.CensusEvents = CensusEvents
        # self.IOEvents = IOEvents
        self.Patients = Patients
    
###############################################################################
    def _notempty(self,df):
        if df.empty:
            flag=0
        else:
            flag=1
        return flag

########################### Added by Ben ######################################
    def calAge(self, _group):
        _group['age'] = (_group.intime - _group.dob).astype('timedelta64[h]') / 24
        _group['age'] = _group.age.astype('int')
        return _group

    def getItemData(self, _itemid, _valcol, uom_str=None, encounter_list=None,
                    unique_census_time=False, in_anno=False):
        """
        Function queries the CHARTEVENTS, CENSUSCEVENTS, and D_PATIENTS table
        for an item.
        :param _itemid: Integer. CHLA item_id of interest
        :param _valcol: Integer. Either "1" or "2". 1 corresponds to value1 column, 2 corresponds to value2
        :param uom_str: String. Unique unit of measurement.
          Use this option only when the user is sure about the unit of measurement.
        :param encounter_list: List of integers. CHLA CEID list of interest
        :param unique_census_time: Boolean. Flag to set unique intime and outtime for each encounter
        :param in_anno: Boolean. Flag to include annoation in the return dataframe
        :return: Dataframe. Dataframe queried by item_id and encounter_id
        """

        # Query ChartEvents table
        _itemdataframe = pd.DataFrame()
        if encounter_list is not None:
            # try:
            #     qchartevents = session.query(self.ChartEvents). \
            #         filter(self.ChartEvents.CEID.in_(encounter_list)). \
            #         filter(self.ChartEvents.ITEMID == _itemid).\
            #         filter(self.ChartEvents.PIDN != 0)
            #     _itemdataframe = pd.read_sql(qchartevents.statement, qchartevents.session.bind)
            # except:
            qchartevents = session.query(self.ChartEvents). \
                filter(self.ChartEvents.ITEMID == _itemid). \
                filter(self.ChartEvents.PIDN != 0)
            _itemdataframe = pd.read_sql(qchartevents.statement, qchartevents.session.bind)
            _itemdataframe = _itemdataframe.loc[np.in1d(_itemdataframe.CEID, encounter_list), :]

        else:
            qchartevents = session.query(self.ChartEvents).\
                filter(self.ChartEvents.ITEMID == _itemid).\
                filter(self.ChartEvents.PIDN != 0)
            _itemdataframe = pd.read_sql(qchartevents.statement, qchartevents.session.bind)

        if in_anno:
            itemdataframe = _itemdataframe[['ITEMID', 'CEID', 'PIDN', 'CHARTTIME',
                                            'VALUE1NUM', 'VALUE1', 'VALUE2NUM', 'VALUE2',
                                            'ANNOTATION']]
            itemdataframe.columns = ['item_id', 'encounter_id', 'patient_id', 'charttime',
                                     'value1', 'valstr1', 'value2', 'valstr2',
                                     'annotation']
        else:
            itemdataframe = _itemdataframe[['ITEMID','CEID', 'PIDN', 'CHARTTIME',
                                            'VALUE1NUM', 'VALUE1', 'VALUE2NUM', 'VALUE2']]
            itemdataframe.columns = ['item_id', 'encounter_id', 'patient_id', 'charttime',
                                     'value1', 'valstr1', 'value2', 'valstr2']
        # stop
        n_rows = len(itemdataframe.index)
        itemdataframe.loc[:,'valcol'] = pd.Series(_valcol*np.ones((n_rows,), dtype=np.int), index=itemdataframe.index)

        if uom_str is not None:
            uom_str_uq = [uom_str for x in itemdataframe.index]
            itemdataframe.loc[:,'uom'] = pd.Series(uom_str_uq, index=itemdataframe.index)
        else:
            if _valcol==1:
                itemdataframe.loc[:,'uom'] = _itemdataframe['VALUE1UOM']
            elif _valcol==2:
                itemdataframe.loc[:,'uom'] = _itemdataframe['VALUE2UOM']


        qpatients = session.query(self.Patients).\
            filter(self.Patients.PIDN !=0)

        patientsdataframe = pd.read_sql(qpatients.statement, qpatients.session.bind)
        patientsdataframe = patientsdataframe[['PIDN', 'SEX', 'DOB']]
        patientsdataframe.columns = ['patient_id', 'sex', 'dob']
        ## Merge two data frames
        itemdataframe = itemdataframe.merge(patientsdataframe, on='patient_id')

        if not os.path.exists(os.path.join(fileDir, 'pickle_files')):
            os.makedirs(os.path.join(fileDir, 'pickle_files'))
        path_census_df = os.path.join(fileDir, 'pickle_files', 'census_df.pkl')
        if (os.path.isfile(path_census_df)) & (unique_census_time is True):
            censusdataframe = pd.read_pickle(path_census_df)
        elif unique_census_time is True:
            qcensus = session.query(self.CensusEvents).\
                filter(self.CensusEvents.PIDN!=0)
            censusdataframe = pd.read_sql(qcensus.statement, qcensus.session.bind)
            censusdataframe = censusdataframe[['CEID', 'INTIME', 'OUTTIME', 'LOS', 'DISCHSTATUS']]
            censusdataframe.columns = ['encounter_id', 'intime', 'outtime', 'los', 'dischstatus']
            censusdataframe.sort_values(by=['encounter_id', 'intime'], ascending=[1,1], inplace=True)

            inout_per_encounter = censusdataframe.groupby('encounter_id')['intime'].count()
            inout_encnt = inout_per_encounter.index
            inout_count = inout_per_encounter.values
            inout_cmcnt = np.cumsum(inout_count)

            prev_count = 0
            to_drop_indx = np.array([])
            for encounter, curr_count in zip(inout_encnt, inout_cmcnt):
                # print(encounter)
                inout_df = censusdataframe.loc[censusdataframe.index[prev_count:curr_count],:]
                if len(inout_df.intime.unique())>1:
                    try:
                        glob_intime = min(inout_df.intime)
                        glob_outtime = max(inout_df.outtime)
                        glob_los = (glob_outtime - glob_intime).total_seconds() // 60
                        glob_dischstatus = inout_df.dischstatus[inout_df.outtime == glob_outtime]
                        censusdataframe.loc[
                            censusdataframe.index[prev_count:curr_count], 'intime'] = glob_intime
                        censusdataframe.loc[
                            censusdataframe.index[prev_count:curr_count], 'outtime'] = glob_outtime
                        censusdataframe.loc[
                            censusdataframe.index[prev_count:curr_count], 'los'] = glob_los
                        censusdataframe.loc[
                            censusdataframe.index[prev_count:curr_count], 'dischstatus'] = glob_dischstatus
                    except:
                        pass
                prev_count = curr_count
            censusdataframe.to_pickle(path_census_df)
        else:
            qcensus = session.query(self.CensusEvents). \
                filter(self.CensusEvents.PIDN != 0)
            censusdataframe = pd.read_sql(qcensus.statement, qcensus.session.bind)
            censusdataframe = censusdataframe[['CEID', 'INTIME', 'OUTTIME', 'LOS', 'DISCHSTATUS']]
            censusdataframe.columns = ['encounter_id', 'intime', 'outtime', 'los', 'dischstatus']

        itemdataframe = itemdataframe.merge(censusdataframe, on='encounter_id')

        # Exclude patients with no 'charttime', 'dob', 'sex'
        itemdataframe = itemdataframe.dropna(subset=['charttime', 'dob', 'sex'])

        ## Exclude patients with no 'dob' OR 'sex'
        # missing_key = ['sex', 'dob']
        # null_patient = np.array([])
        # for key in missing_key:
        #     null_patient = np.concatenate((null_patient,
        #                                    itemdataframe.patient_id[pd.isnull(itemdataframe[key])]),
        #                                   axis=0)
        # itemdataframe = itemdataframe.loc[np.in1d(~itemdataframe.patient_id, null_patient),:]


        # for key in missing_key:
        #     nullkey_patient = itemdataframe.loc[pd.isnull(itemdataframe[key]), key]. \
        #         groupby(itemdataframe.patient_id).unique().index
        #
        #     for patient in nullkey_patient:
        #         tmp_key = itemdataframe.loc[itemdataframe.patient_id == patient, key].unique()
        #         real_key = tmp_key[~pd.isnull(tmp_key)]
        #
        #         if real_key:
        #             try:
        #                 itemdataframe.loc[itemdataframe.patient_id == patient, key] = real_key
        #             except:
        #                 itemdataframe = itemdataframe.loc[itemdataframe.patient_id != patient, :]
        #         else:
        #             itemdataframe = itemdataframe.loc[itemdataframe.patient_id != patient, :]

        itemdataframe.sort_values(['encounter_id', 'charttime'], inplace=True)

        itemdataframe = itemdataframe.groupby('encounter_id').apply(self.calAge)

        return itemdataframe

    def queryChartItem(self, _itemid):
        """
        Function queries the D_CHARTITEMS table to get the label corresponding to the item_id
        :param _itemid: Integer. CHLA item_id
        :return:
        """
        qchartitem = session.query(self.ChartItems). \
            filter(self.ChartItems.ITEMID == _itemid)

        chartitem = pd.read_sql(qchartitem.statement, qchartitem.session.bind)
        chartitem = chartitem.loc[:,['ITEMID', 'LABEL']]
        chartitem.columns = ['item_id', 'label']

        return chartitem

    def checkUom(self, _itemid, _val_col, _label):
        """
        Function that checks the unit of measurement (uom) from ChartItems table and ChartEvents table
        - Checks the label of ChartItems table
        - Checks the uom of ChartEvents table
        - Checks the annotation of ChartEvents table
        :param _itemid: CHLA item_id of interest
        :param _val_col: value column corresponding to the _itemid
        :param _label: Name of the variable. For example, Temperature, Heart rate, ...
        :return:
        """

        df = self.getItemData(_itemid, _val_col, unique_census_time=True, in_anno=True)
        chartitem = self.queryChartItem(_itemid)
        print("{} uom in ChartItems table: {}".format(_label, chartitem.label))
        print("{} uom in the uom column: {}".format(_label, df.uom.unique()))

        num_nulluom = (pd.isnull(df.uom) == True).sum()
        print("There are {} rows with null uom in the {} table". \
              format(num_nulluom, _label))
        if num_nulluom > 0.001 * len(df.index):
            pattern_end = r';'
            pattern_sta = r'UNITS:'
            uom_str = np.array([])
            nul_idx = np.array([])
            nul_str = np.array([])

            for row, count in zip(df.index, np.arange(len(df.index))):

                anno_str = df.annotation[row]

                try:
                    idx_end = [idx.start(0) for idx in re.finditer(pattern_end, anno_str)]
                    idx_sta = [idx.end(0) for idx in re.finditer(pattern_sta, anno_str)]
                    uom_str = np.append(uom_str,
                                        anno_str[idx_sta[0]:idx_end[0]]. \
                                        lower().replace(" ", ""))
                    uom_str = np.unique(uom_str)

                except:
                    nul_idx = np.append(nul_idx, row)
                    nul_str = np.append(nul_str, anno_str)
                    nul_str = np.unique(nul_str)
                    pass

            print("uom in the annotation column is {}".format(uom_str))
            print("number of rows that don't have uom in the annotation column is {}".format(len(nul_idx)))
            print("annotation for null-uom rows is {}".format(nul_str))

    def convertLacticAcidUom(self, encounter_list=None, uom_str=None):
        """
        Function that converst uom of itemid=1531 (lactic-acid) from mmol/dl to mg/dl
        :return:
        """
        itemid = 1531
        val_col = 1
        f_lactic_acid = os.path.join(fileDir, 'item_df_ism', 'ism_lactic_acid_df.pkl')
        if os.path.exists(f_lactic_acid):
            df = pd.read_pickle(f_lactic_acid)
            if len(encounter_list)>0:
                df = df.loc[np.in1d(df.encounter_id, encounter_list), :]
            else:
                pass
        else:
            df = self.getItemData(itemid, val_col, encounter_list=encounter_list,
                                  uom_str=uom_str, unique_census_time=True,
                                  in_anno=True)
            df.to_pickle(f_lactic_acid)

        pattern_end = r';'
        pattern_sta = r'UNITS:'
        # uom_str = np.array([])
        # nul_idx = np.array([])
        # nul_str = np.array([])

        for row, count in zip(df.index, np.arange(len(df.index))):

            anno_str = df.annotation[row]
            try:
                idx_end = [idx.start(0) for idx in re.finditer(pattern_end, anno_str)]
                idx_sta = [idx.end(0) for idx in re.finditer(pattern_sta, anno_str)]
                anno_uom = anno_str[idx_sta[0]:idx_end[0]].lower().replace(" ", "")
                if anno_uom=='mmol/l':
                    df.value1[row] =  df.value1[row]/0.111
            except:
                pass

        return df





    def queryFeature(self, _df, _lag, twindow=1, scrdf=None,
                     enc_per_pat=False, plot_fig=False, dump_fig=False,
                     xlabel=None, ylabel=None,
                     xlim=None, ylim=None, folder=None,
                     label=None, abv=None, dump_pkl=False, plot_as='plot'):

        """
        Function that query feature with filtering options.
        :param _df: Dataframe of an item using ism_utilities_Ben.queryISM::getItemData
        :param _lag: Lag in hours from the reference time. Can be positive or negative
        :param ref_mode: (Optional) Reference time mode. Should be either "admit" OR "onset" OR "random"
        :param twindow: (Optional) Time window size for onset and random case. Default is 1 hour time window
        :param scrdf: (Optional) SCr dataframe. Only used for "onset" case to setup the reference time.
        :param enc_per_pat: (Optional) Take only one encounter for each patient with longest los.
        :param plot_fig: (optional) Flag to plot the item for each encounter
        :param dump_fig: (optional) Flag to save figure (.png and .pkl) for each encounter
        :param xlabel: (optional) String for x-axis label
        :param ylabel: (optional) String for y-axis label
        :param xlim: (optional) Tuple for x-axis limit. (xmin, xmax)
        :param ylim: (optional) Tuple for y-axis limit. (ymin, ymax)
        :param folder: (optional) Folder name to save the figures.
                        The figure files (.png, .pkl) are saved at ./folder directory.
        :param label: (Optional) Item name. For example, "Temperature", "Heart Rate", ...
        :param abv: (Optional) Abbreviation of variable name. For example, "temp", "hr", ...
        :param dump_pkl: (Optional) Flag to save the output dataframe as a .pkl file.
                          The pickle file is saved at ./pickle_files directory
        :param plot_as: Plotting option. For example, 'plot', 'semilogy', 'semilogx', ...
        :return: Dataframe of an item between _ref_time _ref_time+_lag for every encounter
        """
        _df.sort_values(by=['encounter_id', 'charttime'], inplace=True)
        # initialize the return dataframe
        # df = _df.copy(deep=True)

        # if ref_mode == "admit":
        #     _df['reftime'] = _df.intime
        # elif ref_mode == "onset":
        #     # _df = _df.groupby('encounter_id').apply(fillReftime, ref_mode, _lag, _scrdf)
        #     _df = _df.groupby('encounter_id').apply(paki.fillReftime, ref_mode=ref_mode, scr_df=scrdf)
        # elif ref_mode == "random":
        #     _df = _df.groupby('encounter_id').apply(paki.fillReftime, ref_mode=ref_mode, lag=_lag)
        # stop
        # _df = _df.merge(scrdf.loc[:, ['encounter_id', 'reftime']], on='encounter_id', how='left')
        if 'reftime' not in _df.columns:
            enc_reft = scrdf.groupby('encounter_id')['reftime'].unique().to_frame()
            enc_reft = enc_reft.reset_index()
            enc_reft['reftime'] = np.hstack(enc_reft.reftime)

            #     item_df = item_df.merge(scr_df.loc[:, ['encounter_id', 'reftime']], on='encounter_id', how='inner')
            _df = _df.merge(enc_reft, on='encounter_id', how='inner')

        if _lag > 0:
            _df['fromtime'] = _df.reftime
            _df['totime'] = _df.reftime + np.timedelta64(int(_lag*60), 'm')
            # time_mask = (_df.charttime > _df.reftime) & (_df.charttime < _df.totime)
        else:
            if 'reftime' not in _df.columns:
                stop
            _df['fromtime'] = _df.reftime + np.timedelta64(int(_lag*60), 'm')
            _df['totime'] = _df.reftime + np.timedelta64(int((_lag + twindow)*60), 'm')
            # time_mask = (_df.charttime < _df.reftime) & (_df.charttime > _df.totime)
        time_mask = (_df.charttime > _df.fromtime) & (_df.charttime < _df.totime)
        _df = _df.loc[time_mask, :]

        null_mask1 = (pd.isnull(_df.value1)) & (pd.isnull(_df.valstr1)) & (_df.valcol == 1)
        null_mask2 = ((pd.isnull(_df.value2)) & (pd.isnull(_df.valstr2)) & (_df.valcol == 2))
        null_mask = null_mask1 | null_mask2
        # null_mask = np.concatenate((null_mask, pd.isnull(df.value2[df.valcol == 2])), axis=0)
        _df = _df.loc[~null_mask, :]

        if enc_per_pat:
            _df = paki.filterEncPerPat(_df)


        fileDir = os.path.dirname("__file__")
        if plot_fig:
            # ceid_list = _df.encounter_id.unique()
            ceid_reftime = pd.DataFrame(_df.groupby(["encounter_id"])['reftime'].unique())
            ceid_reftime.reset_index(inplace=True)
            ceid_reftime.columns = ['encounter_id', 'reftime']
            ceid_reftime.reftime = np.hstack(ceid_reftime.reftime)

            for ceid, onset in zip(ceid_reftime.encounter_id, ceid_reftime.reftime):
                df_sub = _df.loc[_df.encounter_id == ceid, :]
                pid = df_sub.patient_id.unique()[0]
                dtime = (df_sub.charttime - onset).astype('timedelta64[m]') / 60
                ax = plt.figure()
                if df_sub.valcol.unique() == 1:
                    eval("plt.{}(dtime, df_sub.value1, marker='o',\
                                 linestyle='-', color='red', linewidth=3)". \
                         format(plot_as))
                elif df_sub.valcol.unique() == 2:
                    eval("plt.{}(dtime, df_sub.value2, marker='o',\
                                linestyle='-', color='red', linewidth=3)". \
                         format(plot_as))
                _title = "{} \n Patient ID={:05d} Encounter ID={:05d}". \
                    format(label, int(pid), int(ceid))
                plt.title(_title)
                if xlabel is not None:
                    plt.xlabel(xlabel)
                if ylabel is not None:
                    plt.ylabel(ylabel)
                plt.grid(True)
                #         plt.show()
                if xlim is not None:
                    plt.xlim(xlim)
                if ylim is not None:
                    plt.ylim(ylim)

                if dump_fig:
                    fig_dir = os.path.join(fileDir, folder)
                    if not os.path.exists(fig_dir):
                        os.makedirs(fig_dir)
                    _ftitle = os.path.join(fig_dir,
                                           '{}_pid{:05d}_ceid{:05d}.png'
                                           .format(abv, int(pid), int(ceid)))
                    plt.savefig(_ftitle, bbox_inches='tight')
                    pkl_output = open(_ftitle[:-3] + "pkl", 'wb')
                    pickle.dump(ax, pkl_output)
                    pkl_output.close()
                else:
                    plt.show()

        if dump_pkl:
            _ftitle_pkl = "{}_df.pkl".format(abv)
            pkl_dir = os.path.join(fileDir, 'pickle_files')
            if not os.path.exists(pkl_dir):
                os.makedirs(pkl_dir)
            _df.to_pickle(os.path.join(pkl_dir, _ftitle_pkl))

        return _df


    def getIOMatrix(self, _df, _path2chartitem_csv, _stat_dic, _lag, twindow=1):
        """
        Function that returns the predictor/response variable matrix to train/test the pediatric AKI
        prediction model.
        :param _df: Dataframe. Input SCr dataframe (Should have unique encounter per patient)
        :param _path2chartitem_csv: String. path to the chartitem.csv file that contains mapping between
        input variables in the pediatric AKI prediction model (label) and item_id in CHLA db (item_id_CHLA),
        value column in CHLA db (val_col_CHLA), and the unique unit of measurement (uom_proc).
        Note that uniqueness of the uom_proc is checked by checkUom(_itemid, _val_col, _label)
        :param _stat_dic: Dictionary. Statistics dictionary of interest for each input variable in the prediction model.
        Keys in the dictionary should be found in the label column in chartitem.csv.
        Each value should be a subset of ["min", "max", "mean", "median", "unique", "last"]
        :param _lag: Lag in hours from the reference time.
        Reference time is either admission time OR AKI onset time OR random time
        :param ref_mode: (Optional) Either "admit" OR "onset" OR "random"
        :param twindow: (Optional) Time window size that defines the time window within which the input data is extracted.
        Only used when _lag is negative. For example, data between onset+_lag and onset+_lag+twindow will be extracted
        :return: Input/Output Matrix for the pediatric AKI prediction model.
        For example, columns of the returned matrix are ['patient_id', 'encounter_id', 'AKI_stage', 'nsbp_min', ...]
        """


        encounter_id_list = _df.encounter_id.unique().astype(np.int32)

        chartitems_df = pd.DataFrame.from_csv(_path2chartitem_csv)

        list_pref = chartitems_df.label.unique()


        default_entry = ['encounter_id', 'age', 'sex']

        glob_mat = pd.DataFrame()
        for entry in default_entry:
            series = _df.groupby('patient_id')[entry].unique()
            series.name = entry
            df = pd.DataFrame(series)
            df.reset_index(inplace=True)
            if glob_mat.empty:
                glob_mat = df
            else:
                glob_mat = glob_mat.merge(df, on='patient_id')
            glob_mat[entry] = np.hstack(glob_mat[entry])


        out_col = pd.DataFrame(_df.groupby('patient_id')['AKI_stage'].max())
        out_col.reset_index(inplace=True)
        out_col.columns = ['patient_id', 'AKI_stage']
        glob_mat = glob_mat.merge(out_col, on="patient_id", how="left")

        for prefix in list_pref:
            print(prefix)
            fname_itemdf = os.path.join(fileDir, 'item_df_ism', 'ism_{}_df.pkl'.format(prefix))
            items = [(row[0], row[1], row[2]) for row in
                     zip(chartitems_df.loc[chartitems_df.label == prefix, 'item_id_CHLA'],
                         chartitems_df.loc[chartitems_df.label == prefix, 'val_col_CHLA'],
                         chartitems_df.loc[chartitems_df.label == prefix, 'uom_proc'])]
            if os.path.exists(fname_itemdf):
                itemsdf = pd.read_pickle(fname_itemdf)
                itemsdf = itemsdf.loc[np.in1d(itemsdf.encounter_id, encounter_id_list), :]
            else:
                itemsdf_big = pd.DataFrame()
                itemsdf = pd.DataFrame()

                for item in items:
                    scrdf_big = pd.read_pickle(os.path.join(fileDir, 'item_df_ism', 'ism_creatinine_df.pkl'))
                    # ceid_big = scrdf_big.encounter_id.unique().tolist()
                    ceid_big = scrdf_big.encounter_id.unique().astype(np.int32)
                    if prefix == 'lactic_acid':
                        # itemdf = self.convertLacticAcidUom(encounter_list=encounter_id_list, uom_str=item[2])
                        itemdf_big = self.convertLacticAcidUom(encounter_list=ceid_big, uom_str=item[2])
                        itemdf = itemdf_big.loc[np.in1d(itemdf_big.encounter_id, encounter_id_list), :]
                    else:
                        itemdf_big = self.getItemData(int(item[0]), int(item[1]), uom_str=item[2],
                                                      encounter_list=ceid_big,
                                                      unique_census_time=True, in_anno=True)
                        itemdf = itemdf_big.loc[np.in1d(itemdf_big.encounter_id, encounter_id_list), :]
                    itemsdf_big = itemsdf_big.append(itemdf_big)
                    itemsdf = itemsdf.append(itemdf)
                itemsdf_big.to_pickle(os.path.join(fileDir, 'item_df_ism', 'ism_{}_df.pkl'.format(prefix)))

            if not itemsdf.empty:
                itemsdf.sort_values(by=['encounter_id', 'charttime'], inplace=True)
                itemsdf = self.queryFeature(itemsdf, _lag, twindow=twindow, scrdf=_df)

            if items[0][1] == 1:
                itemsdf = itemsdf.loc[:, ['patient_id', 'encounter_id', 'charttime', 'intime', 'value1',
                                          'valstr1', 'uom']]
                itemsdf.columns = ['patient_id', 'encounter_id', 'charttime', 'intime', 'value',
                                   'valstr', 'uom']
            elif items[0][1] == 2:
                itemsdf = itemsdf.loc[:, ['patient_id', 'encounter_id', 'charttime', 'intime', 'value2',
                                          'valstr2', 'uom']]
                itemsdf.columns = ['patient_id', 'encounter_id', 'charttime', 'intime', 'value',
                                   'valstr', 'uom']

            # else:
            #     items = [(row[0], row[1], row[2]) for row in
            #              zip(chartitems_df.loc[chartitems_df.label == prefix, 'item_id_CHLA'],
            #                  chartitems_df.loc[chartitems_df.label == prefix, 'val_col_CHLA'],
            #                  chartitems_df.loc[chartitems_df.label == prefix, 'uom_proc'])]
            #     itemsdf = pd.DataFrame()
            #     # scrdf_all = pd.read_pickle(os.path.join(fileDir, 'item_df_ism', 'ism_creatinine_df.pkl'))
            #     # encounter_id_list_all = scrdf_all.encounter_id.unique().tolist()
            #     for item in items:
            #         itemdf = self.getItemData(int(item[0]), int(item[1]), uom_str=item[2],
            #                                   encounter_list=encounter_id_list,
            #                                   unique_census_time=True, in_anno=True)
            #
            #         if (not itemdf.empty) and ('value1' in itemdf.columns):
            #             itemdf.sort_values(by=['encounter_id', 'charttime'], inplace=True)
            #             itemdf = self.queryFeature(itemdf, _lag, twindow=twindow, scrdf=_df)
            #             if item[1] == 1:
            #                 itemdf = itemdf.loc[:, ['patient_id', 'encounter_id',
            #                                         'charttime', 'intime', 'value1',
            #                                         'valstr1', 'uom']]
            #                 itemdf.columns = ['patient_id', 'encounter_id',
            #                                   'charttime', 'intime', 'value',
            #                                   'valstr', 'uom']
            #             elif item[1] == 2:
            #                 itemdf = itemdf.loc[:, ['patient_id', 'encounter_id',
            #                                         'charttime', 'intime', 'value2',
            #                                         'valstr2', 'uom']]
            #                 itemdf.columns = ['patient_id', 'encounter_id',
            #                                   'charttime', 'intime', 'value',
            #                                   'valstr', 'uom']
            #         itemsdf - itemsdf.append(itemdf)
            #
            #
            #
            #
            #     if item[0]==1531:
            #         itemdf = self.convertLacticAcidUom(encounter_list=encounter_id_list,
            #                                            uom_str=item[2])
            #     else:
            #
            #         if os.path.exists(fname_itemdf):
            #             itemdf = pd.read_pickle(fname_itemdf)
            #             itemdf = itemdf.loc[np.in1d(itemdf.encounter_id, encounter_id_list), :]
            #         else:
            #             scrdf_all = pd.read_pickle(os.path.join(fileDir, 'item_df_ism', 'ism_creatinine_df.pkl'))
            #             encounter_id_list_all = scrdf_all.encounter_id.unique().tolist()
            #             itemdf_all = self.getItemData(int(item[0]), int(item[1]), uom_str=item[2],
            #                                           encounter_list=encounter_id_list_all,
            #                                           unique_census_time=True, in_anno=True)
            #             itemdf = itemdf_all.loc[np.in1d(itemdf_all.encounter_id, encounter_id_list),:]
            #
            #     if (not itemdf.empty) and ('value1' in itemdf.columns):
            #         # itemdf['AKI_stage'] = _df.AKI_stage[np.in1d(_df.index, itemdf.index)]
            #
            #         itemdf.sort_values(by=['encounter_id', 'charttime'], inplace=True)
            #         itemdf = self.queryFeature(itemdf, _lag, twindow=twindow, scrdf=_df)
            #
            #         if item[1] == 1:
            #             itemdf = itemdf.loc[:, ['patient_id', 'encounter_id',
            #                                     'charttime', 'intime', 'value1',
            #                                     'valstr1', 'uom']]
            #             itemdf.columns = ['patient_id', 'encounter_id',
            #                               'charttime', 'intime', 'value',
            #                               'valstr', 'uom']
            #         elif item[1] == 2:
            #             itemdf = itemdf.loc[:, ['patient_id', 'encounter_id',
            #                                     'charttime', 'intime', 'value2',
            #                                     'valstr2', 'uom']]
            #             itemdf.columns = ['patient_id', 'encounter_id',
            #                               'charttime', 'intime', 'value',
            #                               'valstr', 'uom']
            #     itemsdf = itemsdf.append(itemdf)

            try:
                itemsdf.sort_values(by=['encounter_id', 'charttime'], inplace=True)
            except:
                stop
            if prefix !='race':
                itemsdf.value = itemsdf.value.astype(float)
            itemsdf_grouped = itemsdf.groupby('patient_id')['value']
            itemsdf_grouped_str = itemsdf.groupby('patient_id')['valstr']

            # Get the input matrix
            suffices = _stat_dic[prefix]

            for suffix in suffices:
                col_inmat = pd.Series()
                label_full = prefix + "_" + suffix
                # print(label_full)
                if suffix == 'mean':
                    try:
                        col_inmat = itemsdf_grouped.mean()
                    except:
                        stop
                elif suffix == 'median':
                    col_inmat = itemsdf_grouped.median()
                elif suffix == 'max':
                    col_inmat = itemsdf_grouped.max()
                elif suffix == 'min':
                    col_inmat = itemsdf_grouped.min()
                elif suffix == 'unique':
                    col_inmat = np.hstack(itemsdf_grouped_str.unique())
                elif suffix == 'last':
                    col_inmat = itemsdf_grouped.apply(paki.getLastVal)

                col_inmat = pd.DataFrame(col_inmat)
                col_inmat.reset_index(inplace=True)
                col_inmat.columns = ['patient_id', label_full]

                glob_mat = glob_mat.merge(col_inmat, on='patient_id', how="left")
                # print("{} included in I/O dataframe".format(label_full))

        return glob_mat

    def getDistFeature(self, _itemid, _valcol, _label, _num_pick,
                       _encounter_list, uom_str=None, age_lim=None, sex=None,
                       xlim=None, ylim=None, abv=None, dump_fig=False):

        """
        Function that creates a distribution of a feature from ismdb
        :param _item: List of tuples of item_id
        :param _valcol: List of valcol. Same length with _item
        :param _label: label of the list of item_id
        :param _num_pick: number of random picks per patient
        :param _encounter_list: List of encounters
        :param age_lim: (Optional) List if (lowlim, uplim) tuple to filter age in years
        :param sex: (Optional) "F" or "M" String to filter gender
        :param xlim:
        :param ylim:
        :param abv:
        :param dump_fig:
        :return:
        """

        # ismdb = ism.queryISM()
        _encounter_list = np.int32(_encounter_list)
        _itemid = np.hstack([_itemid])
        _valcol = np.hstack([_valcol])

        # encounter_id_list = _scrdf.encounter_id.unique().astype(np.int32)

        itemsdf = pd.DataFrame()
        for item, valcol in zip(_itemid, _valcol):
            print("Querying {}".format(item))

            itemdf = self.getItemData(int(item), int(valcol), uom_str=uom_str,
                                       encounter_list=_encounter_list,
                                       unique_census_time=True, in_anno=True)
            if not itemdf.empty:

                # itemdf.sort_values(by=['encounter_id', 'charttime'], inplace=True)

                if valcol == 1:
                    itemdf = itemdf.loc[:, ['patient_id', 'encounter_id',
                                            'charttime', 'intime', 'outtime',
                                            'value1', 'valstr1', 'uom', 'sex', 'age']]
                    itemdf.columns = ['patient_id', 'encounter_id',
                                      'charttime', 'intime', 'outtime',
                                      'value', 'valstr', 'uom', 'sex', 'age']
                elif valcol == 2:
                    itemdf = itemdf.loc[:, ['patient_id', 'encounter_id',
                                            'charttime', 'intime', 'outtime',
                                            'value2', 'valstr2', 'uom', 'sex', 'age']]
                    itemdf.columns = ['patient_id', 'encounter_id',
                                      'charttime', 'intime', 'outtime',
                                      'value', 'valstr', 'uom', 'sex', 'age']
                itemsdf = itemsdf.append(itemdf)
        itemsdf = paki.filterEncPerPat(itemsdf)
        itemsdf.sort_values(by=['encounter_id', 'charttime'], inplace=True)
        print("Item dataframe queried.")
        pdf_grid = np.linspace(itemsdf.value.min(), itemsdf.value.max(), 100)

        bw_pool = np.array(itemsdf.value)
        bw_rpick = [random.choice(bw_pool) for count in range(2000)]
        bw = paki.getBandwidth(bw_rpick)

        if sex is not None:
            itemsdf = itemsdf.loc[itemsdf.sex == sex, :]

        if age_lim is not None:

            fig, axs = plt.subplots(nrows=int(np.ceil(len(age_lim) / 2.)), ncols=2, figsize=(13, 7))
            rand_pick_all = list()
            pdf_all = np.empty((0, pdf_grid.size))
            n_pat_all = list()
            for age, caxis in zip(age_lim, axs.reshape(-1)):
                print(age)
                agedf = itemsdf.loc[(itemsdf.age >= age[0] * 365.) & (itemsdf.age < age[-1] * 365.), :]
                n_pat = len(agedf.patient_id.unique())
                n_pat_all.append(n_pat)
                rand_pick = np.array([])
                for patient in agedf.patient_id.unique():
                    pool = np.array(agedf.value[agedf.patient_id == patient])
                    if pool.size > 0:
                        pick = [random.choice(pool) for count in range(_num_pick)]
                        rand_pick = np.concatenate((rand_pick, pick), axis=0)
                # rand_pick = pd.Series(rand_pick)
                pdf = paki.kde_sklearn(rand_pick, pdf_grid, bandwidth=bw)
                pdf_all = np.append(pdf_all, [pdf], axis=0)
                rand_pick_all.append(rand_pick)

                plt.sca(caxis)
                try:
                    # rand_pick.plot.kde()
                    plt.plot(pdf_grid, pdf, linewidth=3)
                    plt.hold(True)
                    plt.hist(rand_pick, bins=np.arange(min(rand_pick), max(rand_pick+bw), bw), normed=True)
                    plt.title("{} Distribution ({} patients) \n Age between {:.2f} - {:.2f} (yr)". \
                              format(_label, n_pat, age[0], age[-1]))
                    plt.xlabel("{} ({})".format(_label, itemsdf.uom.unique()[0]))
                    plt.ylabel('Distribution')
                    if xlim is not None:
                        plt.xlim(xlim)
                    if ylim is not None:
                        plt.ylim(ylim)
                except:
                    pass
            fig.tight_layout()
            plt.show()

            adj_lim = raw_input("Adjust X/Y limits? (y/n)")
            if adj_lim.lower() == 'y':
                _xlim = raw_input("x-axis limit? (llim, hlim)")
                _ylim = raw_input("y-axis limit? (llim, hlim)")
                fig, axs = plt.subplots(nrows=int(np.ceil(len(age_lim) / 2.)),
                                        ncols=2, figsize=(13, 7))
                for caxis, rand_pick, age, n_pat, pdf \
                        in zip(axs.reshape(-1), rand_pick_all, age_lim, n_pat_all, pdf_all):
                    # plt.figure(fig.number)
                    # print(caxis)
                    plt.sca(caxis)
                    # rand_pick.plot.kde()
                    plt.plot(pdf_grid, pdf, linewidth=3)
                    plt.hold(True)
                    plt.hist(rand_pick, bins=np.arange(min(rand_pick), max(rand_pick + bw), bw), normed=True)
                    plt.title("{} Distribution ({} patients) \n Age between {:.2f} - {:.2f} (yr)". \
                              format(_label, n_pat, age[0], age[-1]))
                    plt.xlabel("{} ({})".format(_label, itemsdf.uom.unique()[0]))
                    plt.ylabel('Distribution')
                    eval("plt.xlim({})".format(_xlim))
                    eval("plt.ylim({})".format(_ylim))
            fig.tight_layout()

        else:
            rand_pick = np.array([])
            n_pat = len(itemsdf.patient_id.unique())
            for patient in itemsdf.patient_id.unique():
                pool = np.array(itemsdf.value[itemsdf.patient_id == patient])
                if pool.size > 0:
                    pick = [random.choice(pool) for count in range(_num_pick)]
                    rand_pick = np.concatenate((rand_pick, pick), axis=0)

            # rand_pick = pd.Series(rand_pick)

            fig = plt.figure()
            pdf_all = paki.kde_sklearn(rand_pick, pdf_grid)
            plt.plot(pdf_grid, pdf_all, linewidth=3)
            # rand_pick.plot.kde()
            plt.title("{} Distribution ({} patients)".format(_label, n_pat))
            plt.xlabel("{} ({})".format(_label, itemsdf.uom.unique()[0]))
            plt.ylabel('Distribution')
            if xlim is not None:
                plt.xlim(xlim)
            if ylim is not None:
                plt.ylim(ylim)
            fig.tight_layout()
            plt.show()

            adj_lim = raw_input("Adjust X/Y limits? (y/n)")
            if adj_lim.lower() == 'y':
                fig = plt.figure()
                # plt.rand_pick.plot.kde()
                plt.plot(pdf_grid, pdf_all, linewidth=3)
                plt.title("{} Distribution ({} patients)".format(_label, n_pat))
                plt.xlabel("{} ({})".format(_label, itemsdf.uom.unique()[0]))
                plt.ylabel('Distribution')
                _xlim = raw_input("x-axis limit? (llim, hlim)")
                _ylim = raw_input("y-axis limit? (llim, hlim)")
                eval("plt.xlim({})".format(_xlim))
                eval("plt.ylim({})".format(_ylim))
            fig.tight_layout()

        if dump_fig:
            fig_dir = os.path.join(os.path.dirname("__file__"), "distribution")
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)
            fig_fname = os.path.join(fig_dir, "{}.png".format(abv))
            fig.savefig(fig_fname)
        pkl_fname = os.path.join(fig_dir, "{}.pkl".format(abv))

        with open(pkl_fname, "wb") as f:
            if age_lim is not None:
                pickle.dump([_label, age_lim, pdf_grid, pdf_all], f)
            else:
                pickle.dump([_label, pdf_grid, pdf_all], f)