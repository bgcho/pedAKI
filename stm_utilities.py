from __future__ import print_function
import sys
import os
import pandas as pd
import csv
import matplotlib.pyplot as plt
import matplotlib
import time
from datetime import date
from dateutil.relativedelta import relativedelta
from collections import defaultdict
import datetime
import numpy as np
import pylab as P
import math
import imp
import random
import re
import pdb
import stm_dbinit as dbinit
import stm_tabledef as ptd
import ConfigParser
import pickle

# Config = ConfigParser.ConfigParser()
# Config.read(r'config.ini')
# path = Config.get('path_data','path')
# connection_str = Config.get('path_db','connect_str')

fileDir = os.path.dirname(__file__)

pd.set_option("display.max_colwidth", 80)
pd.set_option("display.max_rows", 300)
pd.set_option("display.max_columns", 60)

path = "z:\\StMaryDataSet\\2016-02-19-Extracts"
# connection_str = 'sqlite:///G://PRNA//Projects//HII4PICU/StMary/stmary.sqlite3'
# connection_str = 'postgresql+psycopg2://postgres:prna@localhost:5432'
connection_str = 'postgresql+psycopg2://postgres:iccadev1@130.140.52.181:5432/postgres'

handle = open(os.path.join(fileDir, "pickle_files_stm", "feature_dict_stm.pkl"), 'r')
feature_dict = pickle.load(handle)

path2normscr = os.path.join(os.path.dirname("__file__"), "csv_files", "ped_normal_scr.csv")
norm_scr_lim = pd.DataFrame.from_csv(path2normscr)
age_yr_ll = np.array(norm_scr_lim.low_age)
age_yr_ul = np.array(norm_scr_lim.upp_age)


"""
Instruction for connection string
connection_str = 'postgresql+psycopg2://userid:pw@ipaddress:port/dbname'
On iccadev1,
userid: postgres
pw: iccadev1
ipaddress: 130.140.52.181
port: 5432
dbname: postgres (This is the Maintenance database name of the db that can be found in properties tab on the remote db)
"""



class importDB:
    def __init__(self):

        self.engine, self.session = dbinit.make_connection(connection_str, db_echo=False)

        print("Dropping all tables")
        dbinit.drop_tables(self.engine)
        print("Recreating tables")
        dbinit.create_tables(self.engine)

        encounter = self._get_encounter()
        attribute = self._get_attribute()
        intervention = self._get_intervention()
        material = self._get_material()
        assessment = self._get_assessment(attribute, intervention)
        labs = self._get_labs(attribute, intervention)
        meds = self._get_meds(attribute, intervention, material)
        fluids = self._get_fluids(attribute, intervention)

        # import tables
        self._import_encounter(encounter)
        self._import_chartevents(assessment, labs)
        self._import_medevents(meds)
        self._import_fluidevents(fluids)

    ###############################################################################
    def _strp_string(self, x):
        return x.strip('.0')

    def _str2float(self, x):
        try:
            return float(x)
        except ValueError:
            return np.nan

    ###############################################################################
    def _standard_datetime(self, tstamp):
        if pd.isnull(tstamp):
            return None
        else:
            return datetime.datetime(tstamp.year, tstamp.month,
                                     tstamp.day, tstamp.hour,
                                     tstamp.minute, tstamp.second)

    ###############################################################################
    def calc_age(self, row):
        return (row['inTime'] - row['dateOfBirth']).days / 365.25

    ###############################################################################
    def _get_encounter(self):
        # create encounter table  - merge encounter and census tables
        # process census table
        pt_census = pd.read_csv(os.path.join(path, 'SMHDARDB.PtCensus-DEID.tsv'), sep='\t', error_bad_lines=False)
        pt_census.loc[:, ['ptCensusId', 'encounterId', 'primaryDiagnosisId']] = \
            pt_census.loc[:, ['ptCensusId', 'encounterId']].astype(int)

        pt_census.loc[:, ['lengthOfStay']] = pt_census.loc[:, ['lengthOfStay']].astype(float)

        pt_census.loc[:, ['is24HrReAdmit', 'isDischarged', 'isDeceased', 'isTransferred']] = \
            pt_census.loc[:, ['is24HrReAdmit', 'isDischarged', 'isDeceased', 'isTransferred']].astype(bool)

        # pt_census.loc[:, ['lengthOfStay']] = pt_census.loc[:, ['lengthOfStay']].astype(float)
        pt_census.drop(['admitType', 'primaryDiagnosisId'], axis=1, inplace=True)

        pt_census.loc[:, 'inTime'] = pd.to_datetime(pt_census['inTime'])
        pt_census.loc[:, 'outTime'] = pd.to_datetime(pt_census['outTime'])
        pt_census.loc[:, 'utcInTime'] = pd.to_datetime(pt_census['utcInTime'])
        pt_census.loc[:, 'utcOutTime'] = pd.to_datetime(pt_census['utcOutTime'])

        # process encounter table
        encounter = pd.read_csv(os.path.join(path, 'SMHDARDB.D_Encounter-DEID.tsv'), sep='\t')
        encounter.loc[:, ['encounterId', 'patientId', 'episodeId']] = \
            encounter.loc[:, ['encounterId', 'patientId', 'episodeId']].astype(int)
        encounter.loc[:, 'dateOfBirth'] = pd.to_datetime(encounter['dateOfBirth'])
        encounter.loc[encounter.loc[:, 'gender'] == 'Male', 'gender'] = 1
        encounter.loc[encounter.loc[:, 'gender'] == 'Female', 'gender'] = 2
        encounter.loc[:, 'gender'] = encounter.loc[:, 'gender'].astype(float)
        encounter.drop(['patientType'], axis=1, inplace=True)
        encounter.dropna(subset=['dateOfBirth'], inplace=True)

        # merge encounter and census
        encounter_census = pd.merge(pt_census, encounter, on=['encounterId'])
        encounter_census.loc[:, 'age'] = encounter_census.apply(self.calc_age, axis=1)
        return encounter_census

    ###############################################################################
    def _get_labs(self, attribute, intervention):
        # merging ptLabs with attribute and interventions
        pt_labresults = pd.read_csv(os.path.join(path, 'SMHDARDB.PtLabResult-DEID.tsv'), sep='\t',
                                    error_bad_lines=False)
        pt_labresults.drop(['baseValueNumber', 'upperNormal', 'lowerNormal', 'baseUOM', 'verboseForm'], axis=1,
                           inplace=True)
        pt_labresults.loc[:, ['ptLabResultId', 'interventionId', 'attributeId', 'encounterId']] = \
            pt_labresults.loc[:, ['ptLabResultId', 'interventionId', 'attributeId', 'encounterId']].astype(int)
        pt_labresults.loc[:, ['valueNumber']] = pt_labresults.loc[:, ['valueNumber']].astype(float)
        pt_labresults.loc[:, ['unitOfMeasure']] = pt_labresults.loc[:, 'unitOfMeasure'].astype(str)
        pt_labresults.loc[:, 'chartTime'] = pd.to_datetime(pt_labresults['chartTime'])
        pt_labresults.loc[:, 'utcChartTime'] = pd.to_datetime(pt_labresults['utcChartTime'])

        tmp = pd.merge(pt_labresults, attribute, on='attributeId', how='left')
        tmp = pd.merge(tmp, intervention, on='interventionId', how='left')
        tmp.rename(columns=dict(conceptLabel_x='attrConceptLabel', conceptCode_x='attrConceptCode',
                                conceptLabel_y='intvConceptLabel', conceptCode_y='intvConceptCode',
                                conceptLabel='matConceptLabel', conceptCode='matConceptCode',
                                shortLabel_x='attrShortLabel', longLabel_x='attrLongLabel',
                                shortLabel_y='intvShortLabel', longLabel_y='intvLongLabel', shortLabel='matShortLabel',
                                longLabel='matLongLabel'), inplace=True)
        return tmp

    ###############################################################################
    def _get_meds(self, attribute, intervention, material):
        pt_medications = pd.read_csv(os.path.join(path, 'SMHDARDB.PtMedication-DEID.tsv'), sep='\t',
                                     error_bad_lines=False)
        pt_medications.drop(['verboseForm', 'baseValueNumber', 'baseUOM', 'upperNormal', 'lowerNormal'], axis=1,
                            inplace=True)
        pt_medications.loc[:,
        ['ptMedicationId', 'interventionId', 'attributeId', 'encounterId', 'materialId', 'valueNumber']] = \
            pt_medications.loc[:,
            ['ptMedicationId', 'interventionId', 'attributeId', 'encounterId', 'materialId', 'valueNumber']]. \
                convert_objects(convert_numeric=True)

        pt_medications.dropna(subset=['ptMedicationId', 'interventionId', 'attributeId', \
                                      'encounterId', 'materialId', 'valueNumber'], inplace=True)

        pt_medications.loc[:, ['ptMedicationId', 'interventionId', 'attributeId', 'encounterId', 'materialId']] = \
            pt_medications.loc[:,
            ['ptMedicationId', 'interventionId', 'attributeId', 'encounterId', 'materialId']].astype(int)
        pt_medications.loc[:, ['unitOfMeasure']] = pt_medications.loc[:, 'unitOfMeasure'].astype(str)
        pt_medications.loc[:, ['valueNumber']] = pt_medications.loc[:, ['valueNumber']].astype(float)
        pt_medications.loc[:, 'chartTime'] = pd.to_datetime(pt_medications['chartTime'])
        pt_medications.loc[:, 'utcChartTime'] = pd.to_datetime(pt_medications['utcChartTime'])

        # merging ptMedications with attribute and intervention
        tmp = pd.merge(pt_medications, attribute, how='left', on=['attributeId'])
        tmp = pd.merge(tmp, intervention, how='left', on=['interventionId'])
        tmp = pd.merge(tmp, material, how='left', on=['materialId'])
        tmp.dropna(subset=['valueNumber', 'chartTime'], inplace=True)

        tmp.rename(columns=dict(conceptLabel_x='attrConceptLabel', conceptCode_x='attrConceptCode',
                                conceptLabel_y='intvConceptLabel', conceptCode_y='intvConceptCode',
                                conceptLabel='matConceptLabel', conceptCode='matConceptCode',
                                shortLabel_x='attrShortLabel', longLabel_x='attrLongLabel',
                                shortLabel_y='intvShortLabel', longLabel_y='intvLongLabel', shortLabel='matShortLabel',
                                longLabel='matLongLabel'), inplace=True)

        return tmp
    ###############################################################################
    def _get_fluids(self, attribute, intervention):
        pt_intake = pd.read_csv(os.path.join(path, 'SMHDARDB.PtIntake-DEID.tsv'), sep='\t',error_bad_lines=False)
        pt_intake.drop(['verboseForm', 'baseValueNumber', 'baseUOM', 'upperNormal', 'lowerNormal', 'materialId'], axis=1,inplace=True)
        pt_intake.loc[:,
        ['ptIntakeId', 'interventionId', 'attributeId', 'encounterId',  'valueNumber']] = \
            pt_intake.loc[:,
            ['ptIntakeId', 'interventionId', 'attributeId', 'encounterId', 'valueNumber']]. \
                convert_objects(convert_numeric=True)
        # pt_intake.dropna(subset=['ptIntakeId', 'interventionId', 'attributeId', \
        #                               'encounterId', 'materialId', 'valueNumber'], inplace=True)

        pt_intake.loc[:, ['ptIntakeId', 'interventionId', 'attributeId', 'encounterId']] = \
            pt_intake.loc[:,
            ['ptIntakeId', 'interventionId', 'attributeId', 'encounterId']].astype(int)
        pt_intake.loc[:, ['unitOfMeasure']] = pt_intake.loc[:, 'unitOfMeasure'].astype(str)
        pt_intake.loc[:, ['valueNumber']] = pt_intake.loc[:, ['valueNumber']].astype(float)
        pt_intake.loc[:, 'chartTime'] = pd.to_datetime(pt_intake['chartTime'])
        pt_intake.loc[:, 'utcChartTime'] = pd.to_datetime(pt_intake['utcChartTime'])

        # merging ptIntake with attribute and intervention
        tmp = pd.merge(pt_intake, attribute, how='left', on=['attributeId'])
        tmp = pd.merge(tmp, intervention, how='left', on=['interventionId'])
        tmp.dropna(subset=['valueNumber', 'chartTime'], inplace=True)

        tmp.rename(columns=dict(conceptLabel_x='attrConceptLabel', conceptCode_x='attrConceptCode',
                                conceptLabel_y='intvConceptLabel', conceptCode_y='intvConceptCode',
                                shortLabel_x='attrShortLabel', longLabel_x='attrLongLabel',
                                shortLabel_y='intvShortLabel', longLabel_y='intvLongLabel'), inplace=True)
        return tmp
    ###############################################################################
    def _get_assessment(self, attribute, intervention):
        # process ptAssessment
        pt_assessment = pd.read_csv(os.path.join(path, 'SMHDARDB.PtAssessment-DEID.tsv'), sep='\t',
                                    error_bad_lines=False)
        pt_assessment.drop(['baseValueNumber', 'upperNormal', 'lowerNormal'], axis=1, inplace=True)
        pt_assessment.loc[:, ['ptAssessmentId', 'interventionId', 'attributeId', 'encounterId']] = \
            pt_assessment.loc[:, ['ptAssessmentId', 'interventionId', 'attributeId', 'encounterId']].astype(int)
        pt_assessment.loc[:, ['valueNumber']] = pt_assessment.loc[:, ['valueNumber']].astype(float)
        pt_assessment.loc[:, ['unitOfMeasure']] = pt_assessment.loc[:, 'unitOfMeasure'].astype(str)
        pt_assessment.loc[:, 'chartTime'] = pd.to_datetime(pt_assessment['chartTime'])
        pt_assessment.loc[:, 'utcChartTime'] = pd.to_datetime(pt_assessment['utcChartTime'])

        # merging ptAssessment with attribute and intervention
        tmp = pd.merge(pt_assessment, attribute, how='left', on=['attributeId'])
        tmp = pd.merge(tmp, intervention, how='left', on=['interventionId'])
        tmp.dropna(subset=['valueNumber', 'chartTime'], inplace=True)
        tmp.rename(columns=dict(conceptLabel_x='attrConceptLabel', conceptCode_x='attrConceptCode',
                                conceptLabel_y='intvConceptLabel', conceptCode_y='intvConceptCode',
                                shortLabel_x='attrShortLabel', longLabel_x='attrLongLabel',
                                shortLabel_y='intvShortLabel', longLabel_y='intvLongLabel'), inplace=True)
        return tmp

    ###############################################################################
    def _get_attribute(self):
        # process attribute table
        attribute = pd.read_csv(os.path.join(path, 'SMHDARDB.D_Attribute-DEID.tsv'), sep='\t', \
                                error_bad_lines=False, warn_bad_lines=False)
        attribute.loc[:, 'conceptCode'] = pd.to_numeric(attribute.loc[:, 'conceptCode'], errors='ignore')
        attribute.dropna(subset=['attributeId', 'conceptCode'], inplace=True)
        attribute.loc[:, ['attributeId', 'conceptCode']] = attribute.loc[:, ['attributeId', 'conceptCode']].astype(int)
        return attribute

    ###############################################################################
    def _get_material(self):
        material = pd.read_csv(os.path.join(path, 'SMHDARDB.D_Material.tsv'), sep='\t', \
                               error_bad_lines=False, warn_bad_lines=False)
        material.loc[:, ['materialId', 'conceptCode']] = material.loc[:, ['materialId', 'conceptCode']] \
            .convert_objects(convert_numeric=True)
        material.dropna(subset=['materialId', 'conceptCode'], inplace=True)
        material.loc[:, ['materialId', 'conceptCode']] = material.loc[:, ['materialId', 'conceptCode']].astype(int)
        return material

    ###############################################################################
    def _get_intervention(self):
        intervention = pd.read_csv(os.path.join(path, 'SMHDARDB.D_Intervention-DEID.tsv'), sep='\t', \
                                   error_bad_lines=False, warn_bad_lines=False)
        intervention.loc[:, ['conceptCode']] = intervention.loc[:, ['conceptCode']].convert_objects(
            convert_numeric=True)
        intervention.dropna(subset=['interventionId', 'conceptCode'], inplace=True)
        intervention.loc[:, ['interventionId', 'conceptCode']] = intervention.loc[:,
                                                                 ['interventionId', 'conceptCode']].astype(int)
        return intervention

    ###############################################################################
    def _import_medevents(self, meds):
        datToAdd = list()
        num_added = 0
        meds = meds.where((pd.notnull(meds)), None)
        for ind, row in meds.iterrows():
            newDat = ptd.MedEvents(encounter_id=row.encounterId,
                                   attr_concept_code=row.attrConceptCode,
                                   attr_concept_label=row.attrConceptLabel,
                                   attr_short_label=row.attrShortLabel,
                                   attr_long_label=row.attrLongLabel,
                                   intv_concept_code=row.intvConceptCode,
                                   intv_concept_label=row.intvConceptLabel,
                                   intv_short_label=row.intvShortLabel,
                                   intv_long_label=row.intvLongLabel,
                                   mat_concept_code=row.matConceptCode,
                                   mat_concept_label=row.matConceptLabel,
                                   mat_short_label=row.matShortLabel,
                                   mat_long_label=row.matLongLabel,
                                   value=row.valueNumber,
                                   valueUOM=row.unitOfMeasure,
                                   tstamp=self._standard_datetime(row.chartTime)
                                   )
            datToAdd.append(newDat)

            if len(datToAdd) > 10000:
                self.session.add_all(datToAdd)
                self.session.commit()
                num_added += len(datToAdd)
                print("Added new data, newDat = {}, num_added = {}".format(len(datToAdd), num_added))
                datToAdd = list()

        if len(datToAdd) > 0:
            self.session.add_all(datToAdd)
            self.session.commit()
            num_added += len(datToAdd)

        print("Imported {} measurements".format(num_added))
    ###############################################################################
    def _import_fluidevents(self, fluids):
        datToAdd = list()
        num_added = 0
        fluids = fluids.where((pd.notnull(fluids)), None)
        for ind, row in fluids.iterrows():
            newDat = ptd.FluidEvents(encounter_id=row.encounterId,
                                   attr_concept_code=row.attrConceptCode,
                                   attr_concept_label=row.attrConceptLabel,
                                   attr_short_label=row.attrShortLabel,
                                   attr_long_label=row.attrLongLabel,
                                   intv_concept_code=row.intvConceptCode,
                                   intv_concept_label=row.intvConceptLabel,
                                   intv_short_label=row.intvShortLabel,
                                   intv_long_label=row.intvLongLabel,
                                   value=row.valueNumber,
                                   valueUOM=row.unitOfMeasure,
                                   tstamp=self._standard_datetime(row.chartTime)
                                   )
            datToAdd.append(newDat)

            if len(datToAdd) > 10000:
                self.session.add_all(datToAdd)
                self.session.commit()
                num_added += len(datToAdd)
                print("Added new data, newDat = {}, num_added = {}".format(len(datToAdd), num_added))
                datToAdd = list()

        if len(datToAdd) > 0:
            self.session.add_all(datToAdd)
            self.session.commit()
            num_added += len(datToAdd)
        print("Imported {} measurements".format(num_added))
    ###############################################################################
    def _import_encounter(self, encounter):
        new_encounters = list()

        for ind, row in encounter.iterrows():
            new_enc = ptd.Encounters(encounter_id=row.encounterId,
                                     patient_id=row.patientId,
                                     episode_id=row.episodeId,
                                     age_at_admit=row.age,
                                     gender=row.gender,
                                     adm_tstamp=self._standard_datetime(row.inTime),
                                     discharge_tstamp=self._standard_datetime(row.outTime),
                                     ICU_LOS_min=row.lengthOfStay,
                                     is_24hr_readmit=row.is24HrReAdmit,
                                     is_discharged=row.isDischarged,
                                     is_deceased=row.isDeceased,
                                     is_transferred=row.isTransferred
                                     )
            new_encounters.append(new_enc)
        self.session.add_all(new_encounters)
        self.session.commit()

    ###############################################################################
    def _import_chartevents(self, assessment, labs):

        datToAdd = list()
        num_added = 0

        # merge assessment and labs
        col = ['encounterId', 'attrConceptCode', 'attrConceptLabel', 'attrShortLabel', 'attrLongLabel',
               'intvConceptCode', 'intvConceptLabel', 'intvShortLabel', 'intvLongLabel', 'valueNumber',
               'unitOfMeasure', 'chartTime']
        assessment_lab_merged = pd.concat([assessment.loc[:, col], labs.loc[:, col]])
        assessment_lab_merged = assessment_lab_merged.where((pd.notnull(assessment_lab_merged)), None)

        for ind, row in assessment_lab_merged.iterrows():
            newDat = ptd.ChartEvents(encounter_id=row.encounterId,
                                     attr_concept_code=row.attrConceptCode,
                                     attr_concept_label=row.attrConceptLabel,
                                     attr_short_label=row.attrShortLabel,
                                     attr_long_label=row.attrLongLabel,
                                     intv_concept_code=row.intvConceptCode,
                                     intv_concept_label=row.intvConceptLabel,
                                     intv_short_label=row.intvShortLabel,
                                     intv_long_label=row.intvLongLabel,
                                     value=row.valueNumber,
                                     valueUOM=row.unitOfMeasure,
                                     tstamp=self._standard_datetime(row.chartTime)
                                     )

            datToAdd.append(newDat)

            if len(datToAdd) > 10000:
                self.session.add_all(datToAdd)
                self.session.commit()
                num_added += len(datToAdd)
                print("Added new data, newDat = {}, num_added = {}".format(len(datToAdd), num_added))
                datToAdd = list()

        if len(datToAdd) > 0:
            self.session.add_all(datToAdd)
            self.session.commit()
            num_added += len(datToAdd)
        print("Imported {} measurements".format(num_added))


###############################################################################
###############################################################################
class queryDB:
    def __init__(self):
        self.engine, self.session = dbinit.make_connection(connection_str, db_echo=False)

    ###############################################################################
    def _notempty(self, df):
        if df.empty:
            flag = 0
        else:
            flag = 1
        return flag

    ###############################################################################
    def getEncounterData(self, encounter_ids=None):
        results = self.session.query(ptd.Encounters)

        if encounter_ids is not None and len(encounter_ids) > 0:
            results = results.filter(ptd.Encounters.encounter_id.in_(encounter_ids))

        # encounters = pd.DataFrame.from_records([{'patient_id': row.patient_id, 'encounter_id': row.encounter_id,
        #                                          'episode_id': row.episode_id, 'age_at_admit': row.age_at_admit,
        #                                          'gender': row.gender, 'adm_tstamp': row.adm_tstamp,
        #                                          'discharge_tstamp': row.discharge_tstamp,
        #                                          'ICU_LOS_min': row.ICU_LOS_min,
        #                                          'is_24hr_readmit': row.is_24hr_readmit,
        #                                          'is_discharged': row.is_discharged,
        #                                          'is_deceased': row.is_deceased,
        #                                          'is_transferred': row.is_transferred
        #                                          } for row in results],
        #                                        columns=['patient_id', 'encounter_id', 'age_at_admit', 'gender',
        #                                                 'adm_tstamp',
        #                                                 'discharge_tstamp', 'ICU_LOS_min', 'is_24hr_readmit',
        #                                                 'is_discharged', 'is_deceased', 'is_transferred'])
        columns = ['patient_id', 'encounter_id', 'age_at_admit', 'gender',
                   'adm_tstamp', 'discharge_tstamp', 'ICU_LOS_min', 'is_24hr_readmit',
                   'is_discharged', 'is_deceased', 'is_transferred']
        encounters = pd.read_sql(results.statement, results.session.bind)
        encounters = encounters.loc[:, columns]

        return encounters

    ###############################################################################
    def getFeatureData(self, encounter_ids=None, feature_ids=None, tstart=None, tstop=None):

        df_tmp = list()
        columns=['encounter_id', 'attr_concept_code','attr_concept_label', 'attr_short_label','attr_long_label',
                 'intv_concept_code','intv_concept_label', 'intv_short_label','intv_long_label', 'value', 'valueUOM',
                 'tstamp']
        results = self.session.query(ptd.ChartEvents)

        if tstart is not None:
            results = results.filter(ptd.ChartEvents.tstamp > tstart)
        if tstop is not None:
            results = results.filter(ptd.ChartEvents.tstamp < tstop)
        if feature_ids is not None and len(feature_ids) > 0:
            results = results.filter((ptd.ChartEvents.attr_concept_code.in_(feature_ids['attr_concept_code'])) &
                                     (ptd.ChartEvents.intv_concept_code.in_(feature_ids['intv_concept_code'])))
        item_df = pd.read_sql(results.statement, results.session.bind)
        # item_df['encounter_id'] = item_df['encounter_id'].astype('int')
        # item_df['patient_id'] = item_df['patient_id'].astype('int')

        if encounter_ids is not None:
            # item_df.groupby('encounter_id').apply(printgroup, encounter_ids)

            item_df =  item_df.groupby('encounter_id').filter(lambda x: np.in1d(x['encounter_id'].unique(), encounter_ids)[0])
            # item_df = item_df.loc[np.in1d(item_df.encounter_id, encounter_ids),:]

        return item_df
        # # The query is limited to 2000 by SQL server. Therefore, I have to
        # # iterate every 2000
        # count = 0
        # k = 500
        # if encounter_ids is not None:
        #     while len(encounter_ids) > count:
        #         results_cnt = results.filter(ptd.ChartEvents.encounter_id.in_(encounter_ids[count:count+k]))
        #         for row in results_cnt:
        #             df_tmp.append([row.encounter_id,row.attr_concept_code,row.attr_concept_label,row.attr_short_label,
        #                            row.attr_long_label,row.intv_concept_code,row.intv_concept_label,row.intv_short_label,
        #                            row.intv_long_label,row.value,row.valueUOM,row.tstamp])
        #         # print count
        #         count = count + k

        # if df_tmp:
            # return pd.DataFrame(df_tmp,columns=columns)
        # else:
        #     return pd.DataFrame(columns=columns)

    ###############################################################################
    def getMeds(self, encounter_ids=None, med_ids=None, tstart=None, tstop=None):
        df_tmp = list()
        columns = ['encounter_id', 'attr_concept_code','attr_concept_label', 'attr_short_label','attr_long_label',
                   'intv_concept_code','intv_concept_label', 'intv_short_label','intv_long_label', 'mat_concept_code',
                   'mat_concept_label', 'mat_short_label','mat_long_label', 'value', 'valueUOM', 'tstamp']
        results = self.session.query(ptd.MedEvents)

        if tstart is not None:
            results = results.filter(ptd.MedEvents.tstamp > tstart)
        if tstop is not None:
            results = results.filter(ptd.MedEvents.tstamp < tstop)
        if med_ids is not None and len(med_ids) > 0:
            results = results.filter((ptd.MedEvents.attr_concept_code.in_(med_ids['attr_concept_code'])) &
                                     (ptd.MedEvents.mat_concept_code.in_(med_ids['mat_concept_code'])))
        # The query is limited to 2000 by SQL server. Therefore, I have to
        # iterate every 2000
        count = 0
        k = 500
        if encounter_ids is not None:
            while len(encounter_ids) > count:
                results_cnt = results.filter(ptd.MedEvents.encounter_id.in_(encounter_ids[count:count+k]))
                for row in results_cnt:
                    df_tmp.append([row.encounter_id,row.attr_concept_code,row.attr_concept_label,row.attr_short_label,
                                   row.attr_long_label,row.intv_concept_code,row.intv_concept_label,
                                   row.intv_short_label,row.intv_long_label,row.mat_concept_code,row.mat_concept_label,
                                   row.mat_short_label,row.mat_long_label,row.value,row.valueUOM,row.tstamp])
                count = count + k

        if df_tmp:
            return pd.DataFrame(df_tmp,columns=columns)
        else:
            return pd.DataFrame(columns=columns)
    ###############################################################################
    def getFluids(self, encounter_ids=None, fluid_ids=None, tstart=None, tstop=None):
        df_tmp = list()
        columns = ['encounter_id', 'attr_concept_code','attr_concept_label', 'attr_short_label','attr_long_label',
                   'intv_concept_code','intv_concept_label', 'intv_short_label','intv_long_label', 'value',
                   'valueUOM', 'tstamp']
        results = self.session.query(ptd.FluidEvents)

        if tstart is not None:
            results = results.filter(ptd.FluidEvents.tstamp > tstart)
        if tstop is not None:
            results = results.filter(ptd.FluidEvents.tstamp < tstop)
        if fluid_ids is not None and len(fluid_ids) > 0:
            results = results.filter((ptd.FluidEvents.attr_concept_code.in_(fluid_ids['attr_concept_code'])) &
                                     (ptd.FluidEvents.intv_concept_code.in_(fluid_ids['intv_concept_code'])))
        # The query is limited to 2000 by SQL server. Therefore, I have to
        # iterate every 2000
        count = 0
        k = 500
        if encounter_ids is not None:
            while len(encounter_ids) > count:
                results_cnt = results.filter(ptd.FluidEvents.encounter_id.in_(encounter_ids[count:count+k]))
                for row in results_cnt:
                    df_tmp.append([row.encounter_id,row.attr_concept_code,row.attr_concept_label,row.attr_short_label,
                                   row.attr_long_label,row.intv_concept_code,row.intv_concept_label,
                                   row.intv_short_label,row.intv_long_label,row.value,row.valueUOM,row.tstamp])
                count = count + k

        if df_tmp:
            return pd.DataFrame(df_tmp,columns=columns)
        else:
            return pd.DataFrame(columns=columns)
    ###############################################################################
    def getOnsetEvent(self, df, weight, typ, am=None):
        """
        This function returns a datetime with the onset of a fluid
        intervention.

        Input:
        1. df: data frame containing fluids
        2. am: threshold to determine a fluid intervention. am can be
           10 ml/kg/hr,20, 40, or 65, and threshold is determined as
           thr > am*weight

        Output:
        1. datetime of onset
        """
        ## identify onset of fluid intervention
        if typ == 1:  # fluids
            tmax = 1  # hr
        if typ == 2:  # blood products
            tmax = 24  # hr

        if am is None:
            am = 10

        tstart = 0
        tstop = 1
        N = len(df)
        onset = []

        df.sort_values(by='tstamp', inplace=True)
        df.loc[:, 'value'] = df.value.astype(float)
        # pdb.set_trace()
        while ((tstart < N) & (tstop < N)):
            # print(tstart)
            timeElap = (df.tstamp.iloc[tstop] - df.tstamp.iloc[tstart]). \
                           total_seconds() / 3600.0  # in hours

            vol = df.value.iloc[tstart:tstop].sum()
            # rate = vol/timeElap
            # find closest weight to fluids measurement
            ts = df.tstamp.iloc[tstart]
            val = np.inf

            thr = weight * am
            # pdb.set_trace()
            if (timeElap > 0) & (timeElap <= tmax) & (vol > thr):
                # this is an event
                onset = df.tstamp.iloc[tstart]
                break
            # for cases where time ellapsed > tmax
            if (timeElap > tmax) & (df.value.iloc[tstart] > thr):
                onset = df.tstamp.iloc[tstart]
                break

            elif (timeElap >= tmax):
                # reset count
                tstart = tstop

            tstop += 1

        return onset

    ###############################################################################

    def getFeaturesEvent(self, encounter_ids, census_df, weight_df, listMedId, listFluidsId, listFeatId, dt, cols):

        ftarr = [np.nan] * len(cols)
        isData = True
        # get first time of admission and last time of discharge
        t_admit = census_df.registration_tstamp.iloc[0]
        t_disch = census_df.discharge_tstamp.iloc[-1]
        LOSm = census_df.LOSm.iloc[0]

        # get info about fluids and meds given
        meds_df = self.getMedData(encounter_ids, med_ids=listMedId)
        if not meds_df.empty:
            meds_df.dropna(subset=['dose'], inplace=True)
            meds_df.sort(columns='tstamp', inplace=True)
            onsetmedsevent = meds_df.tstamp.iloc[0]
        else:
            onsetmedsevent = []

        fluids_df = self.getFluidsData(encounter_ids, fluids_ids=listFluidsId)
        if not fluids_df.empty:
            fluids_df.dropna(subset=['volume'], inplace=True)
            fluids_df = fluids_df[fluids_df['volume'] != 0]
            onsetfluidsevent = self.getOnsetEvent(fluids_df, weight_df, typ=1)
        else:
            onsetfluidsevent = []

        # find the time of first clinical intervention.
        ftarr[cols.index('fMeds')] = 0
        ftarr[cols.index('fFluids')] = 0

        # get onset of first intervention
        if onsetmedsevent and onsetfluidsevent:
            if onsetmedsevent > onsetfluidsevent:
                ftarr[cols.index('fMeds')] = 1
                onset = onsetmedsevent
            else:
                ftarr[cols.index('fFluids')] = 1
                onset = onsetfluidsevent

        if onsetmedsevent and not onsetfluidsevent:
            ftarr[cols.index('fMeds')] = 1
            onset = onsetmedsevent

        if not onsetmedsevent and onsetfluidsevent:
            ftarr[cols.index('fFluids')] = 1
            onset = onsetfluidsevent

        if not onsetmedsevent and not onsetfluidsevent:
            onset = []

        # if there is onset then patient is unstable
        if onset:
            # unstable patient
            ftarr[cols.index('label')] = 1
            t_int = onset - dt

        else:
            # stable patient
            ftarr[cols.index('label')] = 0
            # pick a random time interval since there is no intervention
            int_delta = (t_disch - t_admit).total_seconds()
            random_second = random.randrange(int_delta)
            t_int = t_admit + datetime.timedelta(seconds=random_second)

        # get features
        feat_df = self.getFeatureData(encounter_ids, feature_ids=listFeatId.values(),
                                      tstart=t_int - datetime.timedelta(hours=24), tstop=t_int)

        # drop nan values
        feat_df.dropna(subset=['value'], inplace=True)

        isData = True
        # check there is HR and nSB or iSBP
        if not feat_df.empty:
            tmp_df = feat_df[feat_df.event_id.isin([listFeatId['HR'], listFeatId['nSBP'], listFeatId['iSBP']])]
            if tmp_df.empty:
                isData = False
            elif (tmp_df[tmp_df.event_id == listFeatId['HR']].empty) | (
                        (tmp_df[tmp_df.event_id == listFeatId['nSBP']].empty) &
                        (tmp_df[tmp_df.event_id == listFeatId['iSBP']].empty)):
                isData = False
        else:
            isData = False

        if isData:
            # get list of all available features
            listavfeat = feat_df.event_id.unique()

            ################ compute derived features ####################

            # get iSI
            if (listFeatId['iSBP'] in listavfeat) and (listFeatId['HR'] in listavfeat):

                HR = feat_df[feat_df['event_id'] == listFeatId['HR']].sort(columns='tstamp')
                BP = feat_df[feat_df['event_id'] == listFeatId['iSBP']].sort(columns='tstamp')
                # if time difference between HR and BP measurement is
                # < 1 hr, compute SI
                if (HR.tstamp.tolist()[-1] - BP.tstamp.tolist()[-1]).total_seconds() / 3600. <= 1:
                    HR = HR.value.values[-1]
                    BP = BP.value.values[-1]
                    # make sure no to divide by zero
                    if not np.isnan(HR) and not np.isnan(BP):
                        if np.any(BP):
                            ftarr[cols.index('iSI')] = HR / BP

            # get nSI
            if (listFeatId['nSBP'] in listavfeat) and (listFeatId['HR'] in listavfeat):

                HR = feat_df[feat_df['event_id'] == listFeatId['HR']].sort(columns='tstamp')
                BP = feat_df[feat_df['event_id'] == listFeatId['nSBP']].sort(columns='tstamp')
                # if time difference between HR and BP measurement is
                # < 1 hr, compute SI
                if (HR.tstamp.tolist()[-1] - BP.tstamp.tolist()[-1]).total_seconds() / 3600. <= 1:
                    HR = HR.value.values[-1]
                    BP = BP.value.values[-1]
                    # make sure to no divide by zero
                    if not np.isnan(HR) and not np.isnan(BP):
                        if np.any(BP):
                            ftarr[cols.index('nSI')] = HR / BP

            ############# iterate over available features #################
            for nameFeat, codeFeat in listFeatId.items():
                tmp = feat_df[feat_df['event_id'] == codeFeat]
                tmp = tmp[tmp.value.values != 0]

                if (not tmp.empty) & (nameFeat in cols):
                    tmp.sort(columns='tstamp', inplace=True)
                    ftarr[cols.index(nameFeat)] = tmp.value.values[-1]

            ftarr[cols.index('eId')] = census_df.encounter_id.iloc[0]
            ftarr[cols.index('ptId')] = census_df.patient_id.iloc[0]
            ftarr[cols.index('age')] = census_df.age.mean()
            ftarr[cols.index('LOS')] = LOSm

            if census_df.gender.iloc[0] == 'Male':
                ftarr[cols.index('sex')] = 1
            if census_df.gender.iloc[0] == 'Female':
                ftarr[cols.index('sex')] = 2

        return isData, ftarr

    ###############################################################################

    def getHistogramFeatures(self, encounter_ids, feat_ids):

        hist_feat = pd.DataFrame(columns=feat_ids)
        feat_df = self.getFeatureData(encounter_ids, feature_ids=feat_ids.values())
        feat_df.dropna(subset=['value'], inplace=True)  # remove nan values_
        feat_df = feat_df[feat_df.value.values != 0]  # remove zero values

        for nameFeat, codeFeat in feat_ids.items():
            tmp = feat_df[feat_df['event_id'] == codeFeat]

            # To-do: check unique UOM
            # tmp = tmp.iloc[np.where(tmp.valueUOM!='None')] # remove samples where the UOM is None
            if self._notempty(tmp) == 1:
                # if len(tmp.valueUOM.unique())>1: # if more than one UOM
                #     UOMfound = tmp.valueUOM.unique()
                #     raise Warning("There are more than one units of measurements!")

                hist_feat[nameFeat] = tmp.value.values[np.random.randint(0, len(tmp), size=10)]

        return hist_feat


    def getScrDF(self, pre_scr_df=None, ex_age=False, ex_los=False,
                 ex_aki_adm=False, aki_adm_hr=12, ex_laki=False, aki_late_hr=72,
                 ex_noaki=False, enc_per_pat=False):

        if pre_scr_df is not None:
            scr_df_large = pre_scr_df
            mask_age = (scr_df_large.age>1/12.*365) & (scr_df_large.age<21.*365)
            mask_los = scr_df_large.ICU_LOS_min > 24*60.
            scr_df_large = scr_df_large.loc[mask_age & mask_los,:]
        else:
            encounter_census = self.getEncounterData()

            mask_finite = (encounter_census.gender != 'NaN') & (~pd.isnull(encounter_census.age_at_admit)) \
                          & (~pd.isnull(encounter_census.adm_tstamp))
            mask = mask_finite

            if ex_age:
                mask_age = (encounter_census.age_at_admit >= 1 / 12.) & (encounter_census.age_at_admit <= 21.)
                mask = mask & mask_age
            if ex_los:
                mask_los = encounter_census.ICU_LOS_min >= 24 * 60.
                mask = mask & mask_los

            encounter_census = encounter_census.loc[mask, :]
            if enc_per_pat:
                encounter_census = filterEncPerPat(encounter_census)
            filtered_ceids = encounter_census.encounter_id.tolist()
            scr_ids = feature_dict['creatinine']
            scr_df = self.getFeatureData(encounter_ids=filtered_ceids, feature_ids=scr_ids)
            # scr_df = self.getFeatureData(feature_ids=scr_ids)
            scr_df = scr_df.loc[~pd.isnull(scr_df.value), :]

            scr_df_large = scr_df.merge(encounter_census, how='inner', on='encounter_id')

        # convert umol/l to mg/dl
        if scr_df_large.valueUOM.unique()[0] != 'mg/dl':
            scr_df_large.loc[:, 'value'] = scr_df.loc[:, 'value'] * 0.01131
            scr_df_large.loc[:, 'valueUOM'] = 'mg/dl'

        # change age in days
        # change column name
        if np.sum(np.in1d(scr_df_large.columns, 'age')) == 0:
            scr_df_large.loc[:, 'age_at_admit'] = scr_df_large.loc[:, 'age_at_admit'] * 365
            scr_df_large.loc[scr_df_large.gender=='1.0', 'gender'] = 'M'
            scr_df_large.loc[scr_df_large.gender=='2.0', 'gender'] = 'F'

            scr_df_large.rename(columns={'tstamp': 'charttime',
                                         'age_at_admit': 'age',
                                         'adm_tstamp': 'intime',
                                         'discharge_tstamp': 'outtime',
                                         'gender': 'sex'}, inplace=True)
        if 'dtime_hr' not in scr_df_large.columns:
            scr_df_large = scr_df_large.groupby('encounter_id').apply(tagDtimeHr)
        if 'bs_scr' not in scr_df_large.columns:
            scr_df_large = scr_df_large.groupby('encounter_id').apply(tagBSCr)
        if 'AKI_stage' not in scr_df_large.columns:
            scr_df_large = scr_df_large.groupby('encounter_id').apply(tagAKI)
        if 'reftime' not in scr_df_large.columns:
            scr_df_large = scr_df_large.groupby('encounter_id').apply(fillReftime2, aki_adm_hr)

        if ex_aki_adm:
            scr_df_large = scr_df_large.groupby('encounter_id'). \
                filter(lambda _group: _group.loc[_group.dtime_hr <= aki_adm_hr, 'AKI_stage'].sum() == 0)
        if ex_laki:
            print('Excluding patients who developed AKI after {} hours ... '.format(int(aki_late_hr)))
            scr_df_large = scr_df_large.groupby('encounter_id'). \
                filter(lambda _group: _group.loc[_group.dtime_hr <= aki_late_hr, 'AKI_stage'].sum() > 0)
        if ex_noaki:
            print('Excluding patients who never developed AKI ... ')
            scr_df_large = scr_df_large.groupby('encounter_id'). \
                filter(lambda _group: _group.loc[:, 'AKI_stage'].sum() > 0)

        return scr_df_large

    def getIOComposite(self, scr_df, io_df, labels, feature_stats, timelag, timewin=6):
        encounter_ids = io_df.encounter_id.unique()

        try:
            hr_df = pd.read_pickle(os.path.join(fileDir, 'item_df_stm', 'stm_hr_df.pkl'))
            hr_df = hr_df.loc[np.in1d(hr_df.encounter_id, encounter_ids),:]
        except:
            hr_df = self.getFeatureData(encounter_ids=encounter_ids,
                                        feature_ids={'attr_concept_code': [364075005],
                                                     'intv_concept_code': [364075005]})
        print('hr dataframe queried ..')

        try:
            nsbp_df = pd.read_pickle(os.path.join(fileDir, 'item_df_stm', 'stm_nsbp_df.pkl'))
            nsbp_df = nsbp_df.loc[np.in1d(nsbp_df.encounter_id, encounter_ids), :]
        except:
            nsbp_df = self.getFeatureData(encounter_ids=encounter_ids,
                                          feature_ids={'attr_concept_code': [271649006],
                                                       'intv_concept_code': [17146006]})
        print('nsbp dataframe queried ..')

        try:
            map_df = pd.read_pickle(os.path.join(fileDir, 'item_df_stm', 'stm_nsbp_df.pkl'))
            map_df = map_df.loc[np.in1d(map_df.encounter_id, encounter_ids), :]
        except:
            map_df = self.getFeatureData(encounter_ids=encounter_ids,
                                         feature_ids={'attr_concept_code': [259010008],
                                                      'intv_concept_code': [284019003]})
        print('map dataframe queried ..')

        try:
            fio2_df = pd.read_pickle(os.path.join(fileDir, 'item_df_stm', 'stm_fio2_df.pkl'))
            fio2_df = fio2_df.loc[np.in1d(fio2_df.encounter_id, encounter_ids), :]
        except:
            fio2_df = self.getFeatureData(encounter_ids=encounter_ids,
                                          feature_ids={'attr_concept_code': [250774007],
                                                       'intv_concept_code': [250774007]})
        print('fio2 dataframe queried ..')

        try:
            spo2_df = pd.read_pickle(os.path.join(fileDir, 'item_df_stm', 'stm_spo2_df.pkl'))
            spo2_df = spo2_df.loc[np.in1d(spo2_df.encounter_id, encounter_ids),:]
        except:
            spo2_df = self.getFeatureData(encounter_ids=encounter_ids,
                                          feature_ids={'attr_concept_code': [250554003],
                                                       'intv_concept_code': [250554003]})
        print('spo2 dataframe queried ..')

        try:
            pao2_df = pd.read_pickle(os.path.join(fileDir, 'item_df_stm', 'stm_pao2_df.pkl'))
            pao2_df = pao2_df.loc[np.in1d(pao2_df.encounter_id, encounter_ids), :]
        except:
            pao2_df = self.getFeatureData(encounter_ids=encounter_ids,
                                          feature_ids={'attr_concept_code': [250546000],
                                                       'intv_concept_code': [250546000]})
        print('pao2 dataframe queried ..')



        for item_df in [hr_df, nsbp_df, map_df, fio2_df, spo2_df, pao2_df]:
            # item_df = item_df.merge(scr_df.loc[:, ['encounter_id', 'reftime']], on='encounter_id', how='left')
            enc_reft = scr_df.groupby('encounter_id')['reftime'].unique().to_frame()
            enc_reft = enc_reft.reset_index()
            enc_reft['reftime'] = np.hstack(enc_reft.reftime)

            #     item_df = item_df.merge(scr_df.loc[:, ['encounter_id', 'reftime']], on='encounter_id', how='inner')
            item_df = item_df.merge(enc_reft, on='encounter_id', how='inner')

            item_df.rename(columns={'tstamp': 'charttime'})
            if timelag > 0:
                item_df['fromtime'] = item_df.reftime
                item_df['totime'] = item_df.reftime + np.timedelta64(int(timelag * 60), 'm')
                # time_mask = (_df.charttime > _df.reftime) & (_df.charttime < _df.totime)
            else:
                item_df['fromtime'] = item_df.reftime + np.timedelta64(int(timelag * 60), 'm')
                item_df['totime'] = item_df.reftime + np.timedelta64(int((timelag + timewin) * 60), 'm')
            time_mask = (item_df.charttime > item_df.fromtime) & (item_df.charttime < item_df.totime)
            item_df = item_df.loc[time_mask, :]


        for label in labels:
            if label=='si':
                si_df = hr_df
                for idx, row in si_df.iterrows():
                    fromtime = row.charttime - np.timedelta64(30, 'm')
                    totime = row.charttime + np.timedelta64(30, 'm')
                    row.charttime
            # elif label=='oi':
            #
            # elif label=='osi':
            #
            # else:
            #     item_df = self.getFeatureData(encounter_ids=encounter_ids,
            #                                   feature_ids=feature_ids[feature])





    def getIOMatrix(self, scr_df, feature_ids, feature_stats, timelag, timewin=6):

        encounter_ids = scr_df.encounter_id.unique()
        encounter_census = self.getEncounterData()

        mask_finite = (encounter_census.gender != 'NaN') & (~pd.isnull(encounter_census.age_at_admit)) \
                      & (~pd.isnull(encounter_census.adm_tstamp))
        encounter_census = encounter_census.loc[mask_finite, :]
        encounter_census = encounter_census.loc[np.in1d(encounter_census.encounter_id, encounter_ids)]
        encounter_census = filterEncPerPat(encounter_census)

        default_entry = ['encounter_id', 'age', 'sex']

        glob_mat = pd.DataFrame()
        for entry in default_entry:
            series = scr_df.groupby('patient_id')[entry].unique()
            series.name = entry
            df = pd.DataFrame(series)
            df.reset_index(inplace=True)
            if glob_mat.empty:
                glob_mat = df
            else:
                glob_mat = glob_mat.merge(df, on='patient_id')
            glob_mat[entry] = np.hstack(glob_mat[entry])

        out_col = pd.DataFrame(scr_df.groupby('patient_id')['AKI_stage'].max())
        out_col.reset_index(inplace=True)
        out_col.columns = ['patient_id', 'AKI_stage']
        glob_mat = glob_mat.merge(out_col, on="patient_id", how="left")


        for feature in feature_ids:
            print("{}: {}".format(feature, feature_ids[feature]))
            f_df = os.path.join(fileDir, 'item_df_stm', 'stm_{}_df.pkl'.format(feature))
            if os.path.exists(f_df):
                item_df = pd.read_pickle(f_df)
                item_df = item_df.loc[np.in1d(item_df.encounter_id, encounter_ids), :]
            else:
                item_df = self.getFeatureData(encounter_ids=encounter_ids,
                                              feature_ids=feature_ids[feature])

            if 'age' not in item_df.columns:
                item_df = item_df.merge(encounter_census, how='inner', on='encounter_id')
                item_df.loc[:, 'age_at_admit'] = item_df.loc[:, 'age_at_admit'] * 365
                item_df.loc[item_df.gender == '1.0', 'gender'] = 'M'
                item_df.loc[item_df.gender == '2.0', 'gender'] = 'F'
                item_df.rename(columns={'tstamp': 'charttime',
                                        'age_at_admit': 'age',
                                        'adm_tstamp': 'intime',
                                        'discharge_tstamp': 'outtime',
                                        'gender': 'sex'}, inplace=True)

                item_df.sort_values(by=['encounter_id', 'charttime'], inplace=True)
                enc_reft = scr_df.groupby('encounter_id')['reftime'].unique().to_frame()
                enc_reft = enc_reft.reset_index()
                enc_reft['reftime'] = np.hstack(enc_reft.reftime)


                #     item_df = item_df.merge(scr_df.loc[:, ['encounter_id', 'reftime']], on='encounter_id', how='inner')
                item_df = item_df.merge(enc_reft, on='encounter_id', how='inner')

            # item_df = item_df.merge(scr_df.loc[:, ['encounter_id', 'reftime']], on='encounter_id', how='left')

            # if ref_mode=='onset':
            #     item_df = item_df.groupby('encounter_id').apply(fillReftime, ref_mode=ref_mode, scr_df=scr_df)
            # elif ref_mode=='random':
            #     item_df = item_df.groupby('encounter_id').apply(fillReftime, ref_mode=ref_mode, lag=timelag)

            if timelag > 0:
                item_df['fromtime'] = item_df.reftime
                item_df['totime'] = item_df.reftime + np.timedelta64(int(timelag * 60), 'm')
                # time_mask = (_df.charttime > _df.reftime) & (_df.charttime < _df.totime)
            else:
                if 'reftime' not in item_df.columns:
                    stop
                item_df['fromtime'] = item_df.reftime + np.timedelta64(int(timelag * 60), 'm')
                item_df['totime'] = item_df.reftime + np.timedelta64(int((timelag + timewin) * 60), 'm')
                # time_mask = (_df.charttime < _df.reftime) & (_df.charttime > _df.totime)
            time_mask = (item_df.charttime > item_df.fromtime) & (item_df.charttime < item_df.totime)
            item_df = item_df.loc[time_mask, :]
            item_df_grouped = item_df.groupby('patient_id')['value']

            for stats in feature_stats[feature]:
                col_inmat = pd.Series()
                label_full = feature + "_" + stats
                # print(label_full)
                if stats == 'mean':
                    col_inmat = item_df_grouped.mean()
                elif stats == 'median':
                    col_inmat = item_df_grouped.median()
                elif stats == 'max':
                    col_inmat = item_df_grouped.max()
                elif stats == 'min':
                    col_inmat = item_df_grouped.min()
                elif stats == 'unique':
                    col_inmat = np.hstack(item_df_grouped.unique())
                elif stats == 'last':
                    col_inmat = item_df_grouped.apply(getLastVal)

                col_inmat = pd.DataFrame(col_inmat)
                col_inmat.reset_index(inplace=True)
                col_inmat.columns = ['patient_id', label_full]

                glob_mat = glob_mat.merge(col_inmat, on='patient_id', how="left")

        return glob_mat







########################################################################################################################
# Static functions
def filterEncPerPat(_df):
    """
    Function that maintains one encounter per patient with the longest los
    :param _df: Dataframe with multiple encounters per patient
    :return: dataframe with one encounter per patient
    """
    # initialize the return dataframe
    df = _df.copy(deep=True)

    to_drop_idx = np.array([])
    for patient in _df.patient_id.unique():
        pat_df = _df.loc[_df.patient_id == patient, :]
        los = pat_df.ICU_LOS_min
        intime = pat_df.adm_tstamp
        max_los = los.max()
        min_intime = intime.min()

        if len(pat_df.index[pat_df.ICU_LOS_min == max_los]) == 1:
            to_drop_idx = np.concatenate((to_drop_idx,
                                          pat_df.index[los < max(los)]),
                                         axis=0)
        elif len(pat_df.index[pat_df.adm_tstamp == min_intime]) == 1:
            to_drop_idx = np.concatenate((to_drop_idx,
                                          pat_df.index[intime != min_intime]),
                                         axis=0)

    df.drop(to_drop_idx, inplace=True)
    return df


def tagBSCr(_group):
    age_yr = _group.age.unique()[0] / 365.
    age_mask = (age_yr_ll < age_yr) & (age_yr_ul >= age_yr)

    sex = _group.sex.unique()[0]
    sex_arr = norm_scr_lim.sex.as_matrix().astype(str)
    sex_mask = [sex in normsex for normsex in sex_arr]
    _group['bs_scr'] = norm_scr_lim.upp_scr[age_mask & sex_mask].unique()[0]

    return _group

def tagAKI(_group):
    # print("group name: {}".format(_group.name))
    dtime_hr = (_group.loc[:, 'charttime'] \
        - _group.loc[_group.index[0], 'intime']).astype('timedelta64[m]') / 60

    for idx, row in _group.iterrows():

        # Calculate average rate of change of SCr within 48 hour time window
        # charttime -48h is used as the time window
        # time_mask = (dtime_hr > dtime_hr[idx] - 30) \
        #             & (dtime_hr <= dtime_hr[idx] + 30)
        scr_rate = np.NaN
        _group.loc[idx, 'scr_rate'] = scr_rate
        if row.dtime_hr>=48:
            time_mask = (dtime_hr >= row.dtime_hr - 48) & (dtime_hr <= row.dtime_hr)
            y = np.array(_group.value[time_mask])
            x = np.array(_group.dtime_hr[time_mask])
            A = np.vstack([x, np.ones(len(x))]).T
            if A.shape[0]>1:
                slope, intercept = np.linalg.lstsq(A,y)[0]
                scr_rate = slope*48
                _group.loc[idx, 'scr_rate'] = scr_rate
        cur_val = row.value

        # df_window = _group.loc[time_mask, :]
        # dtime_hr_window = dtime_hr[time_mask]
        # delta_t = dtime_hr_window[dtime_hr_window.index[-1]] - dtime_hr_window[dtime_hr_window.index[0]]
        # value = np.empty(2)
        # value[0] = df_window.value[df_window.index[0]]
        # value[1] = df_window.value[df_window.index[-1]]
        # cur_val = _group.value[idx]
        # scr_rate = np.divide(value[-1] - value[0], delta_t) * 48
        # _group.loc[idx, 'scr_rate'] = scr_rate

        if (cur_val >= 3.0 * _group.bs_scr[idx]) or (cur_val >= 4.0):
            AKI_stage = 3
        elif cur_val >= 2.0 * _group.bs_scr[idx]:
            AKI_stage = 2
        elif (cur_val >= 1.5 * _group.bs_scr[idx]) or (scr_rate >= 0.3):
            AKI_stage = 1
        else:
            AKI_stage = 0
        _group.loc[idx, 'AKI_stage'] = AKI_stage
    print(_group.name)
    return _group

def tagDtimeHr(_group):
    dtime_hr = (_group.loc[:, 'charttime'] \
                - _group.loc[_group.index[0], 'intime']).astype('timedelta64[m]') / 60
    _group['dtime_hr'] = dtime_hr

    return _group

def printgroup(_group, encounter_ids):
    print(_group.name)
    print(np.in1d(_group.encounter_id.unique(), encounter_ids))

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

    elif ref_mode == 'admit':
        _group['reftime'] = _group.intime
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

def uomConvert(from_tag, to_tag, from_df):
    f = open(os.path.join(fileDir, "pickle_files_stm", "stm2ism_uomconv.pkl"), 'r')
    stm2ism_uomconv = pickle.load(f)

    suffices = ['min', 'max', 'mean', 'median', 'last']
    try:
        stm2usm_cfact = {feature: float(stm2ism_uomconv[feature]['conversion']) for feature in stm2ism_uomconv}

        for feature in stm2usm_cfact:
            if (from_tag, to_tag) == ('stm', 'ism'):
                cfact = stm2usm_cfact[feature]
            elif (from_tag, to_tag) == ('ism', 'stm'):
                cfact = 1./stm2usm_cfact[feature]
            if cfact != 1:
                for suffix in suffices:
                    full_feature = feature+"_"+suffix
                    try:
                        from_df.loc[:, full_feature] = from_df.loc[:, full_feature] * cfact
                    except:
                        pass
        return from_df
    except ValueError:
        print('invalid arguments...')


