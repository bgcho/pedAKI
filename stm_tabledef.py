# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 11:35:24 2015

@author: 310153046
"""

from sqlalchemy import ForeignKey, Table
from sqlalchemy import Column, Date, DateTime, Integer, String, Float, Boolean, PickleType, Enum, Unicode
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref
from sqlalchemy.orm.exc import MultipleResultsFound, NoResultFound
from sqlalchemy.ext.associationproxy import association_proxy

Base = declarative_base()

########################################################################
# class Encounters(Base):
#     """"""
#     __tablename__       = "Encounters"
#     id                  = Column(Integer, primary_key=True)
#     encounter_id        = Column(String(200))
#     patient_id          = Column(String(200))
#     episode_id          = Column(String(200))
#     age_at_admit        = Column(Float)
#     gender              = Column(String(200))
#     adm_tstamp          = Column(DateTime)
#     discharge_tstamp    = Column(DateTime)
#     ICU_LOS_min         = Column(Float)
#     is_24hr_readmit     = Column(Boolean)
#     is_discharged       = Column(Boolean)
#     is_deceased         = Column(Boolean)
#     is_transferred      = Column(Boolean)
#
#     def __repr__(self):
#         return 'Encounters(patient internal_id={},{},{} {})'.format(
#             self.patient_id, self.encounter_id, self.age_at_admit,
#             self.gender, self.adm_tstamp)

class Encounters(Base):
    """"""
    __tablename__       = "Encounters"
    id                  = Column(Integer, primary_key=True)
    encounter_id        = Column(Integer,index=True)
    patient_id          = Column(Integer)
    episode_id          = Column(Integer)
    age_at_admit        = Column(Float)
    gender              = Column(String(200))
    adm_tstamp          = Column(DateTime)
    discharge_tstamp    = Column(DateTime)
    ICU_LOS_min         = Column(Float)
    is_24hr_readmit     = Column(Boolean)
    is_discharged       = Column(Boolean)
    is_deceased         = Column(Boolean)
    is_transferred      = Column(Boolean)

    def __repr__(self):
        return 'Encounters(patient internal_id={},{},{} {})'.format(
            self.patient_id, self.encounter_id, self.age_at_admit,
            self.gender, self.adm_tstamp)
########################################################################
class ChartEvents(Base):
   """"""
   __tablename__       = "ChartEvents"
   id                  = Column(Integer,primary_key=True)
   encounter_id        = Column(Integer,index=True)
   attr_concept_code   = Column(Integer)
   attr_concept_label  = Column(String(200))
   attr_short_label    = Column(String(200))
   attr_long_label     = Column(String(200))
   intv_concept_code   = Column(Integer)
   intv_concept_label  = Column(String(200))
   intv_short_label    = Column(String(200))
   intv_long_label     = Column(String(200))
   value             = Column(Float)
   valueUOM          = Column(String(200))
   tstamp            = Column(DateTime)

   def __repr__(self):
       return 'ChartEvents(patient internal_id={},{},{},{},{},{},{})'.format(
           self.encounter_id, self.attr_concept_code, self.attr_concept_label,
           self.intv_concept_code, self.intv_concept_label,self.value, self.tstamp)

class MedEvents(Base):
   """"""
   __tablename__  = "MedEvents"
   id             = Column(Integer,primary_key=True)
   encounter_id   = Column(Integer,index=True)
   attr_concept_code   = Column(Integer)
   attr_concept_label  = Column(String(200))
   attr_short_label    = Column(String(200))
   attr_long_label     = Column(String(200))
   intv_concept_code   = Column(Integer)
   intv_concept_label  = Column(String(200))
   intv_short_label    = Column(String(200))
   intv_long_label     = Column(String(200))
   mat_concept_code    = Column(Integer)
   mat_concept_label   = Column(String(200))
   mat_short_label     = Column(String(200))
   mat_long_label      = Column(String(200))
   value               = Column(Float)
   valueUOM            = Column(String(200))
   tstamp              = Column(DateTime)

   def __repr__(self):
       return 'MedEvents(patient internal_id={},{},{},{},{},{},{})'.format(
           self.encounter_id, self.attr_concept_code, self.attr_concept_label,
           self.intv_concept_code, self.intv_concept_label,self.value, self.tstamp)

class FluidEvents(Base):
   """"""
   __tablename__   = "FluidEvents"
   id              = Column(Integer,primary_key=True)
   encounter_id    = Column(Integer,index=True)
   attr_concept_code   = Column(Integer)
   attr_concept_label  = Column(String(200))
   attr_short_label    = Column(String(200))
   attr_long_label     = Column(String(200))
   intv_concept_code   = Column(Integer)
   intv_concept_label  = Column(String(200))
   intv_short_label    = Column(String(200))
   intv_long_label     = Column(String(200))
   value             = Column(Float)
   valueUOM          = Column(String(200))
   tstamp            = Column(DateTime)

   def __repr__(self):
       return 'FluidEvents(patient internal_id={},{},{},{},{},{},{})'.format(
           self.encounter_id, self.attr_concept_code, self.attr_concept_label,
           self.intv_concept_code, self.intv_concept_label,self.value, self.tstamp)
