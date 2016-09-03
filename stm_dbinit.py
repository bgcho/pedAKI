# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 11:57:06 2015

@author: 310153046
"""

import stm_tabledef as ptd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine import reflection
from sqlalchemy.schema import (
                                MetaData,
                                Table,
                                DropTable,
                                ForeignKeyConstraint,
                                DropConstraint,
                               )

def make_connection(db_conn_str,db_echo=True):
    """Create SQLAlchemy connection to database

    Parameters
    ----------
    db_conn_str : str
        SQLAlchemy connection string.  See http://docs.sqlalchemy.org/en/rel_0_9/core/engines.html for examples
    db_echo : bool, optional
        Flag indicating whether to echo SQL commands, defaults to True

    Returns
    -------
    out : tuple
        First element is the SQLAlchemy engine, second element is the session

    """
    engine = create_engine(db_conn_str, echo=db_echo)

    # create a Session
    Session = sessionmaker(bind=engine)
    session = Session()
    return (engine,session)
    
def drop_tables(engine):
    """
    drop tables
    """
    conn = engine.connect()

    # the transaction only applies if the DB supports
    # transactional DDL, i.e. Postgresql, MS SQL Server
    trans = conn.begin()

    print("  Dropping - Inspecting tables")
    inspector = reflection.Inspector.from_engine(engine)

    # gather all data first before dropping anything.
    # some DBs lock after things have been dropped in
    # a transaction.

    metadata = MetaData()

    tbs = []
    all_fks = []

    print("  Dropping - Getting table names and keys")
    for table_name in inspector.get_table_names():
        fks = []
        for fk in inspector.get_foreign_keys(table_name):
            if not fk['name']:
                continue
            fks.append(
                ForeignKeyConstraint((), (), name=fk['name'])
            )
        t = Table(table_name, metadata, *fks)
        tbs.append(t)
        all_fks.extend(fks)

    print("  Dropping - Dropping constraints")
    for fkc in all_fks:
        conn.execute(DropConstraint(fkc))

    print("  Dropping - Dropping tables")
    for table in tbs:
        conn.execute(DropTable(table))

    trans.commit()

    print("If this is a sqlite database, 'vacuum' command should be run")

def create_tables(engine):
    """
    Create tables
    """
    try:
        ptd.Base.metadata.create_all(engine)
    except AttributeError, e:
        if type(engine) != 'sqlalchemy.engine.base.Engine':
            print('Please make sure that argument to create_tables '
                  'is a sqlalchemy engine')
            print('error = {}'.format(e))
        raise
