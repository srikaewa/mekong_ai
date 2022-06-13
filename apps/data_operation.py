# authentication functions

from sqlalchemy import create_engine, update
import psycopg2

from apps.config import config
import geopandas as gpd
import pandas as pd
from datetime import datetime


def save_to_postgis(gdf, name):
    """ Connect to the PostgreSQL database server """
    conn = None
    # read connection parameters
    params = config(section="postgresql_gis")

    # connect to the PostgreSQL server
    print('Connecting to the PostgreSQL database for importing new ' + name + '...')
    conn = create_engine('postgresql://' + params.get('user') + ':' + params.get('password') + '@' + params.get('host') + '/' + params.get('database'))
    print("YOOOOOOOOOOo")
    print(gdf.head())        
    gdf.to_postgis(name=name, con=conn, if_exists='append', index=False)
    

def load_from_postgis(sql):
    """ Connect to the PostgreSQL database server """
    conn = None
    # read connection parameters
    params = config(section="postgresql_gis")
    # connect to the PostgreSQL server
    print('Connecting to the PostgreSQL database for loading data...')
    conn = create_engine('postgresql://' + params.get('user') + ':' + params.get('password') + '@' + params.get('host') + '/' + params.get('database'))       
    return gpd.read_postgis(sql, con=conn)

def delete_label_rice_variety(label, rice_variety):
    conn = None
    """ Connect to the PostgreSQL database server """
    conn = None
    # read connection parameters
    params = config(section="postgresql_gis")
    # connect to the PostgreSQL server
    print('Connecting to the PostgreSQL database for deleting data...')
    conn = psycopg2.connect(**params)
    cur = conn.cursor()

    sql = "DELETE FROM label_rice_variety WHERE label = '" +label + "'"
    cur.execute(sql)
    conn.commit()

    cur.close()

def update_label_rice_variety(label, rice_variety):
    conn = None
    """ Connect to the PostgreSQL database server """
    conn = None
    # read connection parameters
    params = config(section="postgresql_gis")
    # connect to the PostgreSQL server
    print('Connecting to the PostgreSQL database for update data...')
    conn = psycopg2.connect(**params)
    cur = conn.cursor()

    sql = "UPDATE label_rice_variety SET rice_variety = '" + rice_variety + "' WHERE label = '" + label + "'"
    cur.execute(sql)
    conn.commit()

    cur.close()

def save_ann_model_to_postgis(title, n_layer=1, n_neuron=128, optimizer='Adam', n_epoch=50):
    conn = None
    """ Connect to the PostgreSQL database server """
    conn = None
    # read connection parameters
    params = config(section="postgresql_gis")
    # connect to the PostgreSQL server
    print('Connecting to the PostgreSQL database for creating ann model...')
    conn = psycopg2.connect(**params)
    cur = conn.cursor()
    dt = datetime.now()
    sql = "INSERT INTO ann_model (title,n_layer,n_neuron,optimizer,n_epoch) VALUES ('{}','{}','{}','{}','{}')".format(title, n_layer, n_neuron, optimizer, n_epoch)
    print(sql)
    cur.execute(sql)
    conn.commit()

    cur.close()

def save_blast_ann_model_to_postgis(title, n_layer=1, n_neuron=128, optimizer='Adam', n_epoch=50):
    conn = None
    """ Connect to the PostgreSQL database server """
    conn = None
    # read connection parameters
    params = config(section="postgresql_gis")
    # connect to the PostgreSQL server
    print('Connecting to the PostgreSQL database for creating blast ANN model...')
    conn = psycopg2.connect(**params)
    cur = conn.cursor()
    dt = datetime.now()
    sql = "INSERT INTO blast_ann_model (title,n_layer,n_neuron,optimizer,n_epoch) VALUES ('{}','{}','{}','{}','{}')".format(title, n_layer, n_neuron, optimizer, n_epoch)
    print(sql)
    cur.execute(sql)
    conn.commit()

    cur.close()

def update_ann_model_to_postgis(table, id, title, n_epoch=50):
    conn = None
    """ Connect to the PostgreSQL database server """
    conn = None
    # read connection parameters
    params = config(section="postgresql_gis")
    # connect to the PostgreSQL server
    print('Connecting to the PostgreSQL database for updating ann model...')
    conn = psycopg2.connect(**params)
    cur = conn.cursor()

    sql = "UPDATE {} SET title = '{}', n_epoch = '{}', updated_at = '{}'".format(table, title, n_epoch, datetime.now()) + " WHERE id = '" + id + "'"
    print(sql)
    cur.execute(sql)
    conn.commit()
    cur.close()

def update_ann_model_name_to_postgis(table, id, model_file_name):
    conn = None
    """ Connect to the PostgreSQL database server """
    conn = None
    # read connection parameters
    params = config(section="postgresql_gis")
    # connect to the PostgreSQL server
    print('Connecting to the PostgreSQL database for updating ann model...')
    conn = psycopg2.connect(**params)
    cur = conn.cursor()

    sql = "UPDATE {} SET model_name = '{}', last_trained_at = '{}'".format(table, model_file_name, datetime.now()) + " WHERE id = '" + id + "'"
    cur.execute(sql)
    conn.commit()
    cur.close()

def update_dataset_title_ann_model_to_postgis(table, id, dataset_title):
    conn = None
    """ Connect to the PostgreSQL database server """
    conn = None
    # read connection parameters
    params = config(section="postgresql_gis")
    # connect to the PostgreSQL server
    print('Connecting to the PostgreSQL database for updating ann model...')
    conn = psycopg2.connect(**params)
    cur = conn.cursor()

    sql = "UPDATE {} SET dataset_title = '{}'".format(table, dataset_title) + " WHERE id = '" + id + "'"
    print(sql)
    cur.execute(sql)
    conn.commit()
    cur.close()

def load_all_ann_model_from_postgis():
    conn = None
    """ Connect to the PostgreSQL database server """
    conn = None
    # read connection parameters
    params = config(section="postgresql_gis")
    # connect to the PostgreSQL server
    print('Connecting to the PostgreSQL database for creating ann model...')
    conn = psycopg2.connect(**params)
    cur = conn.cursor()

    sql = "SELECT * FROM ann_model"
    #print(sql)
    cur.execute(sql)
    result = cur.fetchall()
    conn.commit()
    cur.close()
    return pd.DataFrame(result, columns=['title', 'n_layer', 'n_neuron', 'optimizer', 'n_epoch', 'created_at', 'id', 'updated_at',  'dataset_title', 'model_name', 'last_trained_at'])

def load_all_blast_ann_model_from_postgis():
    conn = None
    """ Connect to the PostgreSQL database server """
    conn = None
    # read connection parameters
    params = config(section="postgresql_gis")
    # connect to the PostgreSQL server
    print('Connecting to the PostgreSQL database for loading ann model...')
    conn = psycopg2.connect(**params)
    cur = conn.cursor()

    sql = "SELECT * FROM blast_ann_model"
    #print(sql)
    cur.execute(sql)
    result = cur.fetchall()
    conn.commit()
    cur.close()
    return pd.DataFrame(result, columns=['title', 'n_layer', 'n_neuron', 'optimizer', 'n_epoch', 'created_at', 'id', 'updated_at',  'dataset_title', 'model_name', 'last_trained_at'])


def load_ann_model_from_postgis(idname, column_names):
    conn = None
    """ Connect to the PostgreSQL database server """
    conn = None
    # read connection parameters
    params = config(section="postgresql_gis")
    # connect to the PostgreSQL server
    print('Connecting to the PostgreSQL database for creating ann model...')
    conn = psycopg2.connect(**params)
    cur = conn.cursor()

    sql = "SELECT * FROM ann_model WHERE idname = '" + idname + "'"
    #print(sql)
    cur.execute(sql)
    result = cur.fetchall()
    conn.commit()
    cur.close()
    return pd.DataFrame(result, columns=column_names)

def load_dataset_list(table):
    conn = None
    """ Connect to the PostgreSQL database server """
    conn = None
    # read connection parameters
    params = config(section="postgresql_gis")
    # connect to the PostgreSQL server
    print('Connecting to the PostgreSQL database for loading dataset list...')
    conn = psycopg2.connect(**params)
    cur = conn.cursor()

    sql = "SELECT DISTINCT dataset_title FROM " + table
    #print(sql)
    cur.execute(sql)
    result = cur.fetchall()

    conn.commit()
    cur.close()
    return pd.DataFrame(result, columns=['dataset_title'])

def load_bph_model_data(dataset_title):
    conn = None
    """ Connect to the PostgreSQL database server """
    conn = None
    # read connection parameters
    params = config(section="postgresql_gis")
    # connect to the PostgreSQL server
    print('Connecting to the PostgreSQL database for loading model data...')
    conn = psycopg2.connect(**params)
    cur = conn.cursor()

    sql = "SELECT * FROM data_model_input WHERE dataset_title = '{}' ORDER BY latitude ASC, longitude ASC, date ASC;".format(dataset_title)
    #print(sql)
    cur.execute(sql)
    result = cur.fetchall()

    # Extract the column names
    col_names = []
    for elt in cur.description:
        col_names.append(elt[0])

    conn.commit()
    cur.close()
    return pd.DataFrame(result, columns=col_names)

def load_blast_model_data(dataset_title):
    conn = None
    """ Connect to the PostgreSQL database server """
    conn = None
    # read connection parameters
    params = config(section="postgresql_gis")
    # connect to the PostgreSQL server
    print('Connecting to the PostgreSQL database for loading model data...')
    conn = psycopg2.connect(**params)
    cur = conn.cursor()

    sql = "SELECT * FROM data_blast_model_input WHERE dataset_title = '{}' ORDER BY latitude ASC, longitude ASC, date ASC;".format(dataset_title)
    #print(sql)
    cur.execute(sql)
    result = cur.fetchall()

    # Extract the column names
    col_names = []
    for elt in cur.description:
        col_names.append(elt[0])

    conn.commit()
    cur.close()
    return pd.DataFrame(result, columns=col_names)