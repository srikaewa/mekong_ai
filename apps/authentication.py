# authentication functions

import psycopg2
from apps.config import config

import bcrypt

def check_if_user_exist(email):
    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        # read connection parameters
        params = config(section="postgresql_user")

        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)
		
        # create a cursor
        cur = conn.cursor()
        
	# execute a statement
        #print('PostgreSQL database version:')
        sql = "SELECT * FROM data_user WHERE email = '{email}'".format(email=email)
        cur.execute(sql)

        # display the PostgreSQL database server version
        db_user = cur.fetchone()
        #df_user = pd.read_sql(sql, con=conn)
       
	# close the communication with the PostgreSQL
        cur.close()
        if db_user:            
            return True
        else:
            return False
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')

def create_new_user(email, password, first_name, last_name):
    bytePwd = password.encode('utf-8')
    mySalt = bcrypt.gensalt(12)
    hash = bcrypt.hashpw(bytePwd, mySalt).decode('utf-8')
    #print(hash)
    #print(bcrypt.checkpw(bytePwd, hash))
    # write to database
    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        # read connection parameters
        params = config(section="postgresql_user")

        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database for creating new user...')
        conn = psycopg2.connect(**params)
		
        # create a cursor
        cur = conn.cursor()
        
	# execute a statement
        #print('PostgreSQL database version:')
        sql = """INSERT INTO data_user (email, password, first_name, last_name, is_superuser, is_authorized, is_admin, is_active) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"""

        cur.execute(sql, (email, hash, first_name, last_name, False, False, False, True))

        conn.commit()
        count = cur.rowcount
        #df_user = pd.read_sql(sql, con=conn)
        print(count, "User record inserted successfully into database")

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')

def login(email, password):
    bytePwd = password.encode('utf-8')

    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        # read connection parameters
        params = config(section="postgresql_user")

        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database for logging in...')
        conn = psycopg2.connect(**params)
		
        # create a cursor
        cur = conn.cursor()
        
	# execute a statement
        #print('PostgreSQL database version:')
        sql = """SELECT * FROM data_user WHERE email = %s"""

        cur.execute(sql, (email,))
        user_record = cur.fetchone()
        if not user_record:
            return 0, "Email is not registered!", None
        #print("User : ", user_record[1])
        if bcrypt.checkpw(password.encode('utf-8'), user_record[1].encode('utf-8')):
            if user_record[4]:
                return 4, "Superuser {email} has logged in successfully".format(email=email), user_record
            elif user_record[6]:
                return 3, "Admin user {email} has logged in successfully".format(email=email), user_record
            elif user_record[5]:
                return 2, "Authorized user {email} has logged in successfully".format(email=email), user_record
            else:
                return 1, "User {email} has logged in successfully".format(email=email), user_record
        else:
            return 0, "Invalid email/password!", None


    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')