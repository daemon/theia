import datetime
import random
import time

from psycopg2.pool import ThreadedConnectionPool
from sqlalchemy import *
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, subqueryload, joinedload, Bundle
import psycopg2
import psycopg2.errorcodes as errorcodes
import sqlalchemy.exc as errors

Base = declarative_base()
engine = create_engine("postgresql+psycopg2://alpha:$test123@127.0.0.1/alpha", pool_recycle=3600, pool_size=1)
Base.metadata.bind = engine
Session = sessionmaker(bind=engine)
INIT_SESSION = None
INIT_TRANSACTION = None
INIT_CONNECTION = None

def column_names(table):
    return [c.name for c in table.columns]

last_check_time = time.time()
def get_connection():
    global last_check_time
    connection = connection_pool.getconn()
    if time.time() - last_check_time > 1800:
        last_check_time = time.time()
        try:
            c = connection.cursor()
            c.execute("SELECT 1")
        except:
            pass
        if connection.closed:
            return_connection(connection, close=True)
            return get_connection()
    return connection

def rows_splice(*args):
    rows = args[0]
    tables = args[1:]
    spliced = [[]] * len(tables)
    for row in rows:
        i = 0
        j = 0
        for table in tables:
            c_len = len(table.columns)
            spliced[j].append(row[i:i + c_len])
            i += c_len
            j += 1
    return spliced

def return_connection(connection, close=False):
    connection_pool.putconn(connection, close=close)

def access_point(retries=3):
    def decorator(f):
        def wrapper(*args, **kwargs):
            tries = 0
            created_connection = False
            if not "connection" in kwargs or kwargs["connection"] is None:
                created_connection = True
                kwargs["connection"] = get_connection()
            try:
                if created_connection:
                    while tries < retries:
                        try:
                            value = f(*args, **kwargs)
                            kwargs["connection"].commit()
                            return value
                        except psycopg2.Error as e:
                            kwargs["connection"].rollback()
                            if e.pgcode != errorcodes.DEADLOCK_DETECTED:
                                raise e
                        except:
                            kwargs["connection"].rollback()
                            raise
                else:
                    return f(*args, **kwargs)
            finally:
                if created_connection:
                    return_connection(kwargs["connection"])
        return wrapper
    return decorator

def init_from_row(obj, names, args):
    for (name, arg) in zip(names, args):
        setattr(obj, name, arg)

def bulk_insert_str(c, data):
    columns = len(data[0])
    stmt = " ({}) ".format(",".join(["%s"] * columns))
    return ','.join(c.mogrify(stmt, row).decode("utf-8") for row in data)

def subset_dict(dictionary, keys):
    return {k: dictionary[k] for k in dictionary.keys() & keys}

def join_where(statement, table, dictionary):
    for key in dictionary:
        statement = statement.where(getattr(table.c, key) == dictionary[key])
    return statement

def join_conditions(kwargs, condition, params, col_names=None):
    condition = " " + condition + " "
    join_strings = []
    for key, value in kwargs.items():
        try:
            index = params.index(key)
        except ValueError:
            continue
        if value is None:
            string = "ISNULL(%s)" % (col_names[index] if col_names else key)
        else:
            string = "{}=%s".format(col_names[index] if col_names else key)
        join_strings.append(string)
    conditions = condition.join(join_strings)
    found_params = tuple((value for key, value in kwargs.items() if (key in params and value is not None)))
    return (conditions, found_params)

def sec_random_gen(length, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz123456789$!@#$%^&*()"):
    return ''.join(random.SystemRandom().choice(alphabet) for _ in range(length))

statements = []
def add_init_statements(*args):
    statements.append(args)

@access_point()
def execute_init_statements(connection=INIT_CONNECTION):
    global statements
    c = connection.cursor()
    for statement in statements:
        c.execute(statement)
    statements = []

def initialize(conf):
    global connection_pool
    import db.user
    import db.assignment
    Base.metadata.create_all(engine)
    connection_pool = ThreadedConnectionPool(3, 5, **conf)
    execute_init_statements()