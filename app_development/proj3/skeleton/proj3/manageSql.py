import sqlite3
from sqlite3 import Error

class CreateSQL():


    def __init__(self):
        self.database = "../db/pythonsqlite.db"

    def create_connection(db_file):
        """ create a database connection to the SQLite database
            specified by db_file
        :param db_file: database file
        :return: Connection object or None
        """
        try:
            conn = sqlite3.connect(db_file)
            return conn
        except Error as e:
            print(e)

        return None

    #if __name__ == '__main__':
    #    create_connection("/db/pythonsqlite.db")

    def create_table(conn, create_table_sql):
        """ create a table from the create_table_sql statement
        :param conn: Connection object
        :param create_table_sql: a CREATE TABLE statement
        :return:
        """
        try:
            c = conn.cursor()
            c.execute(create_table_sql)
        except Error as e:
            print(e)

    def main(self):

        #"../db/pythonsqlite.db"

        sql_create_projects_table = """ CREATE TABLE IF NOT EXISTS books (
                                            id integer PRIMARY KEY,
                                            title text NOT NULL,
                                            genre text NOT NULL,
                                            rating integer
                                        ); """

        # create a database connection
        conn = create_connection(database)
        if conn is not None:
            # create projects table
            create_table(conn, sql_create_projects_table)
            # create additional tables here by adding more calls to the create table method
        else:
            print("Error! cannot create the database connection.")


    #if __name__ == '__main__':
    #main()
