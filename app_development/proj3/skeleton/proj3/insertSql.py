#methods to insert rows into the books table

class InsertSQL():

    def __init__(self):
        self.database = database = "../db/pythonsqlite.db"

    #database = '../db/pythonsqlite.db'
    @staticmethod
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

    def add_book(conn, book_title, book_author, book_genre, book_rating):
        """
        Create a new book into the books table
        :param conn:
        :param project:
        :return: project id
        """
        inserts_list = [(book_title, book_author, book_genre, book_rating)]

        sql = '''INSERT INTO books(title,author,genre, rating)
                  VALUES(?,?,?,?)'''
        cur = conn.cursor()
        cur.executemany(sql, inserts_list)
        return cur.lastrowid

    #@staticmethod
    def main(self, book_title, book_author, book_genre, book_rating):

        #"../db/pythonsqlite.db"

        #database = self.database
        # create a database connection
        conn = self.create_connection("../db/pythonsqlite.db")
        with conn:
            # add a new book
            #book = ('Cool App with SQLite & Python', '2015-01-01', '2015-01-30');
            book_id = add_book(conn, book_title, book_author, book_genre, book_rating)
