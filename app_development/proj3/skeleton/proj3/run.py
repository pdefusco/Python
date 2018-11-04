#run book.py with some examples
#the purpose of this script is to mainly test Book.py

#from proj3
from book import Book
from insertSql import InsertSQL

testBook = Book("", "", "")

book1 = Book("Divina Commedia", "Dante Alighieri", "Classic Literature")

book1.print_book_info()
book1.rate_book()
book1.print_book_info()

insert_statement = InsertSQL()
insert_statement.main(book1.title, book1.author, book1.genre, book1.rating )
