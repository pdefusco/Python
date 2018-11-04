'''
Objective: this application is used as a book inventory management and recommendation tool.
The user is presented an interface to track and review books read
Books info is stored in a sql database
The application recommends a book based on past user ratings
'''

class Book(object):

    def __init__(self, title, author, genre, rating = None):
        self.title = title
        self.author = author
        self.genre = genre
        self.rating = rating

    def print_book_info(self):
        print(f"The book is titled: {self.title}")
        print(f"The author is: {self.author}")
        print(f"The book genre is: {self.genre}")
        print(f"The rating given is: {self.rating}")

    def rate_book(self):
        print(f"Please rate the following book: {self.title}")
        print("-------")
        print(f"How many stars (1 to 5) will you rate the book?")

        try:
            self.rating = int(input("> "))

        except ValueError:
            print("Value Error - You did not enter an integer")
            print("Please enter an integer between 1 and 5")

        if self.rating < 1:
            print(f"Error: you rated the book {self.title} less than 1")
            print(f"Please give a rating between 1 and 5 stars")
            self.rating = None
            #rate_book(self)

        elif self.rating > 5:
            print(f"Error: you rated the book {self.title} more than 5")
            print(f"Please give rating between 1 and 5 stars")
            self.rating = None
            #rate_book(self)

        else:
            print(f"Good job, you rated {self.title} with {self.rating} stars")
            
