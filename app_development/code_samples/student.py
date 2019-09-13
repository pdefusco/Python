class Student(object):

    def __init__(self, grade, score):

        self.grade = grade
        self.score = score

    def take_test(self):

        if self.score > 20:
            print('pass')

gigi = Student("12th", 21)

gigi.take_test()
