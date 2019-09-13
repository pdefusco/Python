##Student will move from 9th to 10th grades
##Student will receive a Map object containing the grades he will mvoe through

from sys import exit
from random import randint
from textwrap import dedent

class Grade(object):

    def enter(self):
        print("This grade is not yet configured")
        print("Subclass it and implement enter()")
        exit(1)


class Engine(object):

    def __init__(self, grade_map):
        self.grade_map = grade_map

    def move_grade(self):
        current_grade = self.grade_map.opening_grade()
        last_grade = self.grade_map.next_grade('finished')

        while current_grade != last_grade:
            next_grade_name = current_grade.enter()
            current_grade = self.grade_map.next_grade(next_grade_name)

        current_grade.enter()


class Grade_Nine(Grade):

    summary = ['This is the first grade in high school']

    def enter(self):
        print(Grade_Nine.summary[randint(0, len(self.summary)-1)])
        exit(1)


class Grade_Ten(Grade):

    summary = ['This is the second grade in high school']

    def enter(self):
        print(Grade_Ten.summary[randint(0, len(self.summary)-1)])
        exit(1)

class Finished(Grade):

    def enter(self):
        print('You finished two grades!')
        return 'finished!'

class Map(object):

    grades = {
        'ninth_grade':Grade_Ten(),
        'tenth_grade':Grade_Nine(),
        'finished':Finished()
    }

    def __init__(self, start_grade):
        self.start_grade = start_grade

    def next_grade(self, grade_name):
        val = Map.grades.get(grade_name)
        return val

    def opening_grade(self):
        return self.next_grade(self.start_grade)

a_map = Map('ninth_grade')
a_grade = Engine(a_map)
a_grade.move_grade()
