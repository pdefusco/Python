#class OOP practice
#HAS relationship practice

class Other(object):

    def override(self):
        print("Other override()")

    def implicit(self):
        print("OTHER implicit()")

    def altered(self):
        print("OTHER altered")

class Child(object):

    #child "contains" or HAS an instance of OTHER
    #instance of other contained in instance of Child acts fundamentally as an atrribute
    #except you can call its methods and use them as needed

    def __init__(self):
        self.other = Other()

    def implicit(self):
        self.other.implicit()

    def override(self):
        print("Child override")

    def altered(self):
        print("CHILD, before other altered")
        self.other.altered()
        print("CHILD, after other altered")

son = Child()

son.implicit()
son.override()
son.altered()
