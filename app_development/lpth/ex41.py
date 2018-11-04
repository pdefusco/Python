#class practice
#Dog IS an animal

class Animal(object):

    def __init__ (self):
        self.hasLegs = True
        self.hasHead = True
        self.numberOfLegs = 5

    def run(self, distance):
        time_walked = distance*10
        print(f"Time walked: {time_walked} Minutes")
        return time_walked

    def animal_noise(self):
        print('Making default noise')

class Dog(Animal):

    def __init__(self):

        super(Dog, self).__init__()
        self.numberOfLegs = 4

    def run(self, distance):
        time_walked = distance*1
        print(f"Time walked: {time_walked} Minutes")
        return time_walked

    def animal_noise(self):
        super(Dog, self).animal_noise()

dog1 = Dog()
dog1.run(100)
animal1 = Animal()
animal1.run(100)

print(animal1.numberOfLegs)
print(dog1.numberOfLegs)
print(dog1.hasLegs)

dog1.animal_noise()
