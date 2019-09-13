## New App

## Class Structure
## Furniture
## Table is Furniture
## Chair is Furniture
## Table has Legs
## Chair has Legs

class Legs(object):

    def __init__(self, number_of_legs, leg_height):
        self.number_of_legs = number_of_legs
        self.leg_height = leg_height

class Furniture(object):

    def __init__(self, price):
        self.price = price

class Table(Furniture):

    def __init__(self, height, length):
        self.legs = Legs(4,10)
        self.height = height
        self.length = length
        self.code = str(height)+str(length)+'TT'

    def print_table_code(self):
        print(self.code)

    def print_table_legs(self):
        print(f'The number of Legs is: {self.legs.number_of_legs}')
        print(f'Leg height is: {self.legs.leg_height}')

class Chair(Furniture):

    def __init__(self, legs, height, length):
        self.legs = legs
        self.height = height
        self.length = length
        self.code = str(height)+str(length)+'TT'

    def print_chair_code(self):
        print(f'Printing Chair Code: {self.code}')

    def print_chair_legs(self):
        print(f'The number of legs is: {self.legs.number_of_legs}')
        print(f'The height of chair legs is: {self.legs.leg_height}')

print('------ Table Info ------')

print(f'Please enter table info')
print(f'How tall should the table be?')
input_table_height = input('>')
print(f'How long should the table be?')
input_table_length = input('>')

table = Table(input_table_height,input_table_length)
table.print_table_code()
table.print_table_legs()

print('------ Chair Info ------')

print(f'Please enter chair info')
print(f'How tall should the chair be?')
input_chair_height = input('>')
print(f'How long should the chair be?')
input_chair_length = input('>')
print(f'Please enter chair legs info')
print(f'How many legs should the chair have?')
input_chair_legs_count = input('>')
print(f'How tall should the chair legs be?')
input_chair_legs_height = input('>')

chair = Chair(Legs(input_chair_legs_count,input_chair_legs_height), input_chair_height, input_chair_length)
chair.print_chair_code()
chair.print_chair_legs()

chair = Chair(Legs(4,4), input_chair_height, input_chair_length)
chair.print_chair_code()
chair.print_chair_legs()
