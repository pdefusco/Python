#accessing elements of a list:

animals = ['cow', 'bird', 'dog']

bird = animals[1]

#ex 35.py:

from sys import exit

def blue_room():
    print("\n This is the method from the blue room")

def black_room():
    print("\n This is the method from the black room")

def yellow_room():
    print("\n this is not a good room to be in")
    exit(0)

def enter_room():

    door_choices = ['Blue Door', 'Black Door', 'Yellow Room']

    print("Now you have entered the room")
    print("Choose which door to take among the following: ")

    for i in door_choices:
        print(i)

    choice = input('> ')

    if choice == 'Blue Door':
        print('You have chosen: ', choice)
        blue_room()
    elif choice == 'Black Door':
        print('You have chosen: ', choice)
        black_room()
    elif choice == 'Yellow Room':
        print('You have chosen: ', choice)
        yellow_room()

    else:
        print('Your choice was outside the allowed range of options')
        print('You will re-enter your selection now:')
        enter_room()



enter_room()
