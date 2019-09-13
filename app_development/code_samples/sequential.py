## Sequential
## Program to iterate between different steps
## Showcasing the use of an Engine class to iterate

from sys import exit


class Room(object):

    def enter(self):
        print('Not configured')
        exit(1)

class Engine(object):

    def __init__(self, room_sequence):
        self.room_sequence = room_sequence

    def execute(self):
        current_room = self.room_sequence.opening_room()
        last_room = self.room_sequence.next_room('garage')

        while current_room != last_room:
            next_room_name = current_room.enter()
            current_room = self.room_sequence.next_room(next_room_name)

        current_room.enter()


class Lobby(Room):

    #Class variable:
    room_description = "This room is a lobby"

    def enter(self):
        print(Lobby.room_description)
        return 'livingroom'

class LivingRoom(Room):

    def enter(self):
        print(f'Please let us know if you like the room: ')
        print(f'Enter Yes or No')
        living_room_input = input('>')

        if living_room_input == 'Yes':
            print('You have stated that you like the room')
            return 'garage'

        elif living_room_input == 'No':
            print('You have stated that you do not like the room')
            return 'lobby'

        else:
            print('Please try again')
            return 'livingroom'

class Garage(Room):

    def enter(self):
        print('You completed the tour!')
        return 'garage'

class Map(object):

    rooms = {
        'lobby':Lobby(),
        'livingroom':LivingRoom(),
        'garage':Garage()
    }

    def __init__(self, start_room):
        self.start_room = start_room

    def next_room(self, room_name):
        val = Map.rooms.get(room_name)
        return val

    def opening_room(self):
        return self.next_room(self.start_room)

a_map = Map('lobby')
a_game = Engine(a_map)
a_game.execute()
