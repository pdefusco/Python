#dictionaries

d1 = {'key1':'val1'}

print(d1['key1'])

d1['key2'] = 'val2'
d1['key3'] = 'val3'

print(d1)

del d1['key2']

print(d1)

states = {'California': 'CA',
            'Ohio': 'OH',
            'Massachussettes':'MA',
            'Florida':'FL',
            'Alaska':'AK'
            }

for i in states.keys():
    print(i)

for key, val in list(states.items()):
    print(key + ' - ' + val)

#print("The abbreviation for the state of states[]")
