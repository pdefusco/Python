#loops and lists

list1 = ['a', 'b', 'c']
list2 = [1,2,3,4,5]
list3 = ['brown', 'blue', 'red']

for letter in list1:
    print(letter)

for item in list2:
    print(f"{item}")

list4 = []

for i in list3:
    print(f"adding {i} to the list")
    list4.append(i)

#two dimensional list:

twodlist = [['hello', 'ciao', 'gigi'], ['1', '2', '3']]
for i in twodlist:
    print(i)
