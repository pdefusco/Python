#now we use read, open, input etc, in a more complex script

from sys import argv

script, filename = argv

print(f'we\'re going to erase {filename}')
print('To exit, hit CTRL-C')
print('Otherwise hit return')

input("?")

print("Opening the file: ")
target = open(filename, 'w')

print("Truncating the file. Goodbye!")

target.truncate()

print("Now enter three lines of text:")
line1 = input("Line 1: ")
line2 = input("Line 2: ")
line3 = input("Line 3: ")

print("I'm going to write these to the file")

target.write(line1)
target.write("\n")
target.write(line2)
target.write("\n")
target.write(line3)
target.write("\n")

print("And finally we close the file")
target.close()
#closing the file effectively saves it
