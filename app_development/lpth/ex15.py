#opening a text file from a different script:

from sys import argv

script, filename = argv

txt = open(filename)

print(f"here is hte file {filename}")
print(txt.read())
print("Input the name of the file: ")
filename2 = input("> ")
text_again = open(filename2)
print(text_again.read())
