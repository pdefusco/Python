#python script to copy a file onto another
from sys import argv
from os.path import exists

script, from_file, to_file = argv

print(f"copying from {from_file}, to file {to_file}")
in_file = open(from_file)
indata = in_file.read()

print(f"the input file is {len(indata)} bytes long")
print(f"Does the output file exist? {exists(to_file)}")
print(f"Ready, hit RETURN to continue or CTRL-C to end")

input("?")

out_file = open(to_file,'w')
out_file.write(indata)

print("The file has been printed")

out_file.close()
in_file.close()
