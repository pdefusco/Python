#now we use the prompt variable as an argument to the
#input method in order to actually use a prompt sign

from sys import argv

script, username = argv

prompt = '> '

print(f"Hola {username}, you are running script {script}")
print(f"Please provide the following")
print(f"enter age:")

age = int(input(prompt))

print(f"provide weight")

weight = int(input(prompt))

print("here are the answers:")
print("age: ", age)

#Note you need to use the f format specifier in order to use the brackets to define a variable inside a string
print(f"weight {weight}")
