#this exercise covers exercises 29-31
#basic work with conditional logic:
var1 = 100
var2 = 102
var3 = 400

choice = int(input(" "))

if choice < var2:
    print("choice less than var2")
elif choice > var2:
    print("choice is more than var2")
else:
    print("choice and var 2 must be equal")

option1 = "Car"
option2 = "Motorcycle"
option3 = "swim"

print(f"Now input a string of text among the following options:{option1}, {option2}, {option3}")
second_input = input("> ")

if second_input == option1:
    print("you have picked option1")
    print(f"you will be driving a {option1}")

elif second_input == option2:
    print("you have picked option2")
    print(f"you will be using {option2}")

elif second_input == option3:
    print("selection: ", option3)

elif second_input == "lalala" or second_input == "ciao":
    print("these are unusual selections")

else:
    print("you have made an erronoeus selection")
    print("please make a different selection")
