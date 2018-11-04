#intro to methods:

#function taking as many methods as specified
def func1(*args):
    arg1, arg2 = args
    #will return an error if more or less arguments are provided
    print(f"arg1: {arg1}, arg2: {arg2}")

def func2(arg1, arg2):
    print(arg1,arg2)

def func3(arg1):
    print(arg1)

def func4():
    print("No inputs to print")

func1("1", 2)
func2(1,2)
func3("ee")
func4()
