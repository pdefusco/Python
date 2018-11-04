#putting concepts together:

print("This is a variation of the original exercise")

text = """Ma quanto e bello andare in giro per i colli bolognesi
\t se hai una vespa special che \n ti toglie i problemi
anche con l\'apopstrofo

"""

print("----")
print(text)
print("----")

result = 1**2+3**6
print("now doing some simple math:", result)
print(f"now printing the same result again: {result}")

def more_math(result):
    var1 = result**2
    var2 = result*var1
    var3 = var1/var2
    return var1, var2, var3

#now back to formatting strings:
print("This was anotehr way to print the formatted result: {}".format(result))
print("now we output the three variables from the more_math method:")
more_math_result = more_math(result)
print("Var1: {}, Var2: {}, Var3: {}.".format(*more_math_result))
