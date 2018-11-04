#working with inputs and printing the text

print("How old are you?", end= ' ')
age = input()
print("How tall are you?", end = ' ')
height = input()

print(f"The age is {age} and the height is {height}")

#now inputting numbers:

print("Let's do some math. Multuply value x times value y")
print('Assign value for x: ', end=' ')
x = int(input())
print('Now assign value for y:', end= ' ')
y = int(input())
print("The two values multiplied yield: ", x*y)
