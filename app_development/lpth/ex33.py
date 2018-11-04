#while loops;

count = 0
numbers = []

while count < 100:
    print(f"now we\'re on number {count}")
    numbers.append(count)

    count += 1

    print("now the count is: ", count)

print("Here is the final list: ")

for i in numbers:
    print(i)
