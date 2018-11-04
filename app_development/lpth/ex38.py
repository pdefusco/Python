#ex38.py

long_string = "word1 gigi surf object kayak"

list_of_words = long_string.split(" ")

other_list = ["abacus", "travel", "hotel", "exotic", "bird", "car"]

while len(list_of_words)!=7:
    new_word = other_list.pop()
    print(f"Adding word {new_word} to list of words")
    list_of_words.append(new_word)
    print(f"After adding the new word we have now {len(list_of_words)} words in the list")

print(list_of_words[0])
print(list_of_words[-1])
print(list_of_words.pop())
print("Now we create a new string from the list: ", "".join(list_of_words))
print("Here is a different way to create a string from the list of words: ")
print('#'.join(list_of_words[3:6]))
