import random

s = "abcabc"

unique_chars = list(set(s))
alphabet = 'abcdefghijklmnopqrstuvwxyz'
for char in unique_chars:
    new_char = random.choice(alphabet)
    s = s.replace(char, new_char)
print(s)