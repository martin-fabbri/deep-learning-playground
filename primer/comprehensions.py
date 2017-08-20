halves = [(idx / 2) for idx in range(1, 100)]
print(halves)

fizz = [num for num in range(1, 100) if num % 3 == 0]
print(fizz)

rows = range(4)
cols = range(10)

print([(x, y) for x in rows for y in cols])

total_nums = range(1, 101)

fizz_buzzes = {
    'fizz': [n for n in total_nums if n % 3 == 0],
    'buzz': [n for n in total_nums if n % 7 == 0]
}

print(fizz_buzzes)