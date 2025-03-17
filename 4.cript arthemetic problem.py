import itertools
def cryptarithmetic_solver(equation):
    words = equation.replace('+', ' ').replace('=', ' ').split()
    unique_letters = set(''.join(words))
    if len(unique_letters) > 10:
        return "No solution, more than 10 unique letters"
    digit_combinations = itertools.permutations(range(10), len(unique_letters))
    for combination in digit_combinations:
        letter_to_digit = dict(zip(unique_letters, combination))
        left_side = evaluate_expression(equation.split('=')[0], letter_to_digit)
        right_side = evaluate_expression(equation.split('=')[1], letter_to_digit)
        if left_side == right_side:
            return letter_to_digit
    return "No valid solution found"
def evaluate_expression(expression, letter_to_digit):
    words = expression.split('+')
    total = 0
    for word in words:
        word_value = 0
        for letter in word.strip():
            word_value = word_value * 10 + letter_to_digit[letter]
        total += word_value
    return total
equation = "SEND + MORE = MONEY"
solution = cryptarithmetic_solver(equation)
if isinstance(solution, dict):
    print("Solution found:")
    for letter, digit in solution.items():
        print(f"{letter}: {digit}")
else:
    print(solution)
