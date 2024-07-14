EXAMPLE_SEQUENCE = [1, 4, 12, 9, 22, 5, 1, 9]

square_sequence = [item * item for item in EXAMPLE_SEQUENCE]
print(square_sequence)

sequence_dict = {item: item * item for item in EXAMPLE_SEQUENCE}
print(sequence_dict)

sequence_set = {item for item in EXAMPLE_SEQUENCE}
print(sequence_set)