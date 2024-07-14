from collections import deque, namedtuple, defaultdict, Counter

EXAMPLE_SEQUENCE = ['dog', 'cat', 'mouse', 'parrot', 'frog']

dq = deque(EXAMPLE_SEQUENCE)
dq.append("date")
dq.popleft()
print('Deque: ', dq)

# namedtuple
animals = []
classifications = {
    'dog': ('Mammal', 'Carnivora', 'Canidae'),
    'cat': ('Mammal', 'Carnivora', 'Felidae'),
    'mouse': ('Mammal', 'Rodentia', 'Muridae'),
    'parrot': ('Bird', 'Psittaciformes'),
    'frog': ('Amphibian', 'Anura')
}
Animal = namedtuple('Animal', 'name classification')
for name in EXAMPLE_SEQUENCE:
    classification = classifications[name]
    animal = Animal(name=name, classification=classification)
    animals.append(animal)

for animal in animals:
    print(animal)

# defaultdict
dd = defaultdict(list)
dd["animal"]
print("DefaultDict:", dd)

# Counter
cnt = Counter(EXAMPLE_SEQUENCE)
print("Counter:", cnt)