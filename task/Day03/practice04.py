import json

def save_to_file(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file, ensure_ascii=False)

def read_from_file(filename):
    with open(filename, 'r') as file:
        return json.load(file)


filename = 'data.json'
str = input('아무 문장이나 작성해주세요: ')
save_to_file(str, filename)
data = read_from_file(filename)
print(data)