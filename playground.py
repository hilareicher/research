import random
def read_file(filename):
    with open(filename, 'r') as file:
        return file.read().splitlines()

def intersection(list1, list2):
    return list(set(list1) & set(list2))
# Provided list of items (filename, count)
# load and read neta.csv file from directory
csv_file = open('neta.csv', 'r')
lines = csv_file.readlines()
# lines are in the format of "filename,count"
items = [line.strip().split(',') for line in lines]
csv_file.close()
list2 = read_file('esther.txt')

# intersect list2 with the first element of items
list1 = [item[0] for item in items]
common_elements = intersection(list1, list2)
# Now select 50 items randomly from the intersection
selected_items = random.sample(common_elements, 50)
# print the selected items with their original count
for item in items:
    if item[0] in selected_items:
        print(f"{item[0]},{item[1]}")




