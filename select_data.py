import csv
import random
from math import floor, ceil

count = 0
with open('dataset.csv', 'r') as count_file:
    csv_reader = csv.reader(count_file)
    lines = []
    for row in csv_reader:
        count += 1
        lines.append(row)

test = ceil(count * 3/10)
train = floor(count * 7/10)
print(test, train)

choice1 = random.sample(lines, test)
choice2 = random.sample(lines, train)

with open('test_label.csv', "w", newline='') as sink:
    writer = csv.writer(sink)
    writer.writerow(['directory', 'label'])
    writer.writerows(choice1)
    
with open('train_label.csv', "w", newline='') as sink:
    writer = csv.writer(sink)
    writer.writerow(['directory', 'label'])
    writer.writerows(choice2)
    