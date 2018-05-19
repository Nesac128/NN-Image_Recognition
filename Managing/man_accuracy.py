import csv

plabels = []

with open('predictions_pfile.csv', 'r') as csvfile:
    data = csv.reader(csvfile)
    for row in data:
        plabels.append(row[1])

rlabels = []

with open('labels.txt', 'r') as txtfile:
    for line in txtfile.readlines():
        rlabels.append(line.split('\n')[0])

right = 0

for n in range(57):
    if plabels[n] == rlabels[n]:
        right += 1
    else:
        continue

print(str(right/57*100)+'%')
