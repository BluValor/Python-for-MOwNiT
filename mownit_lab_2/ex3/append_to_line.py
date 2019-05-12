import random

with open('../graphs/facebook_combined.txt', 'r') as istr:
    with open('../graphs/facebook_combined_w.txt', 'w') as ostr:
        for line in istr:
            line = line.rstrip('\n') + ' {\'weight\':' + str(random.randint(1, 100)) + '}'
            print(line, file=ostr)