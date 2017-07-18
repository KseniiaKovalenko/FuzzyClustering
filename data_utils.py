from collections import OrderedDict
from numpy import array

def parse_data(N = 6):
    data = OrderedDict()
    inp = open('data.txt', 'r', encoding="utf8")
    while True:
        data_key = inp.readline().rstrip('\n')
        if data_key == '':
            break
        data_values = []
        for i in range(N):
            val = float(inp.readline())
            data_values.append(val)
        data[data_key] = data_values
    inp.close()
    return data

def to_nparray(ordered_data):
    return array([v for _, v in ordered_data.items()])


