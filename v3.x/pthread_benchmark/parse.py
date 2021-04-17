import os
import csv

from parse_file import *

def get_n_apps(benchmark):
    print benchmark
    return len(benchmark.split('.'))

def parse_recursive(change_keys, const_keys, parameters,
        parameters_to_change, benchmark, parameters_to_observe, csv_obj):
    if not change_keys:
        config = ''
        for name in const_keys:
            config += '.' + name + '_' + parameters[name]
        filename = benchmark + config
        if not os.path.exists('results/' + filename + '.out'):
            print(filename + ' not found')
        n_apps = get_n_apps(benchmark)
        observations = parse_file('results/' + filename + '.out',
                parameters_to_observe, n_apps)
        row = write_row(benchmark, const_keys, parameters,
                parameters_to_observe, observations)
        csv_obj.writerow(row)
        return
    for value in parameters_to_change[change_keys[-1]]:
        parameters[change_keys[-1]] = value
        parse_recursive(change_keys[:-1], const_keys, parameters,
                parameters_to_change, benchmark,
                parameters_to_observe, csv_obj)

def parse(parameters, parameters_to_change, benchmarks, parameters_to_observe):
    change_keys = []
    const_keys = []
    csv_file = open('observations.csv', 'w')
    csv_obj = csv.writer(csv_file)
    headers = []
    headers.append('Benchmark')
    for key in parameters_to_change:
        change_keys.append(key)
        const_keys.append(key)
        headers.append(key)
    for parameter in parameters_to_observe:
        headers.append(parameter)
    csv_obj.writerow(headers)
    for benchmark in benchmarks:
        parse_recursive(change_keys, const_keys, parameters,
                parameters_to_change, benchmark,
                parameters_to_observe, csv_obj)
    csv_file.close()

from parameters import *
parse(parameters, parameters_to_change, benchmarks, parameters_to_observe)
