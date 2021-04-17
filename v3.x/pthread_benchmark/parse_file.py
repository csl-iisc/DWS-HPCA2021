import os
import shutil
import re

from parameters import *

# This method takes filename, list of observations to parse from the file
# It returns a list of values corrospoding to each observation
# We are asssuming an '=' sign separates the key and the value in the file
def parse_file(filename, parameters_to_observe, n_apps):
    if not os.path.exists(filename):
        return []
    row = []
    try:
        f = open(filename)
    except:
        return []
    contents = f.readlines()
    f.close()
    observations = {}
    for line in contents:
        for parameter in parameters_to_observe:
            if re.match(parameter, line):
                print parameter
                value = line.split('=')[1].strip()
                if value.startswith('{'):
                    value = '\t'.join(value.split('\t')[2:2 + n_apps])
                observations[parameter] = value
    try:
        for parameter in parameters_to_observe:
            print parameter, observations[parameter]
            row.append(observations[parameter])
    except:
        return []
    return row

def write_row(benchmark, const_keys, parameters, parameters_to_observe,
        observations):
    row = []
    row.append(benchmark)
    for key in const_keys:
        row.append(parameters[key])
    for observation in observations:
        row.append(observation)
    return row

