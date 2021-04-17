import os
import shutil

def cleanup_recursive(change_keys, const_keys, parameters,
        parameters_to_change, benchmarks):
    if not change_keys:
        config = ''
        for name in const_keys:
            config += '.' + name + '_' + parameters[name]
        for benchmark in benchmarks:
            folder = benchmark + config
            print(folder)
            if os.path.isdir(folder):
                shutil.rmtree(folder)
        return
    for value in parameters_to_change[change_keys[-1]]:
        parameters[change_keys[-1]] = value
        cleanup_recursive(change_keys[:-1], const_keys, parameters,
                parameters_to_change, benchmarks)

def cleanup(parameters, parameters_to_change, benchmarks):
    change_keys = []
    const_keys = []
    for key in parameters_to_change:
        change_keys.append(key)
        const_keys.append(key)
    cleanup_recursive(change_keys, const_keys, parameters,
            parameters_to_change, benchmarks)

from parameters import *
print('cleaning up benchmark folders')
cleanup(parameters, parameters_to_change, benchmarks)
