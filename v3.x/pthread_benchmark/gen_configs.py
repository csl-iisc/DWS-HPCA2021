import os
import shutil

def create_config_file_recursive(change_keys, const_keys, parameters,
        parameters_to_change):
    if not change_keys:
        filename = 'configs/gpgpusim.config'
        for name in const_keys:
            filename += '.' + name + '_' + parameters[name]
        shutil.copyfile('basic.config', filename)
        file = open(filename, 'a+')
        for key in sorted(parameters):
            value = parameters[key]
            file.write('-' + key + ' ' + value + '\n')
        file.close()
        return
    for value in parameters_to_change[change_keys[-1]]:
        parameters[change_keys[-1]] = value
        create_config_file_recursive(change_keys[:-1], const_keys, parameters,
                parameters_to_change)

def create_config_file(parameters, parameters_to_change):
    config = 'default'
    shutil.copyfile('basic.config', 'configs/gpgpusim.config.' + config)
    file = open('configs/gpgpusim.config.' + config, 'a+')
    for key in sorted(parameters):
        value = parameters[key]
        file.write(key + ' ' + value + '\n')
    file.close()
    change_keys = []
    const_keys = []
    for key in parameters_to_change:
        change_keys.append(key)
        const_keys.append(key)
    create_config_file_recursive(change_keys, const_keys, parameters,
            parameters_to_change)

from parameters import *
if not os.path.exists('configs'):
    os.makedirs('configs')
print('creating config files')
create_config_file(parameters, parameters_to_change)
