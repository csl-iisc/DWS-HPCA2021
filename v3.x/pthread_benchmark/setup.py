import os
import shutil
import re

submitFile = open("submit", "w+") 

def write_to_submit_file(submitFile, benchmark, folder):
    for token in re.split('\.', benchmark):
        submitFile.write(token + ',')
    submitFile.write(folder + '\n')

def setup_recursive(change_keys, const_keys, parameters,
        parameters_to_change, benchmarks):
    if not change_keys:
        config = ''
        for name in const_keys:
            config += '.' + name + '_' + parameters[name]
        for benchmark in benchmarks:
            folder = benchmark + config
            if not os.path.isdir(folder):
                os.mkdir(folder)
            shutil.copyfile('configs/gpgpusim.config' + config,
                    folder + '/gpgpusim.config')
            if os.path.exists(folder+'/gpgpu-sim'):
                os.remove(folder+'/gpgpu-sim')
            shutil.copyfile('gpgpu-sim', folder+'/gpgpu-sim')
            os.chmod(folder+'/gpgpu-sim', 0x744)
            shutil.copyfile('config_fermi_islip.icnt',
                    folder+'/config_fermi_islip.icnt')
            write_to_submit_file(submitFile, benchmark, folder)
        return
    for value in parameters_to_change[change_keys[-1]]:
        parameters[change_keys[-1]] = value
        setup_recursive(change_keys[:-1], const_keys, parameters,
                parameters_to_change, benchmarks)

def setup(parameters, parameters_to_change, benchmarks):
    change_keys = []
    const_keys = []
    for key in parameters_to_change:
        change_keys.append(key)
        const_keys.append(key)
    setup_recursive(change_keys, const_keys, parameters,
            parameters_to_change, benchmarks)

from parameters import *
from gen_configs import *
# gen_configs creates the configs files
assert(os.path.exists('configs'))
print('setting up benchmark folders')
setup(parameters, parameters_to_change, benchmarks)
