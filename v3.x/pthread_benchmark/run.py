'''
This is a very very specific script for running benchmarks in a very particular
order
Each script is supposed to run only one benchmark because some benchmarks go
into deadlock
'''

import subprocess
import os
import shutil

def run(executable, args, files, ext):
    process = subprocess.Popen([executable, args])
    stdout, stderr = process.communicate(input)
    dir_name = 'result_' + args + '_' + ext
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    src = './'
    dst = './' + dir_name + '/'
    for f in files:
        shutil.copyfile(src+f, dst+f)

benchmarks = ['3DS', 'BLK', 'CONS', 'RED', 'MM', 'SC', 'SCAN'] 

for benchmark in benchmarks:
    run('gpgpu-sim', benchmark, ['stream0.txt', 'gpgpusim.config'], 'testing')
