import sys
import os
import numpy as np
import subprocess
import shutil
ws = np.linspace(-0.005,0.005,100)
shutil.rmtree('../res_temp', ignore_errors=True)
os.mkdir('../res_temp')
with open('si.txt', 'r') as si:
    for w in ws:
        target = si.readline().replace('\n','')
        subprocess.Popen([
            'ssh',
            '-oStrictHostKeyChecking=no',
            'titus.senez@{}.polytechnique.fr'.format(target),
            'killall -q python; cd {}; source {}/bin/activate; nohup python slave.py -w {}'.format(os.environ['PWD'], os.environ['VIRTUAL_ENV'], w)
        ], stdout=sys.stdout, stderr=sys.stderr)
        print(target, w)
