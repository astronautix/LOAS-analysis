import sys
import os
import numpy as np
import subprocess
import math
import shutil

zs = np.linspace(-2*math.pi, 2*math.pi, 100)
shutil.rmtree('../res_temp', ignore_errors=True)
os.mkdir('../res_temp')
with open('../si.txt', 'r') as si:
    for z in zs:
        target = si.readline().replace('\n','')
        subprocess.Popen([
            'ssh',
            '-oStrictHostKeyChecking=no',
            'titus.senez@{}.polytechnique.fr'.format(target),
            'killall -q python; cd {}; source {}/bin/activate; nohup python slave.py -z {}'.format(os.environ['PWD'], os.environ['VIRTUAL_ENV'], z)
        ], stdout=sys.stdout, stderr=sys.stderr)
        print(target, z)
