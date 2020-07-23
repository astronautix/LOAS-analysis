import sys
import os
import numpy as np
import subprocess
import math
import shutil

xs = np.linspace(-math.pi, math.pi, 41)
shutil.rmtree('../res_temp', ignore_errors=True)
os.mkdir('../res_temp')
with open('../si.txt', 'r') as si:
    for x in xs:
        target = si.readline().replace('\n','')
        subprocess.Popen([
            'ssh',
            '-oStrictHostKeyChecking=no',
            'titus.senez@{}.polytechnique.fr'.format(target),
            'killall -q python; cd {}; source {}/bin/activate; nohup python slave.py -x {}'.format(os.environ['PWD'], os.environ['VIRTUAL_ENV'], x)
        ], stdout=sys.stdout, stderr=sys.stderr)
        print(target, x)
