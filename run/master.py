import sys
import numpy as np
import subprocess
ws = np.linspace(-0.005,0.005,100)
with open('si.txt', 'r') as si:
    for w in ws:
        target = si.readline().replace('\n','')
        subprocess.Popen([
            'ssh',
            '-oStrictHostKeyChecking=no',
            'titus.senez@{}.polytechnique.fr'.format(target),
            'killall -q python; cd $(pwd); source $VIRTUAL_ENV/bin/activate; nohup python slave.py -w {}'.format(w)
        ], stdout=sys.stdout, stderr=sys.stderr)
        print(target, w)
