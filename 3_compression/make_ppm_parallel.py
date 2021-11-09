import subprocess as sp
from multiprocessing import Pool
import glob

def task(txt_fn):
    c = sp.Popen(['7z', 'a', '-t7z', '-m0=ppmd',
                  '-mx=9', '-ms=on', f'{txt_fn}_ppmd.7z', txt_fn, '-y'], stdout=sp.PIPE)
    ret = c.wait()
    if not ret == 0:
        raise Exception("Error in file", txt_fn)

p = Pool(4)
txt_files = sorted(glob.glob('../data/features/*.txt'))
print('Total ', len(txt_files))
for f in txt_files:
    p.apply_async(task, args=(f,))

p.close()
p.join()

import IPython
IPython.embed()
