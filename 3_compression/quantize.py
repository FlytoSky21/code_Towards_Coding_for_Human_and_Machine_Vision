from xml.etree.ElementTree import parse
import re
import numpy as np

def quantize_svg(svg_fn):
  d = parse(svg_fn)
  s = d.getroot()[0].get('d')
  pat = re.compile(r'[MLC][\-0-9][\-0-9\. ]*')
  moves = pat.findall(s)
  qmoves = []
  for m in moves:
    cmd = m[0]
    nums = m[1:]
    params = np.array(nums.split(' ')).astype(float)
    qparams = np.round(params).astype(int)
    qpstr = ' '.join([str(i) for i in qparams])
    qm = cmd+qpstr
    qmoves.append(qm)
  return ''.join(qmoves)
