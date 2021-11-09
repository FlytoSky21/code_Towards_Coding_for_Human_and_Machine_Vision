import os
from xml.etree.ElementTree import parse
import re

import cv2
import numpy as np
import glob
from PIL import Image
from svglib.svglib import svg2rlg
import tqdm
from reportlab.graphics import renderPDF, renderPM
from quantize import quantize_svg
import time

def parse_svg(svgf):
  d = parse(svgf)
  s = d.getroot()[0].get('d')
  pat = re.compile(r'[MLC][\-0-9][\-0-9\. ]*')
  moves = pat.findall(s)
  qmoves = []
  parse_res = []
  for m in moves:
    cmd = m[0]
    nums = m[1:]
    try:
      params = np.array(nums.split(' ')).astype(float)
    except:
      import IPython
      IPython.embed()
      raise
    parse_res.append((cmd, params))
  return parse_res

def get_ref_point_line(a, b):
  va = np.array(a)
  vb = np.array(b)
  ab = va - vb
  mid = (va + vb) // 2
  vx = np.array((0, 1))
  vy = np.array((1, 0))
  if np.abs(vx.dot(ab)) > np.abs(vy.dot(ab)): # horizontal
    tp1 = mid + vy * 3
    tp2 = mid - vy * 3
  else:
    tp1 = mid + vx * 3
    tp2 = mid - vx * 3
  return tp1.astype(int), tp2.astype(int)


def get_bezier_inner_point(s, a, b, t):
  vs = np.array(s)
  va = np.array(a)
  vb = np.array(b)
  vt = np.array(t)

  mst = (vs + vt) // 2
  m1 = (va + vs) // 2
  m2 = (vb + va) // 2
  m3 = (vt + vb) // 2

  mm1 = (m1 + m2) // 2
  mm2 = (m2 + m3) // 2
  mmm = (mm1 + mm2) // 2
  v12 = mm2 - mm1

  vx = np.array((0, 1))
  vy = np.array((1, 0))

  vm = mst - mmm

  if np.abs(vx.dot(v12)) > np.abs(vy.dot(v12)): # horizontal
    vh = vy # vector y is the direction
  else:
    vh = vx

  vsign = np.sign(vh.dot(vm))
  tp = mmm + vsign * vh * 3
  return tp.astype(int)

fmt_s1 = '''<?xml version="1.0" standalone="yes"?>
<svg width="256" height="256">
<path style="stroke:#010101; fill:none;" d="'''

fmt_s2 = '''"/>
</svg>'''


def handle_image(src_fn, svg_fn, text_fn, rgb_fn, tri_fn, tri_fn_e):
  quantize_start_time = time.time()
  svgf = open(svg_fn, 'r')
  quantized_d = quantize_svg(svgf)
  dir_path = text_fn.split('\\')[0]
  if not os.path.exists(dir_path):
    os.mkdir(dir_path)
  with open(text_fn,'w') as f:
    f.write(quantized_d)
  text_fid = text_fn.split('/')[-1][:-4]
  with open(f'D:/VGGFace2/train_256/features/{text_fid}_qsvg.svg', 'w') as f:
    f.write(fmt_s1+quantized_d+fmt_s2)
  quantize_end_time = time.time()
  quantize_time = quantize_end_time-quantize_start_time

  qsvgf = open(f'D:/VGGFace2/train_256/features/{text_fid}_qsvg.svg', 'r')
  drawing = svg2rlg(qsvgf)
  sketch_im = np.array(renderPM.drawToPIL(drawing))
  resize_sketch_im = np.array(renderPM.drawToPIL(drawing).resize((256,256)))
  qsvgf.close()

  sample_start_time = time.time()
  svgf.seek(0)
  params = parse_svg(svgf)
  svgf.close()

  im = Image.open(src_fn)    #.resize((128,128), Image.ANTIALIAS)
  imd = np.array(im)

  im_mask = np.zeros((256, 256, 3), np.uint8)
  res_strs = []
  try:
    for i,p in enumerate(params):
      cmd = p[0]
      if cmd == 'L':
        pa = [params[i-1][1][-2], params[i-1][1][-1]] # x, y
        pb = p[1][-2], p[1][-1]
        tp1, tp2 = get_ref_point_line(pa, pb)
        if np.max(tp1) < 256 and np.min(tp1) >= 0:
          rgb1 = imd[tp1[1], tp1[0]] # y = tp1[1], x = tp1[0]
          bs = rgb1.astype(np.uint8).tobytes()
          res_strs.append(bs)
          im_mask[tp1[1], tp1[0]] = 255
        if np.max(tp2) < 256 and np.min(tp2) >= 0:
          rgb2 = imd[tp2[1], tp2[0]]
          bs = rgb2.astype(np.uint8).tobytes()
          res_strs.append(bs)
          im_mask[tp2[1], tp2[0]] = 255
      elif cmd == 'C':
        ps = [params[i-1][1][-2], params[i-1][1][-1]] # x, y
        pt = p[1][-2], p[1][-1]
        pa = p[1][-6], p[1][-5]
        pb = p[1][-4], p[1][-3]
        tp = get_bezier_inner_point(ps, pa, pb, pt)
        if np.max(tp) < 256 and np.min(tp) >= 0:
          rgbp = imd[tp[1], tp[0]]
          bs = rgbp.astype(np.uint8).tobytes()
          res_strs.append(bs)
          im_mask[tp[1], tp[0]] = 255
  except:
    import IPython
    IPython.embed()
    raise
  with open(rgb_fn, 'wb') as f:
    f.write(b''.join(res_strs))

  sample_end_time = time.time()
  sample_time = sample_end_time-sample_start_time

  out_img = np.zeros((256, 256*3, 3), np.uint8)

  out_img[:,0:256] = imd  #* (im_mask // 255)
  out_img[:,256:512] = sketch_im
  out_img[:,512:] = im_mask
  out_im = Image.fromarray(out_img)

  dir_path = tri_fn.split('\\')[0]
  if not os.path.exists(dir_path):
    os.mkdir(dir_path)
  out_im.save(tri_fn)

  out_img[:,0:256] = imd
  out_img[:,256:512] = sketch_im
  out_img[:,512:] = 0
  out_im = Image.fromarray(out_img)  #.resize((256*3,256),Image.BICUBIC)
  dir_path = tri_fn_e.split('\\')[0]
  if not os.path.exists(dir_path):
    os.mkdir(dir_path)
  out_im.save(tri_fn_e)

  return quantize_time,sample_time

# SVG and corresponding RGB images placed in the folders
svg_path = 'D:/VGGFace2/train_256/edges'
quantize_time_g = 0
sample_time_g = 0
for fpath,dirname,fnames in os.walk(svg_path):
  if fnames:
    src_files = fpath.replace('edges', 'imgs') + '/*.png'
    svg_files = fpath + '/*.svg'
    svgs = sorted(glob.glob(svg_files))
    srcs = sorted(glob.glob(src_files))
    for im, s in tqdm.tqdm(zip(srcs,svgs)):
      fid = s.split('/')[-1][6:-4]
      quantize_time_l,sample_time_l = handle_image(im, s, f'D:/VGGFace2/train_256/features/{fid}.txt',
                   f'D:/VGGFace2/train_256/features/{fid}.rgb',
                   f'D:/VGGFace2/train_256/HVTrainData/{fid}_ec.png',
                   f'D:/VGGFace2/train_256/MVTrainData/{fid}_e.png')
      quantize_time_g += quantize_time_l
      sample_time_g +=sample_time_l
print(f'quantize_time is {quantize_time_g}')
print(f'sample_time is {sample_time_g}')


