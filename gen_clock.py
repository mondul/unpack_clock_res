#!/usr/bin/env python3

"""
DT NO.1 smart watch - watch face generator
Pure Python implementation, based on:
https://gist.github.com/dipcore/26a8d0d6508675e5815087398f14499c
Check the program entry point for arguments list
"""

from argparse import ArgumentParser
from os import path, getcwd, makedirs, listdir, linesep
from sys import argv
from re import findall as regex_search, IGNORECASE
from json import load as load_json
from PIL import Image
from struct import pack as pack_struct
from imagequant import quantize_raw_rgba_bytes
from lz4.block import compress as lz4_compress

# ------------------------------
# Global variables

g_clock_id_prefix_dict = {
  '454_454': 983040,
  '400_400': 917504,
  '466_466': 851968,
  '390_390': 786432,
  '410_502': 720896,
  '320_384': 655360,
  '320_385': 655360,
  '368_448': 589824,
  '390_450': 524288,
  '360_360': 458752,
}

# ------------------------------
def _extract_clock_id_from_src_folder(src_dir) -> int:
  """
  Extract clock id from folder name: first integer in [50000..65535].
  """
  base = path.basename(path.abspath(src_dir))
  for s in regex_search(r'\d+', base):
    try:
      v = int(s)
    except Exception:
      continue
    if 50000 <= v <= 65535:
      return v
  raise RuntimeError(f'--clock-id not provided and no id in folder name: {base} (expected 50000..65535)')

# ------------------------------
def _detect_clock_size_from_first_layer(src_dir):
  """
  Detect watchface resolution from the first layer image referenced by config.json.

  Rules:
    - Uses the *first* image in the *first* config entry (first layer).
    - Resolution must be one of the resolutions previously offered in the interactive menu.
  """
  allowed = {
    (454, 454): '454_454',
    (400, 400): '400_400',
    (466, 466): '466_466',
    (390, 390): '390_390',
    (410, 502): '410_502',
    (320, 384): '320_384',
    (320, 385): '320_385',
    (368, 448): '368_448',
    (390, 450): '390_450',
    (360, 360): '360_360',
  }

  config_path = path.join(src_dir, 'config.json')
  if not path.exists(config_path):
    raise FileNotFoundError(f'config.json not found in: {src_dir}')

  with open(config_path, 'r', encoding='utf-8') as f:
    config = load_json(f)
  if not isinstance(config, list) or not config:
    raise RuntimeError('config.json must be a non-empty JSON array')

  first = config[0]
  img_arr = first.get('imgArr', [])
  if not img_arr:
    raise RuntimeError('First config entry has empty imgArr; cannot detect resolution')

  img0 = img_arr[0]
  if isinstance(img0, list):
    img0 = img0[-1]
  if not isinstance(img0, str) or not img0:
    raise RuntimeError('First layer image reference is invalid')

  img_path = path.join(src_dir, img0)
  if not path.exists(img_path):
    raise FileNotFoundError(f'First layer image not found: {img_path}')

  with Image.open(img_path) as im:
    size = im.size

  if size not in allowed:
    allowed_str = ', '.join([f'{w}×{h}' for (w, h) in allowed.keys()])
    raise RuntimeError(f'Unsupported watchface resolution {size[0]}×{size[1]}. Allowed: {allowed_str}')

  return allowed[size]

# ------------------------------
def get_filename_list(src_dir):
  """
  Recursively collect all file names under a directory.

  :param src_dir: directory path
  :return: list of file names (lowercased)
  """
  my_list = []
  for filename in listdir(src_dir):
    filePath = path.join(src_dir, filename)
    if path.isdir(filePath):
      my_list += get_filename_list(filePath)
    else:
      my_list.append(filename.lower())
  else:
    return my_list

# ------------------------------
def check_clock(src_dir):
  """
  Validate a watchface directory and its config, counting missing assets.
  :param src_dir: watchface directory
  :return: list of file names
  """
  if not path.isdir(src_dir):
    raise FileNotFoundError('Directory does not exist! [%s] ' % src_dir)

  clock_name = path.basename(src_dir)

  config_path = path.join(src_dir, 'config.json')
  if not path.exists(config_path):
    raise FileNotFoundError('Config file missing! [%s] ' % config_path)

  try:
    with open(config_path, 'r', encoding='utf-8') as config_fd:
      config = load_json(config_fd)
  except Exception as e:
    raise RuntimeError('Config file read error! [%s] %s' % (clock_name, e))

  my_list = get_filename_list(src_dir)
  config_img_list = []
  errs = []

  for item in config:
    if len(item['imgArr']) != item['num']:
      if item['name']:
        errs.append('Watchface[%s]: [image count mismatch]: %s' % (clock_name, item['name']))
      else:
        errs.append('Watchface[%s]: [image count mismatch]' % clock_name)

    for idx, img in enumerate(item['imgArr']):
      if isinstance(img, list):
        original_img = img[-1]
        img[-1] = img[-1].lower()
        if img[-1] not in my_list:
          errs.append('Watchface[%s]: [image missing] %s' % (clock_name, img[-1]))
        elif original_img not in config_img_list:
          config_img_list.append(original_img)
      elif isinstance(img, int):
        continue
      else:
        original_img = img
        img = img.lower()
        if item['drawType'] == 55 and idx == 2:
          continue
        if img not in my_list:
          errs.append('Watchface[%s]: [image missing] %s' % (clock_name, img))
        elif original_img not in config_img_list:
          config_img_list.append(original_img)

  if len(errs):
    err_str = 'Watchface validation failed:'
    for err in errs:
      err_str += (linesep + ' • ' + err)
    raise RuntimeError(err_str)

  return config_img_list

# ------------------------------
def make_header(img_type: int, file_size: int, width: int, height: int):
  return pack_struct(
    'BBBBBBBBBBBBBBBB',
    img_type, 0,
    file_size & 0xFF, (file_size >> 8) & 0xFF, (file_size >> 16) & 0xFF,
    height & 0xFF,
    ((height >> 8) & 0x0F) | ((width & 0x0F) << 4),
    (width >> 4) & 0xFF,
    0, 0, 0, 0, 0, 0, 0, 0
  )

# ------------------------------
def rgba_palette_to_bgra256(palette_rgba: list[int]) -> bytes:
  """
  imagequant returns palette as a flat list [R,G,B,A, R,G,B,A, ...].
  We pad/truncate to 256 entries and convert to BGRA8888 bytes.
  """
  if len(palette_rgba) % 4 != 0:
    raise RuntimeError("Palette length is not a multiple of 4")

  n_colors = len(palette_rgba) // 4
  if n_colors > 256:
    palette_rgba = palette_rgba[: 256 * 4]
    n_colors = 256

  # Pad up to 256 colors with transparent black
  if n_colors < 256:
    palette_rgba = palette_rgba + [0, 0, 0, 0] * (256 - n_colors)

  out = bytearray(256 * 4)
  for i in range(256):
    r = palette_rgba[i * 4 + 0]
    g = palette_rgba[i * 4 + 1]
    b = palette_rgba[i * 4 + 2]
    a = palette_rgba[i * 4 + 3]
    out[i * 4 + 0] = b
    out[i * 4 + 1] = g
    out[i * 4 + 2] = r
    out[i * 4 + 3] = a
  return bytes(out)

# ------------------------------
def compress_rgb(header: bytearray, payload: bytes) -> bytes:
  """Pure-Python replacement for compress_rgb.exe.

  LZ4-compress payload in rgb_data with the same 16-byte header but compressed flag set to 1.

  Requires `lz4` (pip install lz4).
  """
  if header[1] == 1:
    return bytes(header) + payload  # already compressed

  payload_len = header[2] | (header[3] << 8) | (header[4] << 16)

  if payload_len <= 0 or payload_len > len(payload):
    payload_len = len(payload)
    header[2] = payload_len & 0xFF
    header[3] = (payload_len >> 8) & 0xFF
    header[4] = (payload_len >> 16) & 0xFF

  comp = lz4_compress(payload[:payload_len], mode='high_compression', compression=12, store_size=False)
  header[1] = 1
  return bytes(header) + comp

# ------------------------------
def image_data(img_path, _compress=True):
  """
  Converts a JPG/GIF/PNG to the expected image chunk format

  :param img_path: Image file to process
  :param _compress: Flag to compress RGBA from PNG with LZ4
  : return: tuple with data bytes and its length

  Requires `imagequant` (pip install imagequant).
  """

  # Extract image extension in lowercase
  _, img_ext = path.splitext(img_path)
  img_ext = img_ext[1:].lower()

  file_size = path.getsize(img_path)

  # Process image by extension. If JPG/GIF, just add headers to the binary content

  if img_ext == 'jpg':
    img_jpg = Image.open(img_path)
    width, height = img_jpg.size
    # 9: Raw JPEG payload
    img_data = make_header(9, file_size, width, height)

    with open(img_path, 'rb') as img_f:
      img_data += img_f.read()

  elif img_ext == 'gif':
    img_gif = Image.open(img_path)
    width, height = img_gif.size
    # 3: Raw GIF payload
    img_data = make_header(3, file_size, width, height)

    with open(img_path, 'rb') as img_f:
      img_data += img_f.read()

  # For PNG the image is loaded first as pillow RGBA, then quantized and converted to BGRA8888
  elif img_ext == 'png':
    img_png = Image.open(img_path).convert('RGBA')
    width, height = img_png.size
    rgba = img_png.tobytes()  # RGBA bytes, row-major

    indices, palette = quantize_raw_rgba_bytes(rgba, width, height)

    if len(indices) != width * height:
      raise RuntimeError(f'Cannot process {img_path} - Expected {width*height} index bytes, got {len(indices)}')

    try:
      palette_bgra = rgba_palette_to_bgra256(palette)
    except Exception as e:
      raise RuntimeError(f'Cannot process {img_path} - {e}')

    if len(palette_bgra) != 1024:
      raise RuntimeError(f'Cannot process {img_path} - Palette must be 1024 bytes (256*4)')

    index8_payload = palette_bgra + indices

    # Build header, 75: index8-like payload (palette + pixels)
    header = make_header(75, len(index8_payload), width, height)

    if _compress:
      try:
        img_data = compress_rgb(bytearray(header), index8_payload)
      except Exception as e:
        raise RuntimeError(f'Cannot process {img_path} - LZ4 compression failed: {e}')
    else:
      img_data = header + index8_payload

  else:
    raise RuntimeError(f'Cannot process {img_path} - Unsupported image format')

  return (img_data, len(img_data))

# ------------------------------
def gen_clock(src_folder, clock_id, face_size, thumbnail_path, is_compressed, is_idle, is_internal, out_folder):
  """
  Generate the watch face
  """

  src_dir = path.abspath(src_folder)
  if not path.isdir(src_dir):
    raise NotADirectoryError(f'Source is not a directory: {src_dir}')

  # Clock ID
  if clock_id is None:
    clock_id = _extract_clock_id_from_src_folder(src_dir)
  else:
    if not (50000 <= clock_id <= 65535):
      raise RuntimeError(f'--clock-id must be in [50000..65535], got {clock_id}')

  # Build output file name according to clock id
  out_file_name = f'Clock{clock_id}_res'

  # Detect resolution from first layer image and enforce watch face size
  clock_size = face_size if face_size is not None else _detect_clock_size_from_first_layer(src_dir)

  clock_id |= g_clock_id_prefix_dict[clock_size]
  if is_internal:
    clock_id |= 0x80000000

  print('Generating watchface %d (0x%08X)...' % (clock_id & 0xFFFF, clock_id))

  # Find thumbnail if not provided
  if thumbnail_path is None:
    findings = [f for f in listdir(src_dir) if regex_search(r'.*thumbnail.*(png|jpg|gif)$', f, IGNORECASE)]
    if not findings:
      raise FileNotFoundError(f'--thumbnail not provided and no images named like *thumbnail* in {src_dir}')
    else:
      thumbnail_path = path.join(src_dir, findings[0])

  clock_img_data = b''
  clock_img_length = 0

  clock_z_img_data = b''
  clock_z_img_length = 0

  img_objs = {}

  # Process thumbnail
  clock_thumb_data, clock_thumb_length = image_data(thumbnail_path, is_compressed)

  # Process images referenced in the config.json file
  for img in check_clock(src_dir):
    img_data, img_length = image_data(path.join(src_dir, img), is_compressed)
    if img.startswith('z_'):
      clock_z_img_data += img_data
      img_objs[img] = [clock_z_img_length, img_length]
      clock_z_img_length += img_length
    else:
      clock_img_data += img_data
      img_objs[img] = [clock_img_length, img_length]
      clock_img_length += img_length

  # Start building _res
  try:
    config_path = path.join(src_dir, 'config.json')
    with open(config_path, 'r', encoding='utf-8') as conFd:
      layer_list = load_json(conFd)
  except Exception as e:
    raise RuntimeError(f'Failed to read config file [{config_path}]: {e}')

  # 32 is the clock thumbnail start address
  clock_z_img_start = 32 + clock_thumb_length + clock_img_length
  clock_layer_data = b''

  for layer in layer_list:
    clock_layer_data += pack_struct('>i', layer['drawType'])
    clock_layer_data += pack_struct('>i', layer['dataType'])
    if layer['dataType'] in {130, 59, 52}:
      clock_layer_data += pack_struct('>i', layer['interval'])
    if layer['dataType'] in {112}:
      for _, value in enumerate(layer['area_num']):
        clock_layer_data += pack_struct('>i', value)

    clock_layer_data += pack_struct('>i', layer['alignType'])
    clock_layer_data += pack_struct('>i', layer['x'])
    clock_layer_data += pack_struct('>i', layer['y'])
    clock_layer_data += pack_struct('>i', layer['num'])

    for idx, img in enumerate(layer['imgArr']):
      if layer['drawType'] in (10, 15, 21):
        clock_layer_data += pack_struct('>i', img[0])
        clock_layer_data += pack_struct('>i', img[1])
        if img[2].startswith('z_'):
          clock_layer_data += pack_struct('>i', clock_z_img_start + img_objs[img[2]][0])
        else:
          clock_layer_data += pack_struct('>i', img_objs[img[2]][0])
        clock_layer_data += pack_struct('>i', img_objs[img[2]][1])
      elif layer['drawType'] == 55 and idx == 2:
        clock_layer_data += pack_struct('30s', img.encode())
      elif layer['dataType'] in (64, 65, 66, 67) and idx in (10, 11):
        clock_layer_data += pack_struct('>i', img)
      elif layer['drawType'] == 8 and idx in (0, 1, 2):
        clock_layer_data += pack_struct('>i', img)
      elif isinstance(img, int):
        clock_layer_data += pack_struct('>i', img)
      elif img.startswith('z_'):
        clock_layer_data += pack_struct('>i', clock_z_img_start + img_objs[img][0])
        clock_layer_data += pack_struct('>i', img_objs[img][1])
      else:
        clock_layer_data += pack_struct('>i', img_objs[img][0])
        clock_layer_data += pack_struct('>i', img_objs[img][1])

  out_dir = path.abspath(out_folder)
  makedirs(out_dir, exist_ok=True)

  out_path = path.join(out_dir, out_file_name)

  crc_str = 'II@*24dG' if is_idle else 'Sb@*O2GG'

  with open(out_path, 'wb+') as out_f:
    out_f.write(crc_str.encode())
    out_f.write(pack_struct('>I', clock_id))
    out_f.write(pack_struct('>II', 32, clock_thumb_length))

    start_addr = 32 + clock_thumb_length
    out_f.write(pack_struct('>II', start_addr, clock_img_length))
    out_f.write(pack_struct('>I', start_addr + clock_img_length + clock_z_img_length))

    out_f.write(clock_thumb_data)
    out_f.write(clock_img_data)
    out_f.write(clock_z_img_data)
    out_f.write(clock_layer_data)

  print(f'Watchface done: {out_path}')

# ------------------------------
# Program entry point
if __name__ == "__main__":
  parser = ArgumentParser(
    prog=path.basename(argv[0]),
    description='Generate ATS3085-S watchface _res from a source folder.'
  )

  parser.add_argument('--clock-id', type=int, default=None, help='Clock id (50000..65535). If omitted, extracted from src folder name')
  parser.add_argument('--face-size', default=None, choices=g_clock_id_prefix_dict, help='Watch face size. If omitted, extracted from image in the bottom (first) layer')
  parser.add_argument('--thumbnail', default=None, help='Optional thumbnail image path to embed (overrides auto-detect)')
  parser.add_argument('--no-lz4', action='store_true', help='Disable LZ4 compression (enabled by default)')
  parser.add_argument('--idle', action='store_true', help='Use idle magic string (II@*24dG) instead of default (Sb@*O2GG)')
  parser.add_argument('--internal', action='store_true', help='Create a factory watchface that will come pre-installed with the watch firmware')
  parser.add_argument('--out', default=getcwd(), help='Output directory (default: current)')
  parser.add_argument('src', metavar='source-dir', help='Source folder containing config.json and layer images')

  args = parser.parse_args()

  try:
    gen_clock(
      args.src,
      args.clock_id,
      args.face_size,
      args.thumbnail,
      not args.no_lz4,
      args.idle,
      args.internal,
      args.out
    )
  except Exception as e:
    raise SystemExit(f'❌ {e}')
