# Unpack Helper for DT Watch 11 Pro Clock Resources

Unpack helper for resources produced by `gen_clock/gen_clock.py` (V3 format).

## Format Layout

All integers are **big-endian** 32-bit unless noted.

```
Offset  Size  Field
0x00    8     Magic ASCII: "Sb@*O2GG" or "II@*24dG"
0x08    4     clock_id
0x0C    4     thumb_start
0x10    4     thumb_len
0x14    4     v3: img_start   | v1: thumb_pos (unused; typically equals thumb_start)
0x18    4     img_len
0x1C    4     layer_start
```

- **Data blocks**: `[thumbnail][images][z_images][layer_data]`
    - Offsets to `layer_data` and `z_images` start are derived from the header

## Thumbnail block

The thumbnail is stored as an **image chunk** (see “Image chunk format”) or sometimes as a raw
image blob (jpg/png/gif/bmp). The unpacker tries both patterns.

## Image blocks

- **Images** block is a concatenation of chunks (or raw images), referenced by offsets in
  `layer_data`.
- **Z-images** are a separate block located between images and layer_data. These are typically
  referenced with names prefixed `z_` in config files.

## Image chunk format (custom chunk header)

When a chunk has the 16-byte header, unpack.py interprets it as:

```
Byte  Size  Meaning
0     1     img_type
1     1     compressed flag (0/1)
2     3     payload_len (little-endian 24-bit)
5     2     height (12 bits: low 8 in byte5, high 4 in low nibble of byte6)
6     2     width  (12 bits: high nibble of byte6 + byte7)
8     8     unused/reserved (always 0 in observed files)
16    ...   payload bytes (optionally LZ4-compressed)
```

### img_type mapping (as decoded in unpack.py)

| img_type | ext | Notes |
|---------:|-----|-------|
| 3        | gif | Raw GIF payload |
| 9        | jpg | Raw JPEG payload |
| 71       | rgb | rgb8888 (BGRA) |
| 72       | rgb | rgb8565 (RGB565 + alpha byte) |
| 73       | rgb | rgb565 |
| 74       | rgb | rgb1555 (1-bit alpha + RGB) |
| 75       | bmp | index8-like payload (palette + pixels) |

If compressed = 1, payload is LZ4 (block) and must be decompressed to `payload_len` bytes.

## Layer data (`layer_data`)

Layer data is a sequence of layer records. Each layer begins with a fixed header, followed by
`num` entries whose structure depends on `drawType`/`dataType`.

### Layer header

```
int32 drawType
int32 dataType
[optional] int32 interval           (only when dataType in {52, 59, 130})
[optional] int32[] area_num          (only when dataType == 112; count is not stored)
int32 alignType
int32 x
int32 y
int32 num
```

## Script Functionality

- Validates the header and splits the file into thumbnail/images/z-images/layer blobs
- Parses `layer_data` with best-effort heuristics mirroring `gen_clock.py` packing logic
- Extracts every referenced image chunk (deduplicated) and writes:
    - Raw chunks
    - Decoded payloads for recognized image headers (jpg/gif/rgb*)
- Emits `layers.json` (parsed layers + reference table) and `manifest.json`

## Limitations

- When `dataType == 112` (area_num), the list length is unknown in the binary
    - Assumes `--area-num-count` ints (default: 4)
- Image names are not stored in the `.res` file
    - Generated filenames are synthetic

## Install deps


 - PowerShell
```
.\create_venv.ps1
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

- Bash
```
./create_venv.sh
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage example
```
Windows (PowerShell):

```
python .\unpack.py .\ClockXXXXX_res
```

Unix / WSL:

```
python ./unpack.py ./ClockXXXXX_res
```

On success the script creates an output folder named `ClockXXXXX_res_unpacked` containing:

- Raw data: `manifest.json`, `layers.json`, chunks_raw
- Extracted resources (ready to reassemble with gen_clock tool) in chunks_decoded

