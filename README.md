# Unpack Helper for DT Watch 11 Pro Clock Resources

Unpack helper for resources produced by `gen_clock/gen_clock.py` (V3 format).

## Format Layout

See `gen_clock.py::gen_clock_res`:

- **Bytes 0..7**: 8-byte magic string (`Sb@*O2GG` or `II@*24dG`)
- **Bytes 8..31**: Big-endian fields
    - `clock_id` (4 bytes)
    - `thumb_start` (4 bytes)
    - `thumb_len` (4 bytes)
    - `img_start` (4 bytes)
    - `img_len` (4 bytes)
    - `layer_start` (4 bytes)
    - `z_img_start = img_start + img_len`

- **Data blocks**: `[thumbnail][images][z_images][layer_data]`
    - Offsets to `layer_data` and `z_images` start are derived from the header

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
pip install -r dev-requirements.txt
```

- Bash
```
./create_venv.sh
source .venv/bin/activate
pip install -r dev-requirements.txt
```

## Usage example
```
Windows (PowerShell):

```powershell
python .\unpack.py .\ClockXXXXX_res
```

Unix / WSL:

```bash
python ./unpack.py ./ClockXXXXX_res
```

On success the script creates an output folder named `ClockXXXXX_res_unpacked` containing:

- Raw data: `manifest.json`, `layers.json`, chunks_raw
- Extracted resources (redy to reassemble with gen_clock tool) in chunks_decoded

