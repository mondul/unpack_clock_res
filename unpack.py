#!/usr/bin/env python3
"""
Unpack helper for resources produced by gen_clock/gen_clock.py (V3 format).
The format layout (see gen_clock.py::gen_clock_res):
- Bytes 0..7   : 8-byte magic string ('Sb@*O2GG' or 'II@*24dG')
- Bytes 8..31  : big-endian fields
    clock_id (4), thumb_start (4), thumb_len (4), img_start (4), img_len (4),
    layer_start (4). z_img_start = img_start + img_len.
- Data blocks: [thumbnail][images][z_images][layer_data]
  Offsets to layer_data (and z_images start) are derived from the header.

This script:
- Validates the header and splits the file into thumbnail/images/z-images/layer blobs.
- Parses layer_data with best-effort heuristics mirroring gen_clock.py packing logic.
- Extracts every referenced image chunk (deduped), writes raw chunks and decoded
  payloads when the embedded image header is recognized (jpg/gif/rgb*).
- Emits layers.json (parsed layers + reference table) and manifest.json.

Limitations:
- When dataType == 112 (area_num), the list length is unknown in the binary.
  We assume --area-num-count ints (default 4).
- Image names are not stored in the .res, so generated filenames are synthetic.
"""
from __future__ import annotations

import argparse
import io
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import lz4.block as lz4_block
except Exception:
    lz4_block = None

try:
    from PIL import Image
except Exception:
    Image = None


def _to_png(width: int, height: int, mode: str, raw_data: bytes, raw_mode: str = None) -> Optional[bytes]:
    if Image is None:
        return None
    buf = io.BytesIO()
    Image.frombytes(mode, (width, height), raw_data, "raw", raw_mode or mode).save(buf, format="PNG")
    return buf.getvalue()

MAGICS = {b"Sb@*O2GG", b"II@*24dG"}


@dataclass
class Header:
    magic: bytes
    clock_id: int
    thumb_start: int
    thumb_len: int
    img_start: int
    img_len: int
    z_start: int
    z_len: int
    layer_start: int
    layer_len: int


def _read_be_u32(data: bytes, offset: int) -> int:
    return struct.unpack_from(">I", data, offset)[0]


def parse_header(data: bytes) -> Header:
    if len(data) < 32:
        raise ValueError("file too small to contain header")

    magic = data[:8]
    if magic not in MAGICS:
        raise ValueError(f"unexpected magic {magic!r}; expected one of {MAGICS}")

    clock_id = _read_be_u32(data, 8)
    thumb_start = _read_be_u32(data, 12)
    thumb_len = _read_be_u32(data, 16)
    img_start = _read_be_u32(data, 20)
    img_len = _read_be_u32(data, 24)
    layer_start = _read_be_u32(data, 28)

    z_start = img_start + img_len
    z_len = layer_start - z_start
    layer_len = len(data) - layer_start

    if thumb_start + thumb_len > len(data):
        raise ValueError("thumbnail exceeds file size")
    if img_start + img_len > len(data):
        raise ValueError("image section exceeds file size")
    if z_start + z_len > len(data):
        raise ValueError("z-image section exceeds file size")
    if layer_start > len(data) or layer_len < 0:
        raise ValueError("layer section exceeds file size")

    return Header(
        magic=magic,
        clock_id=clock_id,
        thumb_start=thumb_start,
        thumb_len=thumb_len,
        img_start=img_start,
        img_len=img_len,
        z_start=z_start,
        z_len=z_len,
        layer_start=layer_start,
        layer_len=layer_len,
    )


@dataclass(eq=True, frozen=True)
class RefKey:
    kind: str  # "img" | "z_img"
    offset: int  # relative to section start
    length: int
    raw_offset: int  # value stored in layer blob


@dataclass
class Ref:
    id: int
    key: RefKey
    file_raw: Optional[str] = None
    file_decoded: Optional[str] = None
    img_type: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    header_ok: Optional[bool] = None
    layer_ids: List[int] = field(default_factory=list)


@dataclass
class LayerParseCtx:
    data: bytes
    pos: int = 0

    def remaining(self) -> int:
        return len(self.data) - self.pos

    def try_peek_i32(self, rel_offset: int = 0) -> Optional[int]:
        at = self.pos + rel_offset
        if at < 0 or at + 4 > len(self.data):
            return None
        return struct.unpack_from(">i", self.data, at)[0]

    def try_read_i32(self) -> Optional[int]:
        if self.pos + 4 > len(self.data):
            return None
        val = struct.unpack_from(">i", self.data, self.pos)[0]
        self.pos += 4
        return val

    def try_read_str30(self) -> Optional[str]:
        if self.pos + 30 > len(self.data):
            return None
        raw = self.data[self.pos : self.pos + 30]
        self.pos += 30
        return raw.split(b"\x00", 1)[0].decode(errors="ignore")

    def read_i32(self) -> int:
        if self.pos + 4 > len(self.data):
            raise ValueError("layer data truncated")
        val = struct.unpack_from(">i", self.data, self.pos)[0]
        self.pos += 4
        return val

    def read_str30(self) -> str:
        if self.pos + 30 > len(self.data):
            raise ValueError("layer data truncated while reading string")
        raw = self.data[self.pos : self.pos + 30]
        self.pos += 30
        return raw.split(b"\x00", 1)[0].decode(errors="ignore")


@dataclass
class Layer:
    drawType: int
    dataType: int
    alignType: int
    x: int
    y: int
    num: int
    start: int
    interval: Optional[int] = None
    area_num: Optional[List[int]] = None
    imgArr: List[object] = field(default_factory=list)


def _classify_ref(raw_offset: int, length: int, hdr: Header) -> Optional[RefKey]:
    if length <= 0:
        return None
    if 0 <= raw_offset <= hdr.img_len and raw_offset + length <= hdr.img_len:
        return RefKey("img", raw_offset, length, raw_offset)
    if hdr.z_start <= raw_offset and raw_offset + length <= hdr.layer_start:
        return RefKey("z_img", raw_offset - hdr.z_start, length, raw_offset)
    return None


def _looks_like_jpeg(data: bytes) -> bool:
    # Require SOI and a Start-Of-Scan marker; this rejects tiny bogus slices.
    if len(data) < 64:
        return False
    if not (data[:2] == b"\xff\xd8" and data[2:3] == b"\xff"):
        return False
    if data.find(b"\xff\xda") == -1:  # SOS
        return False
    # EOI is nice to have, but some encoders omit it; accept either.
    return True


def _looks_like_png(data: bytes) -> bool:
    if len(data) < 32:
        return False
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return False
    return data.rfind(b"IEND") != -1


def _looks_like_gif(data: bytes) -> bool:
    if len(data) < 16:
        return False
    if data[:6] not in {b"GIF87a", b"GIF89a"}:
        return False
    return data[-1:] == b"\x3b"  # trailer


def _looks_like_bmp(data: bytes) -> bool:
    if len(data) < 32:
        return False
    return data[:2] == b"BM"


def _looks_like_raw_image(data: bytes) -> bool:
    return _looks_like_jpeg(data) or _looks_like_png(data) or _looks_like_gif(data) or _looks_like_bmp(data)


def _raw_image_ext(data: bytes) -> Optional[str]:
    if _looks_like_png(data):
        return "png"
    if _looks_like_jpeg(data):
        return "jpg"
    if _looks_like_gif(data):
        return "gif"
    if _looks_like_bmp(data):
        return "bmp"
    return None


def _looks_like_custom_chunk(chunk: bytes) -> bool:
    # gen_clock chunks have a 16-byte header described in parse_chunk().
    if len(chunk) < 16:
        return False
    img_type = chunk[0]
    compressed = chunk[1] in (0, 1)
    if not compressed:
        return False
    return img_type in {3, 9, 71, 72, 73, 74, 75}


def _looks_like_ref(file_data: bytes, raw_offset: int, length: int, hdr: Header, min_len: int) -> bool:
    if length < min_len:
        return False
    key = _classify_ref(raw_offset, length, hdr)
    if key is None:
        return False

    abs_start = (hdr.img_start + key.offset) if key.kind == "img" else (hdr.z_start + key.offset)
    abs_end = abs_start + key.length
    if abs_end > len(file_data) or abs_start < 0:
        return False
    chunk = file_data[abs_start:abs_end]
    # Accept either a raw image blob (jpg/png/gif/bmp) OR a custom chunk header.
    if _looks_like_raw_image(chunk):
        return True

    if len(chunk) < 16:
        # Some watchfaces appear to store proprietary/unknown image payloads.
        # To avoid missing those entirely, accept only when the slice is large
        # enough to be unlikely a random false-positive.
        return length >= max(min_len, 256)

    img_type = chunk[0]
    compressed = chunk[1] == 1
    if img_type not in {3, 9, 71, 72, 73, 74, 75}:
        return length >= max(min_len, 256)
    if not compressed:
        # Uncompressed custom-chunk payload should itself look like the encoded format.
        payload = chunk[16:]
        if img_type == 9 and not _looks_like_jpeg(payload):
            return False
        if img_type == 3 and not _looks_like_gif(payload):
            return False
        # For rgb/indexed payloads we can't cheaply validate here.
    return True


def parse_layers(
    data: bytes,
    file_data: bytes,
    hdr: Header,
    min_len: int = 16,
    area_num_count: int = 4,
) -> Tuple[List[Layer], List[Ref]]:
    ctx = LayerParseCtx(data=data)
    layers: List[Layer] = []
    refs: List[Ref] = []
    ref_index: Dict[RefKey, int] = {}

    def register_ref(key: RefKey, layer_id: Optional[int]) -> int:
        if key in ref_index:
            ref_id = ref_index[key]
            if layer_id is not None:
                ref = refs[ref_id]
                if layer_id not in ref.layer_ids:
                    ref.layer_ids.append(layer_id)
            return ref_id
        ref_id = len(refs)
        new_ref = Ref(id=ref_id, key=key)
        if layer_id is not None:
            new_ref.layer_ids.append(layer_id)
        refs.append(new_ref)
        ref_index[key] = ref_id
        return ref_id

    def _min_bytes_for_img_arr(draw_type: int, num: int) -> int:
        if num <= 0:
            return 0
        if draw_type in {10, 15, 21}:
            return num * 16
        # Most entries are 1x int. drawType==55 has a 30-byte string at idx==2.
        if draw_type == 55 and num > 2:
            return num * 4 + 26
        return num * 4

    def _infer_area_num(ctx0: LayerParseCtx, draw_type: int) -> Optional[List[int]]:
        """Infer variable-length area_num list for dataType==112.

        gen_clock.py writes all values in config.json without storing a count.
        We infer the count by scanning candidates and choosing the first that
        yields plausible align/x/y/num and does not run past the buffer.
        """

        # After area_num: alignType, x, y, num
        # Keep the search bounded to avoid pathological files.
        max_by_size = max(0, (ctx0.remaining() - 16) // 4)
        max_candidates = min(32, max_by_size)

        def plausible(candidate_pos: int) -> bool:
            if candidate_pos + 16 > len(ctx0.data):
                return False
            align = struct.unpack_from(">i", ctx0.data, candidate_pos)[0]
            x = struct.unpack_from(">i", ctx0.data, candidate_pos + 4)[0]
            y = struct.unpack_from(">i", ctx0.data, candidate_pos + 8)[0]
            num = struct.unpack_from(">i", ctx0.data, candidate_pos + 12)[0]

            # Heuristics: alignType is small-ish, num is non-negative and not huge,
            # x/y should be within reasonable watchface coordinate ranges.
            if not (-4 <= align <= 32):
                return False
            if not (0 <= num <= 512):
                return False
            if not (-5000 <= x <= 5000 and -5000 <= y <= 5000):
                return False

            after_header = candidate_pos + 16
            rem = len(ctx0.data) - after_header
            return rem >= _min_bytes_for_img_arr(draw_type, num)

        # Prefer the user-provided count if it looks valid.
        preferred = area_num_count
        if 0 <= preferred <= max_candidates:
            cand_pos = ctx0.pos + preferred * 4
            if plausible(cand_pos):
                vals = [ctx0.try_read_i32() for _ in range(preferred)]
                if all(v is not None for v in vals):
                    return [int(v) for v in vals]  # type: ignore[arg-type]

        # Otherwise scan for a plausible split.
        for c in range(0, max_candidates + 1):
            cand_pos = ctx0.pos + c * 4
            if plausible(cand_pos):
                vals: List[int] = []
                ok = True
                for _ in range(c):
                    v = ctx0.try_read_i32()
                    if v is None:
                        ok = False
                        break
                    vals.append(v)
                if ok:
                    return vals
        return None

    while ctx.pos < len(data):
        start_pos = ctx.pos
        if ctx.pos + 24 > len(data):
            break

        draw_type = ctx.try_read_i32()
        data_type = ctx.try_read_i32()
        if draw_type is None or data_type is None:
            break
        interval: Optional[int] = None
        area_num: Optional[List[int]] = None

        if data_type in {130, 59, 52}:
            interval = ctx.try_read_i32()
            if interval is None:
                break
        if data_type == 112:
            area_num = _infer_area_num(ctx, int(draw_type))

        align_type = ctx.try_read_i32()
        x = ctx.try_read_i32()
        y = ctx.try_read_i32()
        num = ctx.try_read_i32()
        if align_type is None or x is None or y is None or num is None:
            break

        layer_id = len(layers)
        layer = Layer(
            drawType=draw_type,
            dataType=data_type,
            alignType=align_type,
            x=x,
            y=y,
            num=num,
            start=start_pos,
            interval=interval,
            area_num=area_num,
        )

        for idx in range(num):
            # drawType 71 with dataType 44/45 begins with two parameter ints
            if draw_type == 71 and idx in {0, 1}:
                v = ctx.try_read_i32()
                if v is None:
                    ctx.pos = len(data)
                    break
                layer.imgArr.append(v)
                continue

            # Structured image record with two leading ints
            if draw_type in {10, 15, 21}:
                p0 = ctx.try_read_i32()
                p1 = ctx.try_read_i32()
                raw_off = ctx.try_read_i32()
                length = ctx.try_read_i32()
                if p0 is None or p1 is None or raw_off is None or length is None:
                    ctx.pos = len(data)
                    break
                key = _classify_ref(raw_off, length, hdr)
                entry = {
                    "params": [p0, p1],
                    "ref": register_ref(key, layer_id) if key else None,
                    "raw_offset": raw_off,
                    "length": length,
                }
                layer.imgArr.append(entry)
                continue

            # Fixed string slot
            if draw_type == 55 and idx == 2:
                s = ctx.try_read_str30()
                if s is None:
                    ctx.pos = len(data)
                    break
                layer.imgArr.append(s)
                continue

            # Known integer-only slots
            if (data_type in {64, 65, 66, 67} and idx in {10, 11}) or (
                draw_type == 8 and idx in {0, 1, 2}
            ):
                v = ctx.try_read_i32()
                if v is None:
                    ctx.pos = len(data)
                    break
                layer.imgArr.append(v)
                continue

            # Heuristic: offset/length pair
            raw_off = ctx.try_read_i32()
            if raw_off is None:
                ctx.pos = len(data)
                break
            if ctx.pos + 4 <= len(data):
                length = struct.unpack_from(">i", data, ctx.pos)[0]
                if _looks_like_ref(file_data, raw_off, length, hdr, min_len):
                    ctx.pos += 4
                    key = _classify_ref(raw_off, length, hdr)
                    layer.imgArr.append({"ref": register_ref(key, layer_id), "raw_offset": raw_off, "length": length})
                    continue

            # Fallback: treat as plain int
            layer.imgArr.append(raw_off)

        layers.append(layer)

    return layers, refs


def _decode_index8(width: int, height: int, payload: bytes) -> Optional[bytes]:
    if Image is None:
        return None
    expect = 1024 + width * height
    if len(payload) < expect:
        return None
    palette = payload[:1024]
    pixels = payload[1024:1024 + width * height]
    out = bytearray(width * height * 4)
    for i, idx in enumerate(pixels):
        base = idx * 4
        b, g, r, a = palette[base: base + 4]
        o = i * 4
        out[o: o + 4] = bytes((r, g, b, a))
    return _to_png(width, height, "RGBA", bytes(out))


def _decode_rgb_payload(img_type: int, width: int, height: int, payload: bytes) -> Optional[bytes]:
    """Best-effort decode to RGBA PNG bytes for known rgb formats."""
    if img_type == 71:  # rgb8888 -> BGRA
        if len(payload) < width * height * 4:
            return None
        return _to_png(width, height, "RGBA", payload, "BGRA")

    if img_type == 72:  # rgb8565 -> 16-bit color + alpha byte
        expect = width * height * 3
        if len(payload) < expect:
            return None
        out = bytearray(width * height * 4)
        pos = 0
        for i in range(width * height):
            val = payload[pos] | (payload[pos + 1] << 8)
            a = payload[pos + 2]
            r = ((val >> 11) & 0x1F) * 255 // 31
            g = ((val >> 5) & 0x3F) * 255 // 63
            b = (val & 0x1F) * 255 // 31
            o = i * 4
            out[o : o + 4] = bytes((r, g, b, a))
            pos += 3
        return _to_png(width, height, "RGBA", bytes(out))

    if img_type == 73:  # rgb565
        expect = width * height * 2
        if len(payload) < expect:
            return None
        out = bytearray(width * height * 4)
        pos = 0
        for i in range(width * height):
            val = payload[pos] | (payload[pos + 1] << 8)
            r = ((val >> 11) & 0x1F) * 255 // 31
            g = ((val >> 5) & 0x3F) * 255 // 63
            b = (val & 0x1F) * 255 // 31
            o = i * 4
            out[o : o + 4] = bytes((r, g, b, 255))
            pos += 2
        return _to_png(width, height, "RGBA", bytes(out))

    if img_type == 74:  # rgb1555 (1-bit alpha + 15-bit color)
        expect = width * height * 2
        if len(payload) < expect:
            return None
        out = bytearray(width * height * 4)
        pos = 0
        for i in range(width * height):
            val = payload[pos] | (payload[pos + 1] << 8)
            a = 255 if (val & 0x8000) else 0
            r = ((val >> 10) & 0x1F) * 255 // 31
            g = ((val >> 5) & 0x1F) * 255 // 31
            b = (val & 0x1F) * 255 // 31
            o = i * 4
            out[o : o + 4] = bytes((r, g, b, a))
            pos += 2
        return _to_png(width, height, "RGBA", bytes(out))

    return None


def parse_chunk(chunk: bytes) -> Tuple[str, Optional[bytes], Dict[str, object]]:
    """Return (ext, payload_bytes_or_None, meta)."""
    meta: Dict[str, object] = {}
    if len(chunk) < 16:
        return "bin", None, meta

    img_type = chunk[0]
    compressed = chunk[1] == 1
    payload_len = chunk[2] | (chunk[3] << 8) | (chunk[4] << 16)
    height = chunk[5] | ((chunk[6] & 0x0F) << 8)
    width = (chunk[6] >> 4) | (chunk[7] << 4)

    meta.update({
        "img_type": img_type,
        "compressed": compressed,
        "width": width,
        "height": height,
        "declared_len": payload_len,
    })

    if img_type not in {3, 9, 71, 72, 73, 74, 75}:
        return "bin", None, meta

    payload = chunk[16:]
    if compressed:
        if lz4_block is None:
            meta["decompress_error"] = "lz4 not available (install: pip install lz4 or pip install -r requirements.txt)"
            return "bin", None, meta
        try:
            payload = lz4_block.decompress(payload, uncompressed_size=payload_len)
            meta["decompressed"] = True
        except Exception as exc:  # noqa: BLE001
            meta["decompress_error"] = str(exc)
            return "bin", None, meta
    else:
        meta["decompressed"] = False

    if payload_len and payload_len <= len(payload):
        payload = payload[:payload_len]
        meta["header_ok"] = True
    else:
        meta["header_ok"] = False

    ext_map = {3: "gif", 9: "jpg", 71: "rgb", 72: "rgb", 73: "rgb", 74: "rgb", 75: "bmp"}
    ext = ext_map.get(img_type, "bin")

    return ext, payload, meta


def extract_refs(
    data: bytes,
    hdr: Header,
    refs: List[Ref],
    layers: List[Layer],
    out_dir: Path,
) -> Dict[Tuple[int, int], Dict[str, str]]:
    img_blob = data[hdr.img_start : hdr.img_start + hdr.img_len]
    z_blob = data[hdr.z_start : hdr.z_start + hdr.z_len]

    raw_dir = out_dir / "chunks_raw"
    decoded_dir = out_dir / "chunks_decoded"
    raw_dir.mkdir(parents=True, exist_ok=True)
    decoded_dir.mkdir(parents=True, exist_ok=True)

    layer_refs: List[List[int]] = []
    for layer in layers:
        seen: set[int] = set()
        ordered: List[int] = []
        for entry in layer.imgArr:
            if isinstance(entry, dict) and isinstance(entry.get("ref"), int):
                ref_id = int(entry["ref"])
                if ref_id not in seen:
                    seen.add(ref_id)
                    ordered.append(ref_id)
        layer_refs.append(ordered)

    ref_payloads: Dict[int, Dict[str, object]] = {}
    for ref in refs:
        blob = img_blob if ref.key.kind == "img" else z_blob
        start = ref.key.offset
        end = start + ref.key.length
        if end > len(blob):
            continue
        chunk = blob[start:end]

        ext, payload, meta = parse_chunk(chunk)
        ref.img_type = meta.get("img_type")
        ref.width = meta.get("width")
        ref.height = meta.get("height")
        ref.header_ok = bool(meta.get("header_ok", False)) if meta else None

        decoded_bytes: Optional[bytes] = None
        decoded_ext: Optional[str] = None

        if payload:
            decoded_ext = ext
            if ref.img_type in {3, 9}:  # gif/jpg, already raw
                decoded_bytes = payload
            elif ref.img_type == 75:  # index8-like payload (palette + pixels)
                decoded_bytes = _decode_index8(ref.width or 0, ref.height or 0, payload)
                decoded_ext = "png" if decoded_bytes else "bmp"
                if decoded_bytes is None:
                    decoded_bytes = payload
            elif ref.img_type in {71, 72, 73, 74}:
                decoded_bytes = _decode_rgb_payload(ref.img_type, ref.width or 0, ref.height or 0, payload)
                decoded_ext = "png" if decoded_bytes else ext
        else:
            raw_ext = _raw_image_ext(chunk)
            if raw_ext:
                decoded_bytes = chunk
                decoded_ext = raw_ext

        ref_payloads[ref.id] = {
            "chunk": chunk,
            "decoded_bytes": decoded_bytes,
            "decoded_ext": decoded_ext,
        }

    file_map: Dict[Tuple[int, int], Dict[str, str]] = {}
    for layer_id, ref_ids in enumerate(layer_refs):
        layer_tag = f"layer_{layer_id:03d}"
        for local_idx, ref_id in enumerate(ref_ids):
            payload = ref_payloads.get(ref_id)
            if not payload:
                continue
            ref = refs[ref_id]
            chunk = payload["chunk"]

            raw_name = f"{layer_tag}_chunk_{local_idx:03d}_{ref.key.kind}.bin"
            raw_path = raw_dir / raw_name
            raw_path.write_bytes(chunk)
            rel_raw = str(raw_path.relative_to(out_dir))

            if ref.file_raw is None:
                ref.file_raw = rel_raw

            decoded_bytes = payload.get("decoded_bytes")
            decoded_ext = payload.get("decoded_ext")
            rel_decoded: Optional[str] = None
            if decoded_bytes and decoded_ext:
                decoded_name = f"{layer_tag}_chunk_{local_idx:03d}.{decoded_ext}"
                decoded_path = decoded_dir / decoded_name
                decoded_path.write_bytes(decoded_bytes)
                rel_decoded = str(decoded_path.relative_to(out_dir))
                if ref.file_decoded is None:
                    ref.file_decoded = rel_decoded

            entry: Dict[str, str] = {"raw": rel_raw}
            if rel_decoded:
                entry["decoded"] = rel_decoded
            file_map[(layer_id, ref_id)] = entry

    return file_map


def write_outputs(
    out_dir: Path,
    hdr: Header,
    layers: List[Layer],
    refs: List[Ref],
    layer_ref_files: Dict[Tuple[int, int], Dict[str, str]],
    thumb: bytes,
    layer_blob: bytes,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Thumbnail
    thumb_raw = out_dir / "thumbnail.bin"
    thumb_raw.write_bytes(thumb)
    ext, payload, meta = parse_chunk(thumb)
    thumb_info = {
        "raw": str(thumb_raw.relative_to(out_dir)),
        "ext": ext,
        "img_type": meta.get("img_type"),
        "width": meta.get("width"),
        "height": meta.get("height"),
        "header_ok": meta.get("header_ok"),
        "compressed": meta.get("compressed"),
        "decompress_error": meta.get("decompress_error"),
    }
    if payload:
        decoded = None
        decoded_ext = ext
        if meta.get("img_type") == 75:
            decoded = _decode_index8(meta.get("width") or 0, meta.get("height") or 0, payload)
            decoded_ext = "png" if decoded else "bmp"
            if decoded is None:
                decoded = payload
        elif meta.get("img_type") in {3, 9}:
            decoded = payload
        elif meta.get("img_type") in {71, 72, 73, 74}:
            decoded = _decode_rgb_payload(meta.get("img_type"), meta.get("width") or 0, meta.get("height") or 0, payload)
            decoded_ext = "png" if decoded else ext
        if decoded:
            thumb_decoded = out_dir / f"thumbnail.{decoded_ext}"
            thumb_decoded.write_bytes(decoded)
            thumb_info["decoded"] = str(thumb_decoded.relative_to(out_dir))

    (out_dir / "layer_data.bin").write_bytes(layer_blob)

    def header_to_dict(header: Header) -> Dict[str, object]:
        return {
            "magic": header.magic.decode(errors="ignore"),
            "clock_id": header.clock_id,
            "thumb_start": header.thumb_start,
            "thumb_len": header.thumb_len,
            "img_start": header.img_start,
            "img_len": header.img_len,
            "z_start": header.z_start,
            "z_len": header.z_len,
            "layer_start": header.layer_start,
            "layer_len": header.layer_len,
        }

    def layer_to_dict(layer: Layer) -> Dict[str, object]:
        return {
            "drawType": layer.drawType,
            "dataType": layer.dataType,
            "alignType": layer.alignType,
            "x": layer.x,
            "y": layer.y,
            "num": layer.num,
            "start": layer.start,
            "interval": layer.interval,
            "area_num": layer.area_num,
            "imgArr": layer.imgArr,
        }

    def ref_to_dict(ref: Ref) -> Dict[str, object]:
        return {
            "id": ref.id,
            "kind": ref.key.kind,
            "offset": ref.key.offset,
            "length": ref.key.length,
            "raw_offset": ref.key.raw_offset,
            "file_raw": ref.file_raw,
            "file_decoded": ref.file_decoded,
            "img_type": ref.img_type,
            "width": ref.width,
            "height": ref.height,
            "header_ok": ref.header_ok,
        }

    layers_out = {
        "layers": [layer_to_dict(layer) for layer in layers],
        "refs": [ref_to_dict(ref) for ref in refs],
    }
    (out_dir / "layers.json").write_text(json.dumps(layers_out, indent=2))

    manifest = {
        "header": header_to_dict(hdr),
        "thumbnail": thumb_info,
        "layer_bytes": len(layer_blob),
        "ref_count": len(refs),
        "layer_count": len(layers),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    _emit_config(out_dir, layers, refs, layer_ref_files)


def _emit_config(
    out_dir: Path,
    layers: List[Layer],
    refs: List[Ref],
    layer_ref_files: Dict[Tuple[int, int], Dict[str, str]],
) -> None:
    """Generate a config-like json that points at decoded chunk files.

    Because original filenames are not preserved inside the .res, the emitted
    config uses the decoded chunk filenames (chunk_XXX.*). This aims to mirror
    the structure of gen_clock's config.json so assets can be tweaked/repacked.
    """

    decoded_dir = out_dir / "chunks_decoded"
    decoded_dir.mkdir(parents=True, exist_ok=True)

    def ref_name(layer_id: int, ref_id: int) -> str:
        layer_entry = layer_ref_files.get((layer_id, ref_id))
        if layer_entry:
            path_str = layer_entry.get("decoded") or layer_entry.get("raw")
            if path_str:
                return Path(path_str).name
        ref = refs[ref_id]
        path_str = ref.file_decoded or ref.file_raw
        if not path_str:
            return f"ref_{ref_id}.bin"
        return Path(path_str).name

    config_layers: List[Dict[str, object]] = []
    for layer_id, layer in enumerate(layers):
        item: Dict[str, object] = {
            "drawType": layer.drawType,
            "dataType": layer.dataType,
            "alignType": layer.alignType,
            "x": layer.x,
            "y": layer.y,
            "num": layer.num,
        }
        if layer.interval is not None:
            item["interval"] = layer.interval
        if layer.area_num is not None:
            item["area_num"] = layer.area_num

        img_arr: List[object] = []
        for entry in layer.imgArr:
            if isinstance(entry, dict):
                if "params" in entry and "ref" in entry:
                    img_arr.append([entry["params"][0], entry["params"][1], ref_name(layer_id, entry["ref"])])
                elif "ref" in entry:
                    img_arr.append(ref_name(layer_id, entry["ref"]))
                else:
                    img_arr.append(entry)
            else:
                img_arr.append(entry)

        item["imgArr"] = img_arr
        config_layers.append(item)

    config_path = decoded_dir / "config.json"
    config_path.write_text(json.dumps(config_layers, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Unpack gen_clock V3 resource file")
    parser.add_argument("source", type=Path, help="ClockXXXXX_res file")
    parser.add_argument("-o", "--out", type=Path, help="output directory")
    parser.add_argument("--min-chunk-len", type=int, default=16, help="minimum length to treat a pair as image chunk")
    parser.add_argument("--area-num-count", type=int, default=4, help="assumed count for dataType==112 area_num list")
    args = parser.parse_args()

    if lz4_block is None:
        print("warning: python module 'lz4' is not installed; compressed chunks will not be decoded (pip install lz4)")

    data = args.source.read_bytes()
    hdr = parse_header(data)

    thumb = data[hdr.thumb_start : hdr.thumb_start + hdr.thumb_len]
    layer_blob = data[hdr.layer_start : hdr.layer_start + hdr.layer_len]

    layers, refs = parse_layers(
        layer_blob,
        file_data=data,
        hdr=hdr,
        min_len=args.min_chunk_len,
        area_num_count=args.area_num_count,
    )

    out_dir = args.out or args.source.parent / f"{args.source.name}_unpacked"
    layer_ref_files = extract_refs(data, hdr, refs, layers, out_dir)
    write_outputs(out_dir, hdr, layers, refs, layer_ref_files, thumb, layer_blob)

    print(f"done; wrote layers.json and {len(refs)} chunk(s) to {out_dir}")


if __name__ == "__main__":
    main()
