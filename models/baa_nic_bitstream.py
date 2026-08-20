"""Fixed-format practical bitstream helpers for the core-v3 BAA-NIC route."""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Iterable
import zlib


MAGIC = b"BAANIC3\x00"
FORMAT_VERSION = 3
MODEL_VERSION = 1
GROUP_COUNT = 4
HEADER_SIZE = 144
SEGMENT_HEADER_SIZE = 24

_HEADER = struct.Struct("<8sHHHHIIhhhhBBH32s32s4IHH20sI")
_SEGMENT = struct.Struct("<BBHIIHHHHI")

if _HEADER.size != HEADER_SIZE:
    raise RuntimeError(f"Unexpected BAA-NIC header size: {_HEADER.size}")
if _SEGMENT.size != SEGMENT_HEADER_SIZE:
    raise RuntimeError(f"Unexpected BAA-NIC segment header size: {_SEGMENT.size}")


@dataclass(frozen=True)
class BitstreamHeader:
    height: int
    width: int
    padding: tuple[int, int, int, int]
    q_index: int
    checkpoint_sha256: str
    contract_sha256: str
    segment_lengths: tuple[int, int, int, int]
    latent_shape: tuple[int, int]
    flags: int = 0


@dataclass(frozen=True)
class ParsedSegment:
    group_id: int
    z_string: bytes
    y_string: bytes
    z_shape: tuple[int, int]
    y_shape: tuple[int, int]
    raw: bytes


def _hash_bytes(value: str | bytes | None) -> bytes:
    if value is None:
        return bytes(32)
    if isinstance(value, bytes):
        if len(value) != 32:
            raise ValueError("Hash bytes must contain exactly 32 bytes")
        return value
    normalized = value.strip().lower()
    if len(normalized) != 64:
        raise ValueError("SHA256 values must contain exactly 64 hexadecimal characters")
    try:
        return bytes.fromhex(normalized)
    except ValueError as exc:
        raise ValueError("Invalid hexadecimal SHA256 value") from exc


def pack_segment(
    group_id: int,
    z_string: bytes,
    y_string: bytes,
    z_shape: tuple[int, int],
    y_shape: tuple[int, int],
) -> bytes:
    if group_id not in range(1, GROUP_COUNT + 1):
        raise ValueError(f"group_id must be in [1, {GROUP_COUNT}]")
    payload = bytes(z_string) + bytes(y_string)
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    header = _SEGMENT.pack(
        group_id,
        0,
        FORMAT_VERSION,
        len(z_string),
        len(y_string),
        int(z_shape[0]),
        int(z_shape[1]),
        int(y_shape[0]),
        int(y_shape[1]),
        crc,
    )
    return header + payload


def parse_segment(data: bytes, offset: int = 0) -> tuple[ParsedSegment, int]:
    if len(data) - offset < SEGMENT_HEADER_SIZE:
        raise ValueError("Truncated BAA-NIC segment header")
    fields = _SEGMENT.unpack_from(data, offset)
    group_id, _flags, version, z_length, y_length, z_h, z_w, y_h, y_w, expected_crc = fields
    if version != FORMAT_VERSION:
        raise ValueError(f"Unsupported segment format version: {version}")
    if group_id not in range(1, GROUP_COUNT + 1):
        raise ValueError(f"Invalid segment group ID: {group_id}")
    end = offset + SEGMENT_HEADER_SIZE + z_length + y_length
    if end > len(data):
        raise ValueError(f"Truncated payload for group {group_id}")
    payload_start = offset + SEGMENT_HEADER_SIZE
    payload = data[payload_start:end]
    actual_crc = zlib.crc32(payload) & 0xFFFFFFFF
    if actual_crc != expected_crc:
        raise ValueError(f"CRC mismatch for group {group_id}")
    z_string = payload[:z_length]
    y_string = payload[z_length:]
    return (
        ParsedSegment(
            group_id=group_id,
            z_string=z_string,
            y_string=y_string,
            z_shape=(z_h, z_w),
            y_shape=(y_h, y_w),
            raw=data[offset:end],
        ),
        end,
    )


def pack_header(header: BitstreamHeader) -> bytes:
    if header.q_index not in range(4):
        raise ValueError("q_index must be in [0, 3]")
    if len(header.segment_lengths) != GROUP_COUNT:
        raise ValueError("Exactly four segment lengths are required")
    pad_top, pad_bottom, pad_left, pad_right = header.padding
    body = _HEADER.pack(
        MAGIC,
        FORMAT_VERSION,
        MODEL_VERSION,
        HEADER_SIZE,
        int(header.flags),
        int(header.height),
        int(header.width),
        int(pad_top),
        int(pad_bottom),
        int(pad_left),
        int(pad_right),
        int(header.q_index),
        GROUP_COUNT,
        0,
        _hash_bytes(header.checkpoint_sha256),
        _hash_bytes(header.contract_sha256),
        *(int(value) for value in header.segment_lengths),
        int(header.latent_shape[0]),
        int(header.latent_shape[1]),
        bytes(20),
        0,
    )
    crc = zlib.crc32(body[:-4]) & 0xFFFFFFFF
    return body[:-4] + struct.pack("<I", crc)


def parse_header(data: bytes) -> BitstreamHeader:
    if len(data) < HEADER_SIZE:
        raise ValueError("Truncated BAA-NIC fixed header")
    fields = _HEADER.unpack_from(data, 0)
    (
        magic,
        format_version,
        _model_version,
        header_length,
        flags,
        height,
        width,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        q_index,
        group_count,
        _reserved,
        checkpoint_hash,
        contract_hash,
        *tail,
    ) = fields
    if magic != MAGIC:
        raise ValueError("Invalid BAA-NIC magic")
    if format_version != FORMAT_VERSION or header_length != HEADER_SIZE:
        raise ValueError("Unsupported BAA-NIC header format")
    if group_count != GROUP_COUNT:
        raise ValueError(f"Expected {GROUP_COUNT} latent groups, found {group_count}")
    expected_crc = tail[-1]
    actual_crc = zlib.crc32(data[: HEADER_SIZE - 4]) & 0xFFFFFFFF
    if actual_crc != expected_crc:
        raise ValueError("BAA-NIC header CRC mismatch")
    segment_lengths = tuple(int(value) for value in tail[:GROUP_COUNT])
    latent_h, latent_w = int(tail[GROUP_COUNT]), int(tail[GROUP_COUNT + 1])
    return BitstreamHeader(
        height=int(height),
        width=int(width),
        padding=(int(pad_top), int(pad_bottom), int(pad_left), int(pad_right)),
        q_index=int(q_index),
        checkpoint_sha256=checkpoint_hash.hex(),
        contract_sha256=contract_hash.hex(),
        segment_lengths=segment_lengths,
        latent_shape=(latent_h, latent_w),
        flags=int(flags),
    )


def assemble_full_stream(header: BitstreamHeader, segments: Iterable[bytes]) -> bytes:
    segment_list = [bytes(segment) for segment in segments]
    if len(segment_list) != GROUP_COUNT:
        raise ValueError("A full stream requires exactly four segments")
    lengths = tuple(len(segment) for segment in segment_list)
    if lengths != header.segment_lengths:
        raise ValueError("Header segment lengths do not match encoded segments")
    return pack_header(header) + b"".join(segment_list)


def parse_stream(data: bytes) -> tuple[BitstreamHeader, list[ParsedSegment]]:
    raw = bytes(data)
    header = parse_header(raw)
    segments: list[ParsedSegment] = []
    seen: set[int] = set()
    offset = HEADER_SIZE
    while offset < len(raw):
        segment, offset = parse_segment(raw, offset)
        if segment.group_id in seen:
            raise ValueError(f"Duplicate group segment: {segment.group_id}")
        expected_length = header.segment_lengths[segment.group_id - 1]
        if len(segment.raw) != expected_length:
            raise ValueError(f"Segment length mismatch for group {segment.group_id}")
        seen.add(segment.group_id)
        segments.append(segment)
    if offset != len(raw):
        raise ValueError("Invalid trailing BAA-NIC bitstream bytes")
    return header, segments


def prefix_stream(full_stream: bytes, prefix_groups: int) -> bytes:
    header, segments = parse_stream(full_stream)
    if len(segments) != GROUP_COUNT or [segment.group_id for segment in segments] != [1, 2, 3, 4]:
        raise ValueError("prefix_stream requires a complete physical-order stream")
    if prefix_groups not in range(1, GROUP_COUNT + 1):
        raise ValueError(f"prefix_groups must be in [1, {GROUP_COUNT}]")
    end = HEADER_SIZE + sum(header.segment_lengths[:prefix_groups])
    return bytes(full_stream[:end])


def subset_stream(full_stream: bytes, group_ids: Iterable[int]) -> bytes:
    header, segments = parse_stream(full_stream)
    if len(segments) != GROUP_COUNT:
        raise ValueError("subset_stream requires a complete source stream")
    by_id = {segment.group_id: segment.raw for segment in segments}
    requested = [int(group_id) for group_id in group_ids]
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("group_ids must be a non-empty sequence without duplicates")
    if any(group_id not in by_id for group_id in requested):
        raise ValueError("group_ids contains an unavailable group")
    return pack_header(header) + b"".join(by_id[group_id] for group_id in requested)
