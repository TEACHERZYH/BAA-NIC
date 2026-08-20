"""One-tile smoke test for the four-layer OpenJPEG LRCP prefix path."""

from __future__ import annotations

from pathlib import Path
import os
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from build_core_v3_openjpeg_cache import encode_image, layer_prefixes, run_codec, tensor_to_png
from core_v3_data import CoreV3FixedTileDataset


def main() -> None:
    compressor = ROOT / "tools" / "codecs" / "bin" / "opj_compress.exe"
    decompressor = ROOT / "tools" / "codecs" / "bin" / "opj_decompress.exe"
    if not compressor.is_file():
        compressor = ROOT / "tools" / "codecs" / "openjpeg-2.5.4-linux" / "bin" / "opj_compress"
        decompressor = ROOT / "tools" / "codecs" / "openjpeg-2.5.4-linux" / "bin" / "opj_decompress"
    if not compressor.is_file() or not decompressor.is_file():
        print("SKIP: local OpenJPEG tools are unavailable")
        return
    data_root = None if os.name == "nt" else Path.home() / "datasets" / "VisDrone"
    dataset = CoreV3FixedTileDataset(
        ROOT / "data" / "core_v3_visdrone_development.csv", limit=1, data_root=data_root
    )
    with TemporaryDirectory(prefix="core_v3_openjpeg_smoke_") as temporary_name:
        temporary = Path(temporary_name)
        source = temporary / "input.png"
        stream = temporary / "stream.j2k"
        tensor_to_png(dataset[0]["image"], source)
        encode_image(compressor, source, stream, [400.0, 200.0, 100.0, 50.0])
        prefixes = layer_prefixes(stream.read_bytes())
        for index, prefix in enumerate(prefixes, start=1):
            prefix_path = temporary / f"layer_{index}.j2k"
            reconstruction = temporary / f"layer_{index}.png"
            prefix_path.write_bytes(prefix)
            run_codec([str(decompressor), "-i", str(prefix_path), "-o", str(reconstruction)])
            if not reconstruction.is_file():
                raise RuntimeError("OpenJPEG did not reconstruct a layer prefix")
    print("PASS: OpenJPEG four-layer LRCP prefixes are independently decodable")


if __name__ == "__main__":
    main()
