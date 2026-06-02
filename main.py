import io
import struct
import zlib
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image

BASE_DIR = Path(__file__).parent

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://besnikrunjeva.github.io",
        "http://localhost:5173",
        "http://192.168.178.73:5173",
    ],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

VALID_SIZES = {"3.5oz", "7oz", "12oz"}
USDZ_ALIGN = 64


def pack_usdz(files: list[tuple[str, bytes]]) -> bytes:
    """Repack files into a USDZ-compliant ZIP: stored (no compression), 64-byte aligned."""
    buf = io.BytesIO()
    cd = []

    for name, data in files:
        name_b = name.encode("utf-8")
        crc = zlib.crc32(data) & 0xFFFFFFFF

        # Pad extra field so data starts on a 64-byte boundary
        base = buf.tell() + 30 + len(name_b)
        extra_len = (USDZ_ALIGN - base % USDZ_ALIGN) % USDZ_ALIGN
        extra = b"\x00" * extra_len

        offset = buf.tell()

        buf.write(struct.pack("<4sHHHHHIIIHH",
            b"PK\x03\x04", 20, 0, 0, 0, 0,
            crc, len(data), len(data),
            len(name_b), extra_len,
        ))
        buf.write(name_b)
        buf.write(extra)
        buf.write(data)

        cd.append((name_b, crc, len(data), offset))

    cd_start = buf.tell()
    for name_b, crc, size, offset in cd:
        buf.write(struct.pack("<4sHHHHHHIIIHHHHHII",
            b"PK\x01\x02",
            20, 20, 0, 0, 0, 0,
            crc, size, size,
            len(name_b), 0, 0, 0, 0, 0,
            offset,
        ))
        buf.write(name_b)

    cd_end = buf.tell()
    n = len(cd)
    buf.write(struct.pack("<4sHHHHIIH",
        b"PK\x05\x06", 0, 0, n, n,
        cd_end - cd_start, cd_start, 0,
    ))

    return buf.getvalue()


def swap_usdz_texture(usdz_path: Path, new_png: bytes) -> bytes:
    """Replace the placeholder texture inside a USDZ with new_png."""
    with zipfile.ZipFile(usdz_path, "r") as zf:
        names = zf.namelist()
        files = []
        replaced = False
        for name in names:
            if not replaced and name.startswith("textures/") and name.endswith(".png"):
                files.append((name, new_png))
                replaced = True
            else:
                files.append((name, zf.read(name)))

    if not replaced:
        raise ValueError(f"No texture slot found in {usdz_path.name}")

    return pack_usdz(files)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/generate-ar")
async def generate_ar(
    canvas: UploadFile = File(...),
    size: str = Form(...),
):
    if size not in VALID_SIZES:
        raise HTTPException(400, f"Invalid size '{size}'. Must be one of {VALID_SIZES}")

    # Read the composited canvas PNG sent from the frontend
    canvas_bytes = await canvas.read()

    # Validate it's a real image
    try:
        img = Image.open(io.BytesIO(canvas_bytes))
        img.verify()
    except Exception:
        raise HTTPException(400, "Invalid image data")

    # Ensure PNG format (re-encode if needed)
    img = Image.open(io.BytesIO(canvas_bytes)).convert("RGB")
    png_buf = io.BytesIO()
    img.save(png_buf, "PNG")
    png_bytes = png_buf.getvalue()

    # Swap texture in base USDZ
    usdz_path = BASE_DIR / "models" / f"gota-{size}.usdz"
    try:
        result = swap_usdz_texture(usdz_path, png_bytes)
    except Exception as e:
        raise HTTPException(500, str(e))

    return Response(
        content=result,
        media_type="model/vnd.usdz+zip",
        headers={"Content-Disposition": f'attachment; filename="gota-{size}-branded.usdz"'},
    )
