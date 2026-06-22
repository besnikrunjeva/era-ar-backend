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
        "https://era-react-website.vercel.app",
        "https://eraprintpack.com",
        "https://www.eraprintpack.com",
        "http://localhost:5173",
        "https://localhost:5173",
        "http://192.168.178.163:5173",
        "https://192.168.178.163:5173",
        "http://192.168.178.73:5173",
        "https://192.168.178.73:5173",
    ],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

# Maps model_id → usdz filename in models/
VALID_MODELS = {
    "3.5oz":           "gota-3.5oz.usdz",
    "7oz":             "gota-7oz.usdz",
    "12oz":            "gota-12oz.usdz",
    "mbajtese":        "mbajtese.usdz",
    "akullore-s":      "akullore-s.usdz",
    "akullore-m":      "akullore-m.usdz",
    "kupa-supe":       "kupa-supe.usdz",
    "leter-tavoline":  "leter-tavoline.usdz",
}

USDZ_ALIGN = 64


def pack_usdz(files: list[tuple[str, bytes]]) -> bytes:
    """Repack files into a USDZ-compliant ZIP: stored (no compression), 64-byte aligned."""
    buf = io.BytesIO()
    cd = []

    for name, data in files:
        name_b = name.encode("utf-8")
        crc = zlib.crc32(data) & 0xFFFFFFFF

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
    """Replace the placeholder texture inside a USDZ with new_png.

    Prefers textures/placeholder_*.png so models with multiple textures
    (e.g. kraft photo + design slot) never replace the wrong one.
    Falls back to the first PNG if no placeholder name is found.
    """
    with zipfile.ZipFile(usdz_path, "r") as zf:
        names = zf.namelist()
        raw   = {n: zf.read(n) for n in names}

    # Pass 1: prefer explicit placeholder name
    target = next(
        (n for n in names if n.startswith("textures/placeholder") and n.endswith(".png")),
        None,
    )
    # Pass 2: fallback to first PNG
    if target is None:
        target = next(
            (n for n in names if n.startswith("textures/") and n.endswith(".png")),
            None,
        )

    if target is None:
        raise ValueError(f"No texture slot found in {usdz_path.name}")

    files = [(n, new_png if n == target else raw[n]) for n in names]
    return pack_usdz(files)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/generate-ar")
async def generate_ar(
    canvas: UploadFile = File(...),
    size: str = Form(None),    # gota: "3.5oz" / "7oz" / "12oz"
    model: str = Form(None),   # other products: "mbajtese"
):
    # Resolve model_id from whichever field was sent
    model_id = model or size
    if not model_id or model_id not in VALID_MODELS:
        raise HTTPException(400, f"Unknown model '{model_id}'. Valid: {list(VALID_MODELS)}")

    canvas_bytes = await canvas.read()

    try:
        img = Image.open(io.BytesIO(canvas_bytes))
        img.verify()
    except Exception:
        raise HTTPException(400, "Invalid image data")

    img = Image.open(io.BytesIO(canvas_bytes)).convert("RGB")
    png_buf = io.BytesIO()
    img.save(png_buf, "PNG")
    png_bytes = png_buf.getvalue()

    usdz_path = BASE_DIR / "models" / VALID_MODELS[model_id]
    try:
        result = swap_usdz_texture(usdz_path, png_bytes)
    except Exception as e:
        raise HTTPException(500, str(e))

    filename = f"{model_id}-branded.usdz"
    return Response(
        content=result,
        media_type="model/vnd.usdz+zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
