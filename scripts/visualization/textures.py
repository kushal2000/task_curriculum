"""Procedural, tileable textures for presentation renders (denim, linen napkin, oak, floor).

    .venv_isaaclab3/bin/python scripts/visualization/textures.py      # writes assets/textures/*.png

Generated rather than downloaded so the renders are reproducible from the repo alone and carry no
third-party image licences. Every texture tiles: noise is built from periodic sinusoids / wrapped
FFT noise, so a sheet or table that repeats it shows no seam.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

TEXTURE_DIR = Path(__file__).resolve().parents[2] / "assets" / "textures"


def _fft_noise(n: int, scale: float, rng, aniso=(1.0, 1.0)) -> np.ndarray:
    """Tileable smooth noise in [-1, 1]: white noise low-passed in Fourier space (wraps exactly)."""
    k = np.fft.fftfreq(n)
    kx, ky = np.meshgrid(k * aniso[0], k * aniso[1])
    filt = np.exp(-(kx**2 + ky**2) * (scale**2) * 50.0)
    out = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((n, n))) * filt))
    return out / (np.abs(out).max() + 1e-9)


def _to_u8(rgb: np.ndarray) -> np.ndarray:
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


def denim(n: int = 1024, threads: int = 96, seed: int = 0) -> np.ndarray:
    """3/1 right-hand twill: indigo warp floats over a white weft, with slub and fade."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n] / n * threads  # thread coordinates
    xi, yi = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - xi, y - yi
    # Warp (vertical, indigo) is on top 3 of every 4 picks, stepping one thread per pick -> diagonal.
    warp_on_top = ((xi - yi) % 4) != 0
    # Thread cross-section shading: round yarns are darker at their edges.
    warp_shade = 0.55 + 0.45 * np.sin(np.pi * fx) ** 0.6
    weft_shade = 0.55 + 0.45 * np.sin(np.pi * fy) ** 0.6
    # Per-warp-thread dye variation (ring-dyed yarn) and slub along each thread.
    per_thread = rng.normal(0.0, 0.08, threads)[xi % threads]
    slub = 0.10 * _fft_noise(n, 0.004, rng, aniso=(1.0, 0.08))
    indigo = np.array([0.075, 0.115, 0.25])
    weft = np.array([0.52, 0.55, 0.62])
    w = (warp_shade * (1.0 + per_thread + slub))[..., None] * indigo * 1.3
    # The weft only peeks out between warp floats, so it is half-hidden in indigo fuzz.
    f = weft_shade[..., None] * (0.55 * weft + 0.45 * indigo)
    rgb = np.where(warp_on_top[..., None], w, f)
    # Large-scale fade (worn patches) and fine fibre noise.
    fade = 0.5 + 0.5 * _fft_noise(n, 0.02, rng)
    rgb = rgb * (0.9 + 0.12 * fade[..., None]) + 0.04 * fade[..., None]
    rgb += 0.025 * rng.standard_normal((n, n, 1))
    return _to_u8(rgb)


def linen(n: int = 1024, seed: int = 1) -> np.ndarray:
    """A cream linen napkin with a red woven border check (reads as a cloth, shows every fold)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n] / n
    base = np.array([0.93, 0.89, 0.80])
    # Plain weave micro-texture.
    t = 160
    weave = 0.04 * np.sin(2 * np.pi * x * t) * np.sin(2 * np.pi * y * t)
    slub = 0.05 * _fft_noise(n, 0.003, rng, aniso=(1.0, 0.1)) + 0.05 * _fft_noise(n, 0.003, rng, aniso=(0.1, 1.0))
    rgb = base * (1.0 + weave + slub)[..., None]
    # Gingham-style check: red bands every 1/8, double stripe near the hem.
    red = np.array([0.72, 0.12, 0.13])
    def bands(c):
        d = np.abs(((c * 8) % 1.0) - 0.5)
        return (d > 0.40).astype(float) * 0.55
    hem = lambda c: ((np.minimum(c, 1 - c) > 0.035) & (np.minimum(c, 1 - c) < 0.06)).astype(float)
    m = np.clip(bands(x) + bands(y) + hem(x) + hem(y), 0.0, 1.0)
    rgb = rgb * (1 - m[..., None]) + (red * (1.0 + slub[..., None])) * m[..., None]
    rgb += 0.015 * rng.standard_normal((n, n, 1))
    return _to_u8(rgb)


def oak(n: int = 1024, planks: int = 4, seed: int = 2) -> np.ndarray:
    """Light oak planks running along x: growth rings, pores, plank-to-plank tone, thin seams."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n] / n
    plank = np.floor(y * planks).astype(int)
    offs = rng.uniform(0, 1, planks + 1)[plank]
    tone = rng.normal(0.0, 0.05, planks + 1)[plank]
    warp = 0.06 * _fft_noise(n, 0.01, rng, aniso=(0.05, 1.0))
    local = y * planks - plank
    rings = np.sin(2 * np.pi * (local * 14.0 + warp * 12 + offs * 7 + 0.35 * np.sin(2 * np.pi * (x + offs))
                                + 0.25 * np.sin(4 * np.pi * (x * 2 + offs))))
    rings = 0.5 + 0.5 * np.sign(rings) * np.abs(rings) ** 0.5
    pores = _fft_noise(n, 0.0015, rng, aniso=(0.05, 1.0))
    light, dark = np.array([0.80, 0.62, 0.42]), np.array([0.62, 0.43, 0.26])
    rgb = light * (1 - 0.6 * rings[..., None]) + dark * 0.6 * rings[..., None]
    rgb *= (1.0 + tone + 0.06 * pores)[..., None]
    seam = (np.abs((y * planks) % 1.0) < 0.004) | (np.abs((y * planks) % 1.0) > 0.996)
    rgb[seam] *= 0.55
    return _to_u8(rgb)


def floor(n: int = 1024, seed: int = 3) -> np.ndarray:
    """Warm grey polished concrete."""
    rng = np.random.default_rng(seed)
    base = np.array([0.56, 0.55, 0.53])
    v = 0.06 * _fft_noise(n, 0.03, rng) + 0.03 * _fft_noise(n, 0.006, rng) + 0.02 * rng.standard_normal((n, n))
    return _to_u8(base * (1.0 + v)[..., None])


def pla(n: int = 1024, seed: int = 4) -> np.ndarray:
    """Matte 3D-printed PLA plate, NEUTRAL light grey (tinted per plate by the renderer): fine
    top-layer infill lines at 45 deg, faint mottling."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n] / n
    base = np.array([0.80, 0.80, 0.80])
    lines = 0.5 + 0.5 * np.sin(2 * np.pi * (x + y) * 90)          # 0.4 mm-ish extrusion lines
    lines = 0.035 * (lines ** 2 - 0.33)
    mottle = 0.03 * _fft_noise(n, 0.02, rng) + 0.012 * rng.standard_normal((n, n))
    return _to_u8(base * (1.0 + lines + mottle)[..., None])


TEXTURES = {"pla": pla, "denim": denim, "linen": linen, "oak": oak, "floor": floor}


#: One print colour per hinged segment of the rigid sheets (render_physx.py cycles through them).
PLATE_PALETTE = [(0.86, 0.42, 0.36), (0.20, 0.60, 0.62), (0.92, 0.72, 0.26), (0.33, 0.45, 0.72),
                 (0.52, 0.70, 0.48), (0.62, 0.42, 0.66), (0.86, 0.76, 0.58), (0.45, 0.70, 0.86)]
for _j, _c in enumerate(PLATE_PALETTE):
    TEXTURES[f"pla_{_j}"] = (lambda c: lambda: _to_u8(pla() / 255.0 / 0.80 * np.asarray(c)))(_c)


def texture_path(name: str) -> Path:
    """Path of texture ``name``, generating it on first use."""
    path = TEXTURE_DIR / f"{name}.png"
    if not path.exists():
        from PIL import Image

        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(TEXTURES[name]()).save(path)
    return path


def load(name_or_path: str) -> np.ndarray:
    """An RGB uint8 array from a texture name (``denim``) or an image path."""
    from PIL import Image

    p = texture_path(name_or_path) if name_or_path in TEXTURES else Path(name_or_path)
    return np.asarray(Image.open(p).convert("RGB"))


if __name__ == "__main__":
    for name in TEXTURES:
        p = TEXTURE_DIR / f"{name}.png"
        p.unlink(missing_ok=True)
        print(texture_path(name))
