# Fabric photo textures

Used only for drawing cloth in the viser replay viewers (`scripts/analysis/fabric_textures.py`); the
simulation never reads them.

Each file is a 10 cm × 10 cm crop (the simulated sheet's size) from the 1k diffuse map of a Poly
Haven scan, resized to 256 px. Poly Haven gives each scan's real size, which is how the crop is
true to scale. The viewer keeps only the light/dark weave detail and recolours it to the swatch colour.

| file | Poly Haven asset | scan size | used for |
|---|---|---|---|
| `terlenka.jpg` | [terlenka](https://polyhaven.com/a/terlenka) | 26.6 × 26.6 cm | chiffon, big napkin |
| `stretch_poplin.jpg` | [stretch_poplin](https://polyhaven.com/a/stretch_poplin) | 29.1 × 28.1 cm | cotton |
| `terry_cloth.jpg` | [terry_cloth](https://polyhaven.com/a/terry_cloth) | 36.7 × 36.3 cm | towel |
| `denim_fabric_03.jpg` | [denim_fabric_03](https://polyhaven.com/a/denim_fabric_03) | 27.1 × 27.1 cm | denim |

License: CC0 (public domain). Photography by colormass, processing by Rico Cilliers.
