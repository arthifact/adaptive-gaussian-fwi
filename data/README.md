# Velocity model provenance

The NumPy arrays store float32 speed in m/s with axes (depth, horizontal position). The loaders average the complete input into the chosen simulation shape. The paper uses 70 × 70 cells at 10 m, irrespective of the native coordinates below. The per-run input SHA-256 identifies the original NumPy file. Each description below was checked by comparing every velocity sample with the upstream distribution.

| Array | Native shape (z, x) | Original spacing (z, x) | Origin and conversion |
| --- | ---: | --- | --- |
| `marmousi.npy` | 751 × 2,301 | 4 m, 4 m | IFP Marmousi. Decode the XDR float32 samples in Madagascar’s `marmvel.hh` in depth-fast order. The values are already in m/s. |
| `overthrust.npy` | 187 × 801 | 25 m, 25 m | Read CD1’s big-endian float32 `overthrust.vites` as (z, y, x) = (187, 801, 801). The supplied section is `volume[:, 454, :]`, with zero-based y index 454. |
| `bp_salt.npy` | 1,911 × 5,395 | 6.25 m, 12.5 m | Exact BP 2004 velocity file `vel_z6.25m_x12.5m_exact.segy.gz`. Decode IBM floats to float32, transpose traces to (z, x), retain m/s. |
| `seam.npy` | 1,501 × 1,751 | 10 m, 20 m | SEAM Phase I elastic 2-D model subset, east-west profile at North 23,900 m. Decode the P-wave file `SEAM_Vp_Elastic_N23900.sgy` and transpose to (z, x), in m/s. |
| `sigsbee.npy` (optional local input) | 1,201 × 3,201 | 25 ft, 25 ft (7.62 m each) | SMAART Sigsbee2A true stratigraphic interval velocity. Read IEEE float32 SEG-Y traces, convert ft/s to m/s by multiplying by 0.3048 and transpose to (z, x). |

## Source credits and terms

- Marmousi was created by the Institut Français du Pétrole. The public Madagascar record documents the source and its format. [Model history](https://ahay.org/RSF/book/data/marmousi/paper.pdf) · [Source samples](https://ahay.org/data/marm/marmvel.hh)
- Overthrust is copyright 1997 Society of Exploration Geophysicists and is distributed under CC BY 4.0. Credit Aminzadeh, Brac and Kunz (1997), *SEG/EAGE 3-D Salt and Overthrust Models*. This repository uses the 2-D slice identified above. The original README and license accompany it. [CD1 and documentation](https://zenodo.org/records/23469292) · [Original notice](notices/overthrust/README) · [Full license](notices/overthrust/LICENSE)
- The BP 2004 velocity data are provided courtesy of BP. Acknowledge BP and Frederic Billette in publications and presentations that use these data. Cite F. J. Billette and S. Brandsberg-Dahl (2005), *The 2004 BP velocity benchmark*. The accompanying redistribution conditions, abstract and NumPy conversion record are included. [Distribution](https://zenodo.org/records/23469290) · [Original documentation](notices/bp2004/2004_Benchmark_READMES.pdf) · [Abstract](notices/bp2004/eage_abstract.pdf) · [Modification record](notices/bp2004/README_Modification)
- SEAM Open Data is © 2013 SEG Advanced Modeling Corporation and licensed under CC BY 4.0. This repository converts the P-wave slice from SEG-Y to NumPy. [SEG Phase I open data](https://seg.org/seam/open-data/) · [Original distribution](https://seam-open-data.s3.us-west-2.amazonaws.com/Phase+I/SEAM_I_2D_Model.zip) · [Distribution license](notices/seam/LICENSE.pdf)
- Sigsbee2A belongs to SMAART Joint Venture. Its distribution README refers to a separate Sigsbee2A Data Release Agreement. The paper comparison uses Marmousi and Overthrust. Readers who obtain the optional Sigsbee model from its distributor can import their own copy using the instructions below. [Dataset description](https://ahay.org/RSF/book/data/sigsbee/paper.pdf) · [SMAART README](https://ahay.org/data/sigsbee/sigsbee2a_readme.doc) · [SEG-Y source](https://ahay.org/data/sigsbee/sigsbee2a_stratigraphy.sgy)

## Check the input files

`SHA256SUMS` gives the checksums of the four versioned NumPy files. From this directory, run `shasum -a 256 -c SHA256SUMS`.

## Optional Sigsbee input

Obtain the upstream `sigsbee2a_stratigraphy.sgy` under the distributor’s terms. From the project root, replace the local path below. The command verifies the source file and converted bytes before writing `data/sigsbee.npy`. That local input becomes available to all four runners automatically.

```sh
uv run --frozen python - /path/to/sigsbee2a_stratigraphy.sgy <<'PY'
import hashlib
import io
from pathlib import Path
import sys

import numpy as np

raw = Path(sys.argv[1]).read_bytes()
if hashlib.sha256(raw).hexdigest() != "d100d2832a13f013a10a957902b4ba8289f6ccf63b942a04ee853b8a140453c4":
    raise ValueError("Unexpected Sigsbee2A SEG-Y file")
trace = np.dtype([("header", "V240"), ("velocity", ">f4", (1201,))])
samples = np.frombuffer(raw, dtype=trace, offset=3600)["velocity"].astype(np.float32)
velocity = np.ascontiguousarray(samples.T * np.float32(0.3048))
buffer = io.BytesIO()
np.save(buffer, velocity, allow_pickle=False)
result = buffer.getvalue()
if hashlib.sha256(result).hexdigest() != "4e95b1a3f266ecfba5220b8dea45b99af117ce78a1a26142c3aa02a1e54e1792":
    raise ValueError("Unexpected converted velocity file")
path = Path("data/sigsbee.npy")
if path.exists() and path.read_bytes() != result:
    raise FileExistsError(path)
if not path.exists():
    with path.open("xb") as stream:
        stream.write(result)
PY
```
