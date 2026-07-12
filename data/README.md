# Data — CHB-MIT Scalp EEG Database

The raw recordings are **not** included in this repository (they are large and
distributed under PhysioNet's terms). Download them and place them here.

## Download

CHB-MIT Scalp EEG Database v1.0.0 — https://physionet.org/content/chbmit/1.0.0/

```bash
# ~40 GB; downloads all patients chb01 … chb24
wget -r -N -c -np https://physionet.org/files/chbmit/1.0.0/
```

Arrange (or symlink) the patient folders so that this path exists:

```
data/physionet/chb01/chb01_01.edf
data/physionet/chb01/chb01-summary.txt
...
```

Alternatively, point the code at an existing copy without moving it:

```bash
export CHBMIT_DATA=/absolute/path/to/physionet
```

`src/config.py` reads `CHBMIT_DATA` if set, otherwise defaults to
`data/physionet` under the repo root.

## Citation

Shoeb, A. (2009). *Application of Machine Learning to Epileptic Seizure Onset
Detection and Treatment.* PhD thesis, MIT. And Goldberger et al. (2000),
*PhysioBank, PhysioToolkit, and PhysioNet*, Circulation 101(23).
