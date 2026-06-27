# External Data

Full ORSDet inference uses the official SKAO SDC1 560 MHz data. These files are
not redistributed in this repository.

Expected files:

```text
sdc1_560MHz_1000h.fits
PrimaryBeam_560MHz.fits
TrainingSet_560MHz.txt
True_560MHz.txt
```

Default local layout:

```text
ORSDet/external_data/560Mhz-1kh/
  sdc1_560MHz_1000h.fits
  PrimaryBeam_560MHz.fits
  TrainingSet_560MHz.txt
  True_560MHz.txt
```

Alternatively set:

```bash
export SDC1_RAW_DATA_DIR=/path/to/560Mhz-1kh
```

The quick release verification in `scripts/verify_release.py` does not need
these raw files. They are only needed to rebuild the full-image `fwd_res`, final
catalog, and official score from the checkpoint.
