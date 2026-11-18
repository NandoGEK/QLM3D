# New chains

`convert.py` writes the HDF5 files again from a folder of chain files, as `run.py chain` prints
them:

```sh
uv run python convert.py <folder with the .jsonl files>
```

It reads the files `ss_*.jsonl`. Chains at other parameters than `beta` = 2 L and `lam` = 0.25
or 0.5 are left out.
