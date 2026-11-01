import sys, glob, os, re, zipfile
import pandas as pd

src = sys.argv[1] if len(sys.argv) > 1 else "CVS data"
os.makedirs("data", exist_ok=True)
for f in sorted(glob.glob(os.path.join(src, "*.zip"))):
    m = re.match(r"(jumping|standing|still|walking)-(\d+)-", os.path.basename(f))
    if not m:
        continue
    act, n = m.group(1), int(m.group(2))
    z = zipfile.ZipFile(f)
    a = pd.read_csv(z.open("Accelerometer.csv"))
    g = pd.read_csv(z.open("Gyroscope.csv"))
    assert (a["time"].values == g["time"].values).all(), f
    df = pd.DataFrame({"time": a["time"], "seconds_elapsed": a["seconds_elapsed"]})
    for c in "xyz":
        df[f"acc_{c}"] = a[c].values
    for c in "xyz":
        df[f"gyr_{c}"] = g[c].values
    df.to_csv(f"data/{act}_{n:02d}.csv", index=False)
    print(act, n, len(df))
