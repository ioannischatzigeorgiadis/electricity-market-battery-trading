"""Download the public DE.csv benchmark, without installing the epftoolbox package."""
from pathlib import Path
from urllib.request import urlopen, Request

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "data" / "raw" / "DE.csv"
URL = "https://zenodo.org/records/4624805/files/DE.csv"


def main() -> None:
    if TARGET.is_file() and TARGET.stat().st_size > 0:
        print(f"Already present: {TARGET}")
        return
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    tmp = TARGET.with_suffix(".csv.part")
    print(f"Downloading public historical EPEX-DE data from {URL}")
    try:
        req = Request(URL, headers={"User-Agent": "research-demo/0.1"})
        with urlopen(req, timeout=120) as response, tmp.open("wb") as out:
            while chunk := response.read(256 * 1024):
                out.write(chunk)
        with tmp.open("rb") as inp:
            header = inp.readline(2048).decode("utf-8-sig", errors="replace")
        if tmp.stat().st_size < 128 or "Price" not in header:
            raise ValueError("Downloaded response does not look like the DE.csv benchmark")
        tmp.replace(TARGET)
        print(f"Saved {TARGET} ({TARGET.stat().st_size:,} bytes)")
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        print(f"Automatic download did not work: {exc}")
        print("Manual fallback: open https://zenodo.org/records/4624805")
        print("Download DE.csv and save it as data/raw/DE.csv, then rerun the next script.")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
