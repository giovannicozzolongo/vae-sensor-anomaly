"""Download NASA C-MAPSS dataset."""

import io
import logging
import zipfile
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

CMAPSS_URLS = [
    "https://data.nasa.gov/download/7nk4-ijiu/application%2Fx-zip-compressed",
    "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip",
]
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"

EXPECTED_FILES = [
    "train_FD001.txt",
    "test_FD001.txt",
    "RUL_FD001.txt",
    "train_FD002.txt",
    "test_FD002.txt",
    "RUL_FD002.txt",
    "train_FD003.txt",
    "test_FD003.txt",
    "RUL_FD003.txt",
    "train_FD004.txt",
    "test_FD004.txt",
    "RUL_FD004.txt",
]


def download_cmapss(data_dir: Path | None = None, max_retries: int = 3) -> Path:
    """Download and extract C-MAPSS dataset.

    Args:
        data_dir: where to save. defaults to data/raw/
        max_retries: number of download attempts
    """
    data_dir = data_dir or DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    # skip if already there
    if all((data_dir / f).exists() for f in EXPECTED_FILES):
        logger.info("dataset already downloaded, skipping")
        return data_dir

    resp = None
    for url in CMAPSS_URLS:
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"trying {url[:60]}... (attempt {attempt}/{max_retries})")
                resp = requests.get(url, timeout=120)
                resp.raise_for_status()
                break
            except Exception as e:
                logger.warning(f"download failed: {e}")
                resp = None
                if attempt == max_retries:
                    break
        if resp is not None:
            break

    if resp is None:
        raise RuntimeError("failed to download from all mirrors")

    _extract_zip(resp.content, data_dir)

    # verify
    missing = [f for f in EXPECTED_FILES if not (data_dir / f).exists()]
    if missing:
        raise RuntimeError(f"missing files after extraction: {missing}")

    logger.info(f"dataset ready in {data_dir}")
    return data_dir


def _extract_zip(content: bytes, data_dir: Path):
    """Extract data files, handling nested zips (PHM mirror has one)."""
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        for member in zf.namelist():
            fname = Path(member).name
            if fname in EXPECTED_FILES:
                file_content = zf.read(member)
                (data_dir / fname).write_bytes(file_content)
                logger.info(f"  extracted {fname}")
            elif fname.endswith(".zip"):
                # nested zip (e.g. CMAPSSData.zip inside outer archive)
                logger.info(f"  found nested zip: {fname}")
                _extract_zip(zf.read(member), data_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_cmapss()
