import base64
import gzip
from pathlib import Path


def encode_file(path: Path) -> str:
    compressed = gzip.compress(path.read_bytes(), compresslevel=9)
    return base64.b64encode(compressed).decode("utf-8")


def build_script():
    to_encode = list(Path("gwd").rglob("*.py")) + [Path("setup.py")] + list(Path("configs").rglob("*.py"))
    file_data = {str(path): encode_file(path) for path in to_encode}
    template = Path("script_template.py").read_text("utf8")
    Path(".build/script.py").write_text(template.replace("{file_data}", str(file_data)), encoding="utf8")


if __name__ == "__main__":
    build_script()
