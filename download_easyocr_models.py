import os
import sys


def main() -> None:
    try:
        import easyocr
    except ImportError as exc:
        raise RuntimeError("easyocr must be installed before running this script") from exc

    langs_env = os.getenv("EASYOCR_LANGS", "en")
    languages = [lang.strip() for lang in langs_env.split(",") if lang.strip()]
    if not languages:
        languages = ["en"]

    # Instantiating the Reader triggers model downloads if missing.
    reader = easyocr.Reader(languages, download_enabled=True)
    # Access the detector/recognizer modules to ensure weights are loaded.
    _ = reader.detector
    _ = reader.recognizer

    print(f"✅ EasyOCR models ready for languages: {', '.join(languages)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(f"Failed to prepare EasyOCR models: {err}", file=sys.stderr)
        sys.exit(1)
