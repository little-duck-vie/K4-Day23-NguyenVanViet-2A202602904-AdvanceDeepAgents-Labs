"""Recover the last complete JSON array from an accidentally concatenated sources.json.

Runs inside the sandbox before finalization and validation.
"""
import json
import sys

SOURCES = "/tmp/work/research/sources.json"


def main():
    try:
        with open(SOURCES, encoding="utf-8") as handle:
            text = handle.read()
        decoder = json.JSONDecoder()
        position, values = 0, []
        while position < len(text):
            while position < len(text) and text[position].isspace():
                position += 1
            if position >= len(text):
                break
            value, position = decoder.raw_decode(text, position)
            if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                values.append(value)
        if not values:
            raise ValueError("no complete JSON source array found")
        sources = values[-1]
        with open(SOURCES, "w", encoding="utf-8") as handle:
            json.dump(sources, handle, ensure_ascii=False, indent=2)
        print(f"NORMALIZED: selected last of {len(values)} complete source arrays ({len(sources)} sources)")
        return 0
    except (OSError, ValueError, TypeError) as exc:
        print(f"cannot normalize sources.json: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
