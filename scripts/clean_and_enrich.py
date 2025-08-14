import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union

# Ensure we can import project modules when running as a script
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings  # type: ignore
from utils import ensure_dirs, save_jsonl  # type: ignore
from processor.helpers import normalize_whitespace, parse_float_from_text  # type: ignore


JsonVal = Union[str, int, float, Dict[str, Any], List[Any]]


def clean_summary_text(summary_text: str) -> str:
    """Clean summary text by removing JavaScript and data layer noise."""
    if not summary_text or summary_text.strip() == "not provided":
        return "not provided"
    
    # Remove JavaScript data layer content
    clean_text = re.sub(r"window\['dataLayer'\].*?;", "", summary_text, flags=re.DOTALL)
    
    # Remove other JavaScript patterns
    clean_text = re.sub(r"window\[.*?\].*?;", "", clean_text, flags=re.DOTALL)
    clean_text = re.sub(r"\{\\u[0-9a-fA-F]{4}.*?\}", "", clean_text)
    
    # Clean up remaining artifacts
    clean_text = re.sub(r"\\u[0-9a-fA-F]{4}", "", clean_text)
    clean_text = re.sub(r"\|\s*Area:\s*$", "", clean_text.strip())
    
    # Normalize whitespace
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    # If text is too short after cleaning, mark as not provided
    if len(clean_text) < 20:
        return "not provided"
    
    return clean_text


def stable_listing_id(url: str) -> str:
    if not url:
        url = ""
    return hashlib.sha1(url.encode("utf-8")).hexdigest()


def parse_price_to_pkr(price_text: str) -> Union[int, str]:
    """Parse a price text into integer PKR.

    Rules:
    - Support formats containing Crore/Lakh/Lac (case-insensitive)
    - Support plain numbers with optional commas and optional Rs/PKR label
    - If cannot parse, return "not provided"
    - KEEP UNIT AS PKR (no conversion to crore/lakh textual units)
    """
    if not price_text or (isinstance(price_text, str) and price_text.strip().lower() == "not provided"):
        return "not provided"

    text = price_text.strip().lower()
    # Remove common labels and extraneous characters, but keep words for crore/lakh detection
    text = text.replace(",", " ")
    text = re.sub(r"pk(?:r)?|rs\.?|rupees|price|approximately|approx\.?", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()

    # Handle Crore / Lakh / Lac variants
    crore_pat = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*(crore|cr)", re.I)
    lakh_pat = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*(lakh|lac)", re.I)
    both_pat = re.compile(
        r"(?:(?P<crore>[0-9]+(?:\.[0-9]+)?)\s*(?:crore|cr))?\s*(?:(?P<lakh>[0-9]+(?:\.[0-9]+)?)\s*(?:lakh|lac))?",
        re.I,
    )

    # Case: combined like "1 crore 25 lakh"
    m_both = both_pat.fullmatch(text)
    if m_both and (m_both.group("crore") or m_both.group("lakh")):
        crore_val = float(m_both.group("crore")) if m_both.group("crore") else 0.0
        lakh_val = float(m_both.group("lakh")) if m_both.group("lakh") else 0.0
        pkr = int(round(crore_val * 10_000_000 + lakh_val * 100_000))
        return pkr

    # Single crore/lakh patterns anywhere
    m_crore = crore_pat.search(text)
    if m_crore:
        num = float(m_crore.group(1))
        return int(round(num * 10_000_000))

    m_lakh = lakh_pat.search(text)
    if m_lakh:
        num = float(m_lakh.group(1))
        return int(round(num * 100_000))

    # Grouped numeric like "14,000,000" or "14 000 000"
    m_grouped = re.search(r"\b\d{1,3}(?:[\s,]\d{3})+(?:\.\d+)?\b", text)
    if m_grouped:
        digits_only = re.sub(r"[^0-9]", "", m_grouped.group(0))
        try:
            return int(digits_only)
        except Exception:
            pass

    # Plain numeric (single token)
    m_plain = re.search(r"\b\d+(?:\.\d+)?\b", text)
    if m_plain:
        num_s = m_plain.group(0)
        try:
            if "." in num_s:
                return int(round(float(num_s)))
            return int(num_s)
        except Exception:
            pass

    return "not provided"


def parse_area(area_text: str, property_details: Dict[str, Any]) -> Tuple[str, Union[float, str], Union[float, str]]:
    """Return (unit, value, area_in_marla).

    - unit: one of marla|kanal|sqft|unknown|not provided
    - value: float or "not provided"
    - area_in_marla: float or "not provided" (only kanal converted per requirement)
    """
    if (not area_text) or area_text.strip().lower() == "not provided":
        # Try property_details for area
        for k, v in property_details.items():
            if re.search(r"area", k, re.I):
                area_text = str(v)
                break

    if not area_text:
        return "not provided", "not provided", "not provided"

    text = normalize_whitespace(str(area_text)).lower()
    if not text or text == "not provided":
        return "not provided", "not provided", "not provided"

    # Unit detection
    unit: str = "unknown"
    if "kanal" in text:
        unit = "kanal"
    elif "marla" in text:
        unit = "marla"
    elif re.search(r"sq\s*\.?\s*ft|square\s*feet|sqft", text):
        unit = "sqft"

    value = parse_float_from_text(text)
    if value is None:
        return unit if unit else "unknown", "not provided", "not provided"

    area_in_marla: Union[float, str] = "not provided"
    if unit == "marla":
        area_in_marla = float(value)
    elif unit == "kanal":
        area_in_marla = float(value) * 20.0
    else:
        # sqft or unknown -> no conversion mandated
        area_in_marla = "not provided"

    return unit if unit else "unknown", float(value), area_in_marla


def parse_beds_baths(value_text: str, description: str, kind: str) -> Union[int, str]:
    """Parse integer bedrooms/bathrooms from field text or description."""
    # Try in-value text
    for source in [value_text, description]:
        if not source or source == "not provided":
            continue
        m = re.search(r"(\d{1,2})\s*(%s|%ss?)" % (kind, kind[:-1] if kind.endswith("s") else kind), source, re.I)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
        # fallback: any integer present
        m2 = re.search(r"\b(\d{1,2})\b", source)
        if m2:
            try:
                return int(m2.group(1))
            except Exception:
                pass
    return "not provided"


def normalize_property_details(details: Dict[str, Any]) -> Dict[str, JsonVal]:
    out: Dict[str, JsonVal] = {}
    for k, v in (details or {}).items():
        nk = re.sub(r"[^a-z0-9]+", "_", k.strip().lower())
        nk = nk.strip("_")
        if nk == "":
            continue
        if isinstance(v, str):
            v = normalize_whitespace(v)
            # Coerce numeric when v looks numeric
            f = parse_float_from_text(v)
            if f is not None and re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", v.replace(",", "")):
                out[nk] = f
            else:
                out[nk] = v
        else:
            out[nk] = v
    return out


def dedupe_by_url(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set = set()
    unique: List[Dict[str, Any]] = []
    for r in records:
        url = r.get("url") or ""
        if url in seen:
            continue
        seen.add(url)
        unique.append(r)
    return unique


def to_canonical(rec: Dict[str, Any]) -> Dict[str, Any]:
    url = rec.get("url") or ""
    title = rec.get("title") or "not provided"
    price_raw = rec.get("price") or "not provided"
    location = rec.get("location") or "not provided"
    description = rec.get("description") or "not provided"
    amenities_raw = rec.get("amenities")
    amenities: Union[List[str], str]
    amenities = amenities_raw if isinstance(amenities_raw, list) and amenities_raw else "not provided"
    features = rec.get("parsed_features") or {}
    agent_raw = rec.get("agent") or {}
    agent = {
        "name": (agent_raw.get("name") if isinstance(agent_raw, dict) else None) or "not provided",
        "phone": (agent_raw.get("phone") if isinstance(agent_raw, dict) else None) or "not provided",
    }
    images = rec.get("images") or []
    prop_details_raw = rec.get("property_details_raw") or {}
    prop_details = normalize_property_details(prop_details_raw)

    unit, area_value, area_in_marla = parse_area(rec.get("area") or "not provided", prop_details_raw)

    beds = parse_beds_baths(rec.get("bedrooms") or "", description, "bed")
    baths = parse_beds_baths(rec.get("bathrooms") or "", description, "bath")

    canonical: Dict[str, Any] = {
        "listing_id": stable_listing_id(url),
        "url": url,
        "title": title,
        "price_raw": price_raw,
        "price_numeric": parse_price_to_pkr(price_raw),
        "price_currency": "PKR",
        "area_raw": rec.get("area") or "not provided",
        "area_unit": unit if unit else "unknown",
        "area_value": area_value,
        "area_in_marla": area_in_marla,
        "bedrooms": beds,
        "bathrooms": baths,
        "description": description,
        "parsed_features": features,
        "property_details": prop_details,
        "amenities": amenities,
        "agent": agent,
        "images": images if isinstance(images, list) else [],
        "raw": rec,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }
    return canonical


def as_text_summary(c: Dict[str, Any], location: str) -> str:
    parts: List[str] = []
    parts.append(f"Title: {c.get('title')}")
    parts.append(f"Price: {c.get('price_raw')}")
    
    # Clean the area field to remove JavaScript noise
    area_raw = c.get("area_raw", "")
    if area_raw and area_raw != "not provided":
        clean_area = clean_summary_text(area_raw)
        if clean_area != "not provided":
            parts.append(f"Area: {clean_area}")
    
    if location and location != "not provided":
        parts.append(f"Location: {location}")
    parts.append(f"Link: {c.get('url')}")
    
    result = " | ".join(parts)
    return clean_summary_text(result)


def as_text_specs(c: Dict[str, Any]) -> str:
    parts: List[str] = []
    parts.append(f"Bedrooms: {c.get('bedrooms')}")
    parts.append(f"Bathrooms: {c.get('bathrooms')}")
    # Include a few key property details
    pd = c.get("property_details") or {}
    if isinstance(pd, dict) and pd:
        # show up to 10 k:v pairs
        kvs = []
        for i, (k, v) in enumerate(pd.items()):
            if i >= 10:
                break
            kvs.append(f"{k.replace('_', ' ').title()}: {v}")
        parts.append("Details: " + "; ".join(kvs))
    return " | ".join(parts)


def as_text_features(c: Dict[str, Any]) -> str:
    features = c.get("parsed_features") or {}
    true_flags = [k.replace("_", " ") for k, v in features.items() if v]
    amenities = c.get("amenities")
    am_text = "not provided"
    if isinstance(amenities, list) and amenities:
        am_text = ", ".join(amenities[:20])
    return "Features: " + (", ".join(true_flags) if true_flags else "not provided") + f" | Amenities: {am_text}"


def as_text_description(c: Dict[str, Any]) -> str:
    return c.get("description") or "not provided"


def as_text_agent(c: Dict[str, Any]) -> str:
    a = c.get("agent") or {}
    name = a.get("name") if isinstance(a, dict) else None
    phone = a.get("phone") if isinstance(a, dict) else None
    return f"Agent: {name or 'not provided'} | Phone: {phone or 'not provided'} | Link: {c.get('url')}"


def make_chunks(c: Dict[str, Any], base_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    listing_id = c.get("listing_id")
    title = c.get("title")
    url = c.get("url")
    price_text = c.get("price_raw")
    location = base_meta.get("location") or "not provided"

    def chunk(chunk_type: str, text: str) -> Dict[str, Any]:
        return {
            "listing_id": listing_id,
            "url": url,
            "title": title,
            "price_text": price_text,
            "location": location,
            "chunk_type": chunk_type,
            "text": text,
        }

    return [
        chunk("summary", as_text_summary(c, location)),
        chunk("specs", as_text_specs(c)),
        chunk("features", as_text_features(c)),
        chunk("description", as_text_description(c)),
        chunk("agent", as_text_agent(c)),
    ]


def run(input_path: Path, processed_out: Path, chunks_out: Path) -> Tuple[int, int, int, List[Dict[str, Any]], List[Dict[str, Any]]]:
    ensure_dirs()

    with input_path.open("r", encoding="utf-8") as f:
        raw_records: List[Dict[str, Any]] = json.load(f)

    total_raw = len(raw_records)
    unique_raw = dedupe_by_url(raw_records)

    processed: List[Dict[str, Any]] = []
    all_chunks: List[Dict[str, Any]] = []

    for r in unique_raw:
        c = to_canonical(r)
        processed.append(c)
        base_meta = {
            "location": r.get("location") or "not provided",
        }
        all_chunks.extend(make_chunks(c, base_meta))

    # Write outputs
    processed_out.parent.mkdir(parents=True, exist_ok=True)
    with processed_out.open("w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    save_jsonl(all_chunks, chunks_out)

    return total_raw, len(unique_raw), len(all_chunks), processed, all_chunks


def print_samples(processed: List[Dict[str, Any]], chunks: List[Dict[str, Any]]) -> None:
    if not processed:
        print("No processed listings to sample.")
        return
    first = processed[0]
    sample_listing = {
        "listing_id": first.get("listing_id"),
        "url": first.get("url"),
        "title": first.get("title"),
        "price_raw": first.get("price_raw"),
        "price_numeric": first.get("price_numeric"),
        "area_unit": first.get("area_unit"),
        "area_value": first.get("area_value"),
        "bedrooms": first.get("bedrooms"),
        "bathrooms": first.get("bathrooms"),
        "processed_at": first.get("processed_at"),
    }
    print("\nSample processed listing (subset):\n")
    print(json.dumps(sample_listing, ensure_ascii=False, indent=2))

    print("\nFirst 3 chunks:\n")
    for ch in chunks[:3]:
        print(json.dumps(ch, ensure_ascii=False, indent=2))


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clean, deduplicate, and chunk Zameen listings")
    parser.add_argument("--input", default="data/raw/zameen_phase7_raw.json", type=Path)
    parser.add_argument("--processed-out", default="data/processed/zameen_phase7_processed.json", type=Path)
    parser.add_argument("--chunks-out", default="data/processed/zameen_phase7_chunks.jsonl", type=Path)
    args = parser.parse_args(argv)

    total_raw, unique_cnt, chunk_cnt, processed, chunks = run(args.input, args.processed_out, args.chunks_out)

    print(
        f"Totals -> raw: {total_raw} | unique(after dedupe): {unique_cnt} | chunks: {chunk_cnt}"
    )
    print_samples(processed, chunks)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
from pathlib import Path
from typing import Dict, List, Optional

from processor.helpers import normalize_whitespace, parse_float_from_text
from utils import ensure_dirs, load_jsonl


def clean_record(rec: Dict) -> Dict:
    rec = dict(rec)
    rec["title"] = normalize_whitespace(rec.get("title"))
    rec["price_text"] = normalize_whitespace(rec.get("price_text"))
    rec["location"] = normalize_whitespace(rec.get("location"))
    rec["snippet"] = normalize_whitespace(rec.get("snippet"))

    # Derived numeric fields (best-effort)
    rec["price_numeric"] = parse_float_from_text(rec.get("price_text"))
    rec["area_numeric"] = parse_float_from_text(rec.get("area_text"))

    # Canonical RAG document text
    doc_pieces = [
        rec.get("title") or "",
        rec.get("location") or "",
        rec.get("price_text") or "",
        rec.get("beds") or "",
        rec.get("baths") or "",
        rec.get("area_text") or "",
        rec.get("snippet") or "",
        rec.get("url") or "",
    ]
    rec["text"] = normalize_whitespace(" | ".join([p for p in doc_pieces if p]))
    return rec


def dedupe(records: List[Dict]) -> List[Dict]:
    seen = set()
    out = []
    for r in records:
        key = r.get("listing_id") or r.get("url")
        if not key:
            out.append(r)
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def run(input_path: Path, output_path: Path) -> None:
    ensure_dirs()
    records = load_jsonl(input_path)
    records = [clean_record(r) for r in records]
    records = dedupe(records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for r in records:
            import json

            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} cleaned records → {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean and enrich listings")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    run(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

