import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup  # type: ignore

# Add parent directory to path to import our local config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import settings
from scripts.utils import ensure_dirs


# -----------------------
# Logging setup
# -----------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("zameen_scraper")
logger.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(_fmt)
logger.addHandler(ch)

fh = logging.FileHandler(LOG_DIR / "scrape.log", encoding="utf-8")
fh.setFormatter(_fmt)
logger.addHandler(fh)


# -----------------------
# Configurable timings
# -----------------------
POLITE_DELAY_SEC = 1.0
PAGE_DELAY_SEC = 1.5
MAX_RETRIES = 3
TIMEOUT = getattr(settings, "requests_timeout", 20)
USER_AGENT = getattr(settings, "user_agent", "Mozilla/5.0")


# -----------------------
# Optional Selenium fallback
# -----------------------
def try_load_with_selenium(url: str, wait_seconds: float = 3.5) -> Optional[str]:
    try:
        from selenium import webdriver  # type: ignore
        from selenium.webdriver.chrome.options import Options  # type: ignore
        from selenium.webdriver.common.by import By  # type: ignore
        from selenium.webdriver.support.ui import WebDriverWait  # type: ignore
        from selenium.webdriver.support import expected_conditions as EC  # type: ignore
    except Exception:
        logger.warning("Selenium not available. Skipping JS-render fallback for %s", url)
        return None

    try:
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument(f"--user-agent={USER_AGENT}")
        driver = webdriver.Chrome(options=options)
        try:
            driver.set_page_load_timeout(TIMEOUT)
            driver.get(url)
            WebDriverWait(driver, wait_seconds).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "body"))
            )
            html = driver.page_source
            return html
        finally:
            driver.quit()
    except Exception as e:
        logger.warning("Selenium load failed for %s: %s", url, e)
        return None


# -----------------------
# HTTP helpers
# -----------------------
def http_get_with_retry(url: str) -> Optional[requests.Response]:
    headers = {"User-Agent": USER_AGENT}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=TIMEOUT)
            if resp.status_code == 200:
                return resp
            logger.warning("GET %s -> %s", url, resp.status_code)
        except Exception as e:
            logger.warning("GET error (attempt %s/%s) %s: %s", attempt, MAX_RETRIES, url, e)
        time.sleep(POLITE_DELAY_SEC * attempt)
    return None


# robots.txt checks intentionally disabled per user request.
def check_robots_allow(urls_to_check: List[str]) -> bool:  # noqa: D401
    """Disabled: always returns True."""
    return True


# -----------------------
# Extraction helpers
# -----------------------
DETAIL_LINK_PATTERNS = [
    re.compile(r"/Property/", re.I),
    re.compile(r"/Apartment/", re.I),
    re.compile(r"/House/", re.I),
]


def is_property_detail_link(href: Optional[str]) -> bool:
    if not href:
        return False
    href_l = href.lower()
    if href_l.startswith("javascript:") or href_l.startswith("mailto:") or href_l.startswith("#"):
        return False
    if not href_l.endswith(".html"):
        return False
    return any(p.search(href) for p in DETAIL_LINK_PATTERNS)


def collect_detail_links_from_listing(html: str, base_url: str) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    links: List[str] = []

    # Prefer anchors inside listing cards
    cards = soup.select("article, .c0df3811, .ef447dde, ._4e02f5fc")
    for card in cards:
        a = card.select_one("a[href]")
        if a and is_property_detail_link(a.get("href")):
            links.append(urljoin(base_url, a.get("href")))

    # Fallback: any link on page matching pattern
    if not links:
        for a in soup.select("a[href]"):
            href = a.get("href")
            if is_property_detail_link(href):
                links.append(urljoin(base_url, href))

    # Deduplicate while preserving order
    seen: Set[str] = set()
    ordered: List[str] = []
    for u in links:
        if u not in seen:
            seen.add(u)
            ordered.append(u)
    return ordered


def find_next_page_url(current_url: str, html: str) -> Optional[str]:
    soup = BeautifulSoup(html, "lxml")

    # 1) rel="next"
    link = soup.find("a", rel="next")
    if link and link.get("href"):
        return urljoin(current_url, link.get("href"))

    # 2) Pagination list items containing next
    next_link = soup.find("a", string=lambda s: isinstance(s, str) and s.strip().lower() in {"next", ">"})
    if next_link and next_link.get("href"):
        return urljoin(current_url, next_link.get("href"))

    # 3) Predictable pattern -3047-<N>.html
    m = re.search(r"(-\d+)-(\d+)\.html$", current_url)
    if m:
        prefix, page_s = m.groups()
        next_page = int(page_s) + 1
        candidate = re.sub(r"(-\d+)-(\d+)\.html$", rf"\1-{next_page}.html", current_url)
        return candidate

    return None


def extract_text(el) -> str:
    return el.get_text(" ", strip=True) if el else ""


def parse_property_details_table(soup: BeautifulSoup) -> Dict[str, str]:
    details: Dict[str, str] = {}
    # Try common table patterns
    for table in soup.select("table, ._1f3d3b1b, ._2b859f59"):
        rows = table.select("tr") or table.select(".row")
        for r in rows:
            cells = r.find_all(["td", "th", "div", "span"], recursive=True)
            if len(cells) >= 2:
                k = extract_text(cells[0])
                v = extract_text(cells[1])
                if k and v:
                    details[k] = v
    return details


def parse_agent_info(soup: BeautifulSoup) -> Dict[str, str]:
    agent_name = "not provided"
    agent_phone = "not provided"

    name_sel = [
        ".agent-name",
        "[class*='agent'] [class*='name']",
        "[class*='Agency'] [class*='name']",
        "[itemprop='name']",
    ]
    for sel in name_sel:
        el = soup.select_one(sel)
        if el and extract_text(el):
            agent_name = extract_text(el)
            break

    # Try tel links first
    tel = soup.select_one("a[href^='tel:']")
    if tel and tel.get("href"):
        agent_phone = tel.get("href").replace("tel:", "").strip() or agent_phone
    else:
        # Try other selectors
        phone_sel = [".phone", "[class*='phone']", "[class*='contact']"]
        for sel in phone_sel:
            el = soup.select_one(sel)
            if el and extract_text(el):
                agent_phone = extract_text(el)
                break

    return {"name": agent_name, "phone": agent_phone}


def parse_amenities(soup: BeautifulSoup) -> List[str]:
    # Look for sections under an Amenities heading
    amenities: List[str] = []
    for heading in soup.find_all(["h2", "h3", "h4"], string=True):
        text = heading.get_text(strip=True).lower()
        if "amenities" in text or "features" in text:
            ul = heading.find_next(["ul", "ol", "div"])
            if ul:
                for li in ul.select("li, .amenity, .feature"):
                    t = extract_text(li)
                    if t:
                        amenities.append(t)
    # Fallback: common classes
    if not amenities:
        for li in soup.select(".amenities li, .features li"):
            t = extract_text(li)
            if t:
                amenities.append(t)
    # Dedup
    seen = set()
    out = []
    for a in amenities:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


# Image scraping intentionally disabled per user request.


def derive_parsed_features(description: str) -> Dict[str, bool]:
    text = description.lower()
    features = {
        "near_park": any(kw in text for kw in ["near park", "facing park", "adjacent to park", "park facing"]),
        "corner": "corner" in text,
        "furnished": "furnished" in text,
        "gas": "gas" in text,
        "electricity": "electricity" in text or "wps" in text,
        "water": "water" in text,
        "security": "security" in text or "gated" in text,
        "lawn_garden": any(kw in text for kw in ["lawn", "garden"]),
        "basement": "basement" in text,
        "elevator": any(kw in text for kw in ["elevator", "lift"]),
        "servant_quarter": "servant" in text,
        "boring_bore": any(kw in text for kw in ["boring", "bore"]),
        "parking": any(kw in text for kw in ["parking", "car porch", "garage"]),
    }
    return features


def extract_price_from_page(soup: BeautifulSoup, html: str) -> str:
    """Extract price from various sources on the page."""
    # Method 1: Try to extract from dataLayer script
    scripts = soup.find_all('script')
    for script in scripts:
        if script.string and 'property_price' in script.string:
            match = re.search(r'"property_price":(\d+)', script.string)
            if match:
                price_num = int(match.group(1))
                return f"PKR {price_num:,}"
    
    # Method 2: Look for price in various selectors
    price_selectors = [
        'span[aria-label*="PKR"]',
        '[data-testid*="price"]',
        '.price-label',
        'span[class*="price"]',
        'div[class*="price"]',
        '[class*="price"]',
        '._2f838b10',
        '._856a5a61',
        '.price'
    ]
    
    for selector in price_selectors:
        element = soup.select_one(selector)
        if element:
            text = extract_text(element)
            if text and ('PKR' in text or 'Rs' in text or re.search(r'\d{4,}', text)):
                return text
    
    # Method 3: Search for price patterns in text
    price_pattern = re.compile(r'PKR\s*[\d,]+|Rs\.?\s*[\d,]+|\d{4,}\s*PKR', re.IGNORECASE)
    price_match = price_pattern.search(html)
    if price_match:
        return price_match.group().strip()
    
    return "not provided"


def extract_detail_fields(url: str, use_selenium: bool = False) -> Dict[str, Any]:
    resp = http_get_with_retry(url)
    html = resp.text if resp else None
    if not html and use_selenium:
        html = try_load_with_selenium(url)
    if not html:
        logger.warning("Failed to load detail page: %s", url)
        html = ""

    soup = BeautifulSoup(html, "lxml")

    title = extract_text(soup.select_one("h1, h2, ._64dc6f55")) or "not provided"
    price = extract_price_from_page(soup, html)
    location = extract_text(soup.select_one("[class*='location'], ._162e6469, ._6a1f2c3f")) or "not provided"

    # Bedrooms/Bathrooms/Kitchens/Floors - try multiple patterns
    def find_field(patterns: List[str]) -> str:
        for pat in patterns:
            el = soup.find(string=re.compile(pat, re.I))
            if el:
                # get sibling or parent value
                parent = el.parent
                if parent and parent.find_next():
                    val = extract_text(parent.find_next())
                    if val:
                        return val
        return "not provided"

    bedrooms = find_field(["bed", "bedroom"])
    bathrooms = find_field(["bath", "bathroom"])
    kitchens = find_field(["kitchen"])
    floors = find_field(["floor", "storey", "stories"])  # storey/stories alt spellings

    description_el = None
    for sel in ["[class*='description']", "._317b7c8f", "#description", "article"]:
        el = soup.select_one(sel)
        if el and extract_text(el):
            description_el = el
            break
    description = extract_text(description_el) if description_el else "not provided"

    area = "not provided"
    # Try to parse area hint from page
    area_hint = soup.find(string=re.compile(r"(marla|kanal|sq\.?\s*ft|square feet)", re.I))
    if area_hint:
        # Move to nearest value
        area_parent = area_hint.parent
        if area_parent:
            val = extract_text(area_parent)
            if val:
                area = val

    property_details_raw = parse_property_details_table(soup)
    if area == "not provided":
        for k, v in property_details_raw.items():
            if re.search(r"area", k, re.I):
                area = v
                break

    amenities = parse_amenities(soup)
    agent = parse_agent_info(soup)
    # Images intentionally disabled per user request
    images: List[str] = []

    parsed_features = derive_parsed_features(description if description != "not provided" else "")

    record: Dict[str, Any] = {
        "url": url,
        "title": title or "not provided",
        "price": price or "not provided",
        "area": area or "not provided",
        "location": location or "not provided",
        "bedrooms": bedrooms or "not provided",
        "bathrooms": bathrooms or "not provided",
        "kitchens": kitchens or "not provided",
        "floors": floors or "not provided",
        "description": description or "not provided",
        "parsed_features": parsed_features,
        "amenities": amenities if amenities else [],
        "property_details_raw": property_details_raw,
        "agent": agent,
        "images": images,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }
    return record


def count_non_empty_fields(rec: Dict[str, Any]) -> int:
    non_empty = 0
    for k, v in rec.items():
        if v is None:
            continue
        if isinstance(v, str) and v.strip().lower() == "not provided":
            continue
        if isinstance(v, (list, dict)) and len(v) == 0:
            continue
        non_empty += 1
    return non_empty


def load_progress(progress_path: Path) -> Dict[str, Any]:
    if progress_path.exists():
        try:
            with progress_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            logger.warning("Failed to load progress file, starting fresh: %s", progress_path)
    return {"results": [], "processed_urls": []}


def save_progress(progress_path: Path, results: List[Dict[str, Any]], processed: Iterable[str]) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = {"results": results, "processed_urls": list(processed)}
    with progress_path.open("w", encoding="utf-8") as f:
        json.dump(tmp, f, ensure_ascii=False, indent=2)


def scrape_all(start_url: str, max_pages: int, use_selenium: bool, progress_path: Path) -> List[Dict[str, Any]]:
    logger.info("Start scraping from %s (pages=%s)", start_url, max_pages)

    # Robots.txt check
    if not check_robots_allow([start_url]):
        logger.error("robots.txt disallows the start URL. Aborting.")
        return []

    progress = load_progress(progress_path)
    results: List[Dict[str, Any]] = progress.get("results", [])
    processed: Set[str] = set(progress.get("processed_urls", []))

    current_url = start_url
    page_num = 0
    detail_links: List[str] = []

    while current_url and page_num < max_pages:
        logger.info("Fetching listing page %s: %s", page_num + 1, current_url)
        resp = http_get_with_retry(current_url)
        if not resp:
            logger.warning("Failed to fetch listing page: %s", current_url)
            break
        html = resp.text
        page_links = collect_detail_links_from_listing(html, current_url)
        logger.info("Found %s potential detail links on page %s", len(page_links), page_num + 1)
        detail_links.extend(page_links)

        page_num += 1
        time.sleep(PAGE_DELAY_SEC)
        next_url = find_next_page_url(current_url, html)
        # If predictable pattern overshoots, double-check with robots (optional)
        current_url = next_url

    # Deduplicate detail links
    seen_links: Set[str] = set()
    ordered_links: List[str] = []
    for u in detail_links:
        if u not in seen_links:
            seen_links.add(u)
            ordered_links.append(u)

    logger.info("Total unique detail links gathered: %s", len(ordered_links))

    # Check robots for a sample property path; if disallowed, warn
    sample_check = ordered_links[:3] if ordered_links else []
    if sample_check and not check_robots_allow(sample_check):
        logger.error("robots.txt disallows property detail pages. Aborting.")
        return results

    # Visit each property detail
    for idx, url in enumerate(ordered_links, start=1):
        if url in processed:
            logger.info("[%s/%s] Skipping already processed: %s", idx, len(ordered_links), url)
            continue

        logger.info("[%s/%s] Fetching detail: %s", idx, len(ordered_links), url)
        rec = extract_detail_fields(url, use_selenium=use_selenium)
        results.append(rec)
        processed.add(url)
        save_progress(progress_path, results, processed)
        time.sleep(POLITE_DELAY_SEC)

    return results


def write_outputs(results: List[Dict[str, Any]], raw_out_path: Path, template_out_path: Path) -> None:
    raw_out_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("Wrote raw JSON array: %s (%s items)", raw_out_path, len(results))

    # Determine max-fields template
    if results:
        best = max(results, key=count_non_empty_fields)
        with template_out_path.open("w", encoding="utf-8") as f:
            json.dump(best, f, ensure_ascii=False, indent=2)
        logger.info("Wrote template with max fields: %s", template_out_path)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Scrape Zameen Phase 7 listings and details")
    parser.add_argument("--start-url", default=settings.start_url)
    parser.add_argument("--max-pages", type=int, default=8, help="Number of listing pages to traverse")
    parser.add_argument("--use-selenium", action="store_true", help="Enable Selenium fallback for JS-rendered pages")
    parser.add_argument(
        "--progress", default="data/raw/zameen_phase7_progress.json", help="Progress JSON path"
    )
    parser.add_argument(
        "--out", default="data/raw/zameen_phase7_raw.json", help="Final raw JSON array output"
    )
    parser.add_argument(
        "--template-out", default="data/raw/template_max_fields.json", help="Template output path"
    )
    args = parser.parse_args(argv)

    ensure_dirs()
    Path("data/raw").mkdir(parents=True, exist_ok=True)

    progress_path = Path(args.progress)
    out_path = Path(args.out)
    template_path = Path(args.template_out)

    results = scrape_all(args.start_url, args.max_pages, args.use_selenium, progress_path)
    write_outputs(results, out_path, template_path)

    # Print one sample listing (first element) for schema verification if exists
    if results:
        sample = results[0]
        print("\nSample listing (first element):\n")
        print(json.dumps(sample, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

