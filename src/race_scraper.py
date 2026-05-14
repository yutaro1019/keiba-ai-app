"""
Future race scraping and feature preparation for the web UI.

The scraper accepts a netkeiba race_id or race URL, stores the downloaded
entries in data/current_races, and enriches horses with the latest local
history features when horse_id is available.
"""
from __future__ import annotations

import gzip
import json
import os
import pickle
import re
import shutil
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, Iterable, List, Optional
from urllib.parse import urlencode, urljoin

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from predictor import DATA_PKL


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "data", "current_races")
PAYOUT_DIR = os.path.join(BASE_DIR, "data", "payouts")
ODDS_DIR = os.path.join(BASE_DIR, "data", "odds")
CACHE_VERSION = 5
MIN_REASONABLE_RUNNERS = 5
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(PAYOUT_DIR, exist_ok=True)
os.makedirs(ODDS_DIR, exist_ok=True)

REQUIRED_HISTORY_FEATURES = {
    "horse_dist_top3_rate", "horse_surface_top3_rate",
    "horse_venue_top3_rate", "horse_course_top3_rate",
    "jockey_venue_top3_rate", "jockey_course_top3_rate",
    "trainer_venue_top3_rate", "trainer_course_top3_rate",
    "closer_long_distance_fit", "front_long_distance_risk",
}
_HISTORY_CACHE: Optional[pd.DataFrame] = None

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}

VENUE_NAMES = {
    "01": "札幌",
    "02": "函館",
    "03": "福島",
    "04": "新潟",
    "05": "東京",
    "06": "中山",
    "07": "中京",
    "08": "京都",
    "09": "阪神",
    "10": "小倉",
}


@dataclass
class RaceData:
    race_id: int
    source_url: str
    fetched_at: str
    meta: Dict[str, Any]
    rows: pd.DataFrame
    cache_path: Optional[str] = None
    warnings: Optional[List[str]] = None


def race_id_from_source(source: str) -> int:
    text = str(source).strip()
    m = re.search(r"race_id=(\d{12})", text)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d{12})", text)
    if m:
        return int(m.group(1))
    raise ValueError("race_id は12桁で入力してください。")


def netkeiba_url(race_id: int) -> str:
    return f"https://race.netkeiba.com/race/shutuba.html?race_id={int(race_id)}"


def netkeiba_result_url(race_id: int) -> str:
    return f"https://race.netkeiba.com/race/result.html?race_id={int(race_id)}"


def netkeiba_race_list_url(kaisai_date: date) -> str:
    return f"https://race.netkeiba.com/top/race_list.html?kaisai_date={kaisai_date:%Y%m%d}"


def netkeiba_jra_odds_api_url() -> str:
    return "https://race.netkeiba.com/api/api_get_jra_odds.html"


def cache_path_for(race_id: int) -> str:
    return os.path.join(CACHE_DIR, f"{int(race_id)}.json")


def payout_cache_path_for(race_id: int) -> str:
    return os.path.join(PAYOUT_DIR, f"{int(race_id)}.json")


def odds_cache_path_for(race_id: int) -> str:
    return os.path.join(ODDS_DIR, f"{int(race_id)}.json")


def _clean_json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def _records_for_json(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return [
        {k: _clean_json_value(v) for k, v in row.items()}
        for row in df.to_dict(orient="records")
    ]


def save_cache(race: RaceData) -> str:
    path = cache_path_for(race.race_id)
    payload = {
        "cache_version": CACHE_VERSION,
        "race_id": race.race_id,
        "source_url": race.source_url,
        "fetched_at": race.fetched_at,
        "meta": race.meta,
        "warnings": race.warnings or [],
        "rows": _records_for_json(race.rows),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def load_cache(race_id: int) -> Optional[RaceData]:
    path = cache_path_for(race_id)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("cache_version") != CACHE_VERSION:
        return None
    rows = pd.DataFrame(payload.get("rows", []))
    return RaceData(
        race_id=int(payload["race_id"]),
        source_url=payload.get("source_url", netkeiba_url(race_id)),
        fetched_at=payload.get("fetched_at", ""),
        meta=payload.get("meta", {}),
        rows=rows,
        cache_path=path,
        warnings=payload.get("warnings", []),
    )


def _race_date(meta: Dict[str, Any]) -> Optional[date]:
    try:
        value = meta.get("date")
        if not value:
            return None
        return pd.to_datetime(str(value), errors="raise").date()
    except Exception:
        return None


def _is_incomplete_current_or_future_race(race: RaceData) -> bool:
    race_date = _race_date(race.meta or {})
    if race_date is None or race_date < date.today():
        return False
    return len(race.rows) < MIN_REASONABLE_RUNNERS


def _incomplete_race_message(row_count: int) -> str:
    return (
        f"出走表が未確定または取得不足です（取得{row_count}頭）。"
        "出馬確定後に情報更新してから予想してください。"
    )


def list_cached_races(limit: int = 20) -> List[Dict[str, Any]]:
    items = []
    for name in os.listdir(CACHE_DIR):
        if not name.endswith(".json"):
            continue
        path = os.path.join(CACHE_DIR, name)
        try:
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
            meta = payload.get("meta", {})
            items.append({
                "race_id": payload.get("race_id"),
                "title": meta.get("race_name") or meta.get("race_class") or "",
                "date": meta.get("date") or "",
                "fetched_at": payload.get("fetched_at", ""),
                "path": path,
            })
        except Exception:
            continue
    items.sort(key=lambda x: x.get("fetched_at") or "", reverse=True)
    return items[:limit]


def cached_race_ids() -> set[int]:
    ids: set[int] = set()
    for name in os.listdir(CACHE_DIR):
        if not name.endswith(".json"):
            continue
        m = re.match(r"(\d{12})\.json$", name)
        if m:
            ids.add(int(m.group(1)))
    return ids


def processed_race_ids_and_latest_date() -> tuple[set[int], Optional[date]]:
    with gzip.open(DATA_PKL, "rb") as f:
        df = pickle.load(f)
    race_ids = set(pd.to_numeric(df["race_id"], errors="coerce").dropna().astype("int64").tolist())
    latest = None
    if "date" in df.columns and df["date"].notna().any():
        latest = pd.to_datetime(df["date"], errors="coerce").max().date()
    return race_ids, latest


def parse_date(value: Optional[str]) -> Optional[date]:
    if value is None or str(value).strip() == "":
        return None
    return pd.to_datetime(str(value), errors="raise").date()


def iter_dates(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def _race_ids_from_html(html: str) -> List[int]:
    ids: List[int] = []
    seen = set()
    for match in re.finditer(r"race_id(?:=|%3D)(\d{12})", html):
        rid = int(match.group(1))
        if rid not in seen:
            ids.append(rid)
            seen.add(rid)
    return ids


def discover_race_ids_for_date(kaisai_date: date) -> List[int]:
    html = fetch_html(netkeiba_race_list_url(kaisai_date))
    ids = _race_ids_from_html(html)
    if ids:
        return ids

    # The race list page is now populated by JavaScript. Fetch the date tab
    # fragment, find the selected day's subpage, then parse race_id values.
    date_key = f"{kaisai_date:%Y%m%d}"
    date_list_url = "https://race.netkeiba.com/top/race_list_get_date_list.html?" + urlencode({
        "kaisai_date": date_key,
        "encoding": "UTF-8",
    })
    date_list_html = fetch_html(date_list_url)
    soup = BeautifulSoup(date_list_html, "html.parser")
    sub_urls: List[str] = []
    for link in soup.select('li[date] a[href*="race_list_sub.html"]'):
        parent = link.find_parent("li")
        if parent and parent.get("date") != date_key:
            continue
        href = (link.get("href") or "").split("#", 1)[0]
        href = href.replace("\u00a4t_group", "&current_group").replace("&amp;", "&")
        if href:
            sub_urls.append(urljoin("https://race.netkeiba.com/top/", href))

    if not sub_urls:
        for match in re.finditer(r'race_list_sub\.html\?[^"\']+', date_list_html):
            href = match.group(0).split("#", 1)[0]
            if f"kaisai_date={date_key}" in href:
                sub_urls.append(urljoin("https://race.netkeiba.com/top/", href))

    seen = set(ids)
    for sub_url in sub_urls:
        sub_html = fetch_html(sub_url)
        for rid in _race_ids_from_html(sub_html):
            if rid not in seen:
                ids.append(rid)
                seen.add(rid)
    return ids


def candidate_race_dates(count: int = 4, start: Optional[date] = None) -> List[Dict[str, str]]:
    base = start or date.today()
    out: List[Dict[str, str]] = []
    cur = base
    weekday_jp = ["月", "火", "水", "木", "金", "土", "日"]
    while len(out) < count:
        if cur.weekday() in (5, 6):
            out.append({
                "value": cur.isoformat(),
                "label": f"{cur:%Y-%m-%d}({weekday_jp[cur.weekday()]})",
            })
        cur += timedelta(days=1)
    return out


def _text(el: Any) -> str:
    if el is None:
        return ""
    return re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip()


def parse_sex_age(value: str):
    m = re.search(r"([牡牝セ騙])\s*(\d+)", str(value))
    if not m:
        return None, np.nan
    return m.group(1), float(m.group(2))


def parse_body_weight(value: str):
    text = str(value or "").replace(" ", "")
    m = re.search(r"(\d+)\(([+\-]?\d+)\)", text)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(r"(\d+)", text)
    if m:
        return float(m.group(1)), np.nan
    return np.nan, np.nan


def parse_passing(value: str):
    parts = re.findall(r"\d+", str(value or ""))
    if not parts:
        return np.nan, np.nan, np.nan
    nums = [float(int(p)) for p in parts]
    return nums[0], nums[-1], float(np.mean(nums))


def parse_float(value: str):
    text = str(value or "").replace(",", "")
    m = re.search(r"[+\-]?\d+(?:\.\d+)?", text)
    return float(m.group(0)) if m else np.nan


def _table_headers(table: Any) -> List[str]:
    for tr in table.select("tr"):
        cells = tr.find_all(["td", "th"])
        labels = [_text(c).replace(" ", "") for c in cells]
        if any("馬名" in label for label in labels) or any(
            ("オッズ" in label or "人気" in label) for label in labels
        ):
            return labels
    return []


def _cell_float_by_header(
    cells: List[Any],
    headers: List[str],
    keywords: Iterable[str],
    excludes: Iterable[str] = (),
) -> tuple[float, str]:
    for index, label in enumerate(headers):
        if index >= len(cells):
            continue
        compact = str(label).replace(" ", "")
        if any(word in compact for word in keywords) and not any(word in compact for word in excludes):
            value = parse_float(_text(cells[index]))
            if pd.notna(value):
                return value, compact
    return np.nan, ""


def _float_from_selectors(row: Any, selectors: Iterable[str]) -> float:
    for selector in selectors:
        value = parse_float(_text(row.select_one(selector)))
        if pd.notna(value):
            return value
    return np.nan


def _odds_source_from_label(label: str) -> str:
    if "予想" in label:
        return "予想"
    if "単勝" in label or "オッズ" in label:
        return "取得"
    return ""


def _json_from_jsonp(text: str) -> Dict[str, Any]:
    body = str(text or "").strip()
    if body.startswith("(") and body.endswith(")"):
        body = body[1:-1]
    else:
        m = re.search(r"\((\{.*\})\)\s*;?\s*$", body, flags=re.S)
        if m:
            body = m.group(1)
    return json.loads(body)


def _odds_source_from_status(status: str) -> str:
    if status == "yoso":
        return "予想"
    if status in {"middle", "result"}:
        return "確定"
    return "取得"


def fetch_jra_win_odds(race_id: int) -> Dict[int, Dict[str, Any]]:
    params = {
        "pid": "api_get_jra_odds",
        "input": "UTF-8",
        "output": "jsonp",
        "race_id": str(int(race_id)),
        "type": "1",
        "action": "init",
        "sort": "odds",
        # Keep the response plain JSONP so the scraper does not need the
        # browser-side zlib/base64 decoder used by netkeiba's JavaScript.
        "compress": "0",
    }
    url = netkeiba_jra_odds_api_url() + "?" + urlencode(params)
    payload = _json_from_jsonp(fetch_html(url))
    source = _odds_source_from_status(str(payload.get("status", "")))
    odds_rows = (((payload.get("data") or {}).get("odds") or {}).get("1") or {})

    out: Dict[int, Dict[str, Any]] = {}
    for key, value in odds_rows.items():
        if not isinstance(value, (list, tuple)) or not value:
            continue
        odds = parse_float(value[0])
        popularity = parse_float(value[2] if len(value) > 2 else "")
        if pd.isna(odds) and pd.isna(popularity):
            continue
        out[int(key)] = {
            "odds": odds,
            "popularity": popularity,
            "source": source,
        }
    return out


def fill_odds_from_api(rows: pd.DataFrame, race_id: int) -> pd.DataFrame:
    try:
        api_odds = fetch_jra_win_odds(race_id)
    except Exception:
        return rows
    if not api_odds:
        return rows

    out = rows.copy()
    if "odds_key" not in out.columns:
        out["odds_key"] = np.nan
    if "odds_source" not in out.columns:
        out["odds_source"] = ""

    for idx, row in out.iterrows():
        key_value = row.get("odds_key")
        if pd.isna(key_value):
            key_value = row.get("horse_no")
        if pd.isna(key_value):
            continue
        api_row = api_odds.get(int(key_value))
        if not api_row:
            continue
        if pd.isna(row.get("odds")) and pd.notna(api_row.get("odds")):
            out.at[idx, "odds"] = float(api_row["odds"])
            out.at[idx, "odds_source"] = api_row.get("source", "取得")
        if pd.isna(row.get("popularity")) and pd.notna(api_row.get("popularity")):
            out.at[idx, "popularity"] = float(api_row["popularity"])
    return out


PAYOUT_KIND_LABELS = {
    "単勝": "tansho",
    "複勝": "fukusho",
    "枠連": "wakuren",
    "馬連": "umaren",
    "ワイド": "wide",
    "馬単": "umatan",
    "三連複": "sanrenpuku",
    "3連複": "sanrenpuku",
    "三連単": "sanrentan",
    "3連単": "sanrentan",
}

TICKET_LABELS = {
    "tansho": "単勝",
    "fukusho": "複勝",
    "wakuren": "枠連",
    "umaren": "馬連",
    "wide": "ワイド",
    "umatan": "馬単",
    "sanrenpuku": "三連複",
    "sanrentan": "三連単",
}

ORDERED_TICKETS = {"tansho", "fukusho", "umatan", "sanrentan"}
UNORDERED_TICKETS = {"wakuren", "umaren", "wide", "sanrenpuku"}
ODDS_REQUEST_TYPES = {
    "tansho_fukusho": "1",
    "wakuren": "3",
    "umaren": "4",
    "wide": "5",
    "umatan": "6",
    "sanrenpuku": "7",
    "sanrentan": "8",
}
ODDS_EXPECTED_LEN = {
    "tansho": 1,
    "fukusho": 1,
    "wakuren": 2,
    "umaren": 2,
    "wide": 2,
    "umatan": 2,
    "sanrenpuku": 3,
    "sanrentan": 3,
}


def _ticket_kind_from_label(label: str) -> Optional[str]:
    text = re.sub(r"\s+", "", str(label or ""))
    # Longer labels first so 三連単 is not swallowed by 単.
    for key in sorted(PAYOUT_KIND_LABELS, key=len, reverse=True):
        if key in text:
            return PAYOUT_KIND_LABELS[key]
    return None


def normalize_ticket_kind(kind: str) -> str:
    kind = str(kind or "")
    return kind[:-4] if kind.endswith("_box") else kind


def normalize_ticket_horses(kind: str, horses: Iterable[int]) -> List[int]:
    base = normalize_ticket_kind(kind)
    vals = [int(h) for h in horses if pd.notna(h)]
    if base in UNORDERED_TICKETS:
        return sorted(vals)
    return vals


def ticket_key(kind: str, horses: Iterable[int]) -> str:
    base = normalize_ticket_kind(kind)
    nums = normalize_ticket_horses(base, horses)
    return f"{base}:{'-'.join(str(n) for n in nums)}"


def _line_texts(cell) -> List[str]:
    tmp = BeautifulSoup(str(cell), "lxml")
    for br in tmp.find_all("br"):
        br.replace_with("\n")
    text = tmp.get_text("\n", strip=True)
    return [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]


def _split_compact_numbers(token: str, expected_len: int) -> List[int]:
    token = re.sub(r"\D", "", str(token or ""))
    if not token:
        return []
    if expected_len > 1 and len(token) == expected_len * 2:
        return [int(token[i:i + 2]) for i in range(0, len(token), 2)]
    return [int(x) for x in re.findall(r"\d+", token)]


def _combo_items_from_cell(cell, expected_len: int) -> List[List[int]]:
    combos: List[List[int]] = []
    for line in _line_texts(cell):
        nums = [int(x) for x in re.findall(r"\d+", line)]
        if len(nums) == 1 and expected_len > 1:
            nums = _split_compact_numbers(line, expected_len)
        if len(nums) == expected_len:
            combos.append(nums)
        elif len(nums) > expected_len and len(nums) % expected_len == 0:
            for i in range(0, len(nums), expected_len):
                combos.append(nums[i:i + expected_len])

    if combos:
        return combos

    text = " ".join(_line_texts(cell))
    nums = [int(x) for x in re.findall(r"\d+", text)]
    if len(nums) == 1 and expected_len > 1:
        nums = _split_compact_numbers(text, expected_len)
    if len(nums) >= expected_len:
        for i in range(0, len(nums) - expected_len + 1, expected_len):
            chunk = nums[i:i + expected_len]
            if len(chunk) == expected_len:
                combos.append(chunk)
    return combos


def _yen_items_from_cell(cell) -> List[int]:
    text = " ".join(_line_texts(cell))
    return [int(m.replace(",", "")) for m in re.findall(r"(\d[\d,]*)\s*円", text)]


def _popularity_items_from_cell(cell) -> List[int]:
    text = " ".join(_line_texts(cell))
    return [int(m) for m in re.findall(r"(\d+)\s*人気", text)]


def parse_lap_times_from_soup(soup: BeautifulSoup) -> Dict[str, float]:
    """
    レース結果ページから ラップタイム を解析し、
    race_front_pace / race_back_pace / race_pace_diff を返す。
    取得できない場合は空 dict を返す。
    """
    table = soup.select_one("table.Race_HaronTime")
    if table is None:
        return {}
    lap_vals = []
    for td in table.find_all("td"):
        txt = _text(td).strip()
        m = re.match(r"^(\d+\.\d+)$", txt)
        if m:
            lap_vals.append(float(m.group(1)))
    if len(lap_vals) < 3:
        return {}
    front = float(np.mean(lap_vals[:3]))
    back  = float(np.mean(lap_vals[-3:]))
    return {
        "race_front_pace": round(front, 3),
        "race_back_pace":  round(back,  3),
        "race_pace_diff":  round(back - front, 3),
    }


def parse_netkeiba_payouts(soup: BeautifulSoup, race_id: int) -> Dict[str, Any]:
    payouts: Dict[str, List[Dict[str, Any]]] = {}
    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            th_text = " ".join(_text(th) for th in tr.find_all("th"))
            kind = _ticket_kind_from_label(th_text)
            if kind is None:
                continue
            cells = tr.find_all("td")
            if len(cells) < 2:
                continue

            expected_len = ODDS_EXPECTED_LEN.get(kind, 1)
            combos = _combo_items_from_cell(cells[0], expected_len)
            yens = _yen_items_from_cell(cells[1])
            pops = _popularity_items_from_cell(cells[2]) if len(cells) >= 3 else []
            count = min(len(combos), len(yens))
            if count <= 0:
                continue

            items = payouts.setdefault(kind, [])
            for idx in range(count):
                horses = normalize_ticket_horses(kind, combos[idx])
                items.append({
                    "ticket_kind": kind,
                    "label": TICKET_LABELS.get(kind, kind),
                    "horses": horses,
                    "key": ticket_key(kind, horses),
                    "payout_yen": int(yens[idx]),
                    "odds": float(yens[idx]) / 100.0,
                    "popularity": int(pops[idx]) if idx < len(pops) else None,
                })

    if not payouts:
        raise ValueError("払戻テーブルを取得できませんでした。")

    return {
        "race_id": int(race_id),
        "source_url": netkeiba_result_url(int(race_id)),
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "payouts": payouts,
    }


def save_payout_cache(payload: Dict[str, Any]) -> str:
    path = payout_cache_path_for(int(payload["race_id"]))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def load_payout_cache(race_id: int) -> Optional[Dict[str, Any]]:
    path = payout_cache_path_for(int(race_id))
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def scrape_result_payouts(race_id: int) -> Dict[str, Any]:
    html = fetch_html(netkeiba_result_url(int(race_id)))
    soup = BeautifulSoup(html, "lxml")
    return parse_netkeiba_payouts(soup, int(race_id))


def actual_payout_multiplier(payload: Optional[Dict[str, Any]], kind: str, horses: Iterable[int]) -> Optional[float]:
    if not payload:
        return None
    base = normalize_ticket_kind(kind)
    key = ticket_key(base, horses)
    for item in (payload.get("payouts") or {}).get(base, []):
        if item.get("key") == key:
            odds = parse_float(item.get("odds"))
            if pd.notna(odds) and float(odds) > 0:
                return float(odds)
            payout_yen = parse_float(item.get("payout_yen"))
            if pd.notna(payout_yen) and float(payout_yen) > 0:
                return float(payout_yen) / 100.0
    return None


def fetch_jra_odds_payload(race_id: int, type_code: str) -> Dict[str, Any]:
    params = {
        "pid": "api_get_jra_odds",
        "input": "UTF-8",
        "output": "jsonp",
        "race_id": str(int(race_id)),
        "type": str(type_code),
        "action": "init",
        "sort": "odds",
        "compress": "0",
    }
    url = netkeiba_jra_odds_api_url() + "?" + urlencode(params)
    return _json_from_jsonp(fetch_html(url))


def _leaf_odds_rows(obj: Any, path: Optional[List[str]] = None) -> List[tuple[List[str], Any]]:
    path = path or []
    rows: List[tuple[List[str], Any]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            rows.extend(_leaf_odds_rows(value, path + [str(key)]))
    elif isinstance(obj, list):
        if obj and (parse_float(obj[0]) == parse_float(obj[0])):
            rows.append((path, obj))
        else:
            for idx, value in enumerate(obj):
                rows.extend(_leaf_odds_rows(value, path + [str(idx)]))
    return rows


def _horses_from_odds_path(path: List[str], expected_len: int) -> List[int]:
    joined = " ".join(str(p) for p in path)
    nums = [int(x) for x in re.findall(r"\d+", joined)]
    if len(nums) == expected_len:
        return nums
    if len(nums) == 1:
        compact = re.sub(r"\D", "", joined)
        if expected_len > 1 and len(compact) == expected_len * 2:
            return [int(compact[i:i + 2]) for i in range(0, len(compact), 2)]
    if len(nums) > expected_len:
        return nums[-expected_len:]
    return nums


def _normalize_odds_items(kind: str, odds_obj: Any) -> List[Dict[str, Any]]:
    expected_len = ODDS_EXPECTED_LEN.get(kind, 1)
    items: List[Dict[str, Any]] = []
    for path, value in _leaf_odds_rows(odds_obj):
        horses = _horses_from_odds_path(path, expected_len)
        if len(horses) != expected_len:
            continue
        odds = parse_float(value[0] if isinstance(value, list) and value else np.nan)
        if pd.isna(odds) or float(odds) <= 0:
            continue
        odds_max = np.nan
        if kind in {"fukusho", "wide"} and isinstance(value, list) and len(value) > 1:
            odds_max = parse_float(value[1])
        popularity = np.nan
        if isinstance(value, list) and len(value) > 2:
            popularity = parse_float(value[2])
        norm_horses = normalize_ticket_horses(kind, horses)
        item = {
            "ticket_kind": kind,
            "label": TICKET_LABELS.get(kind, kind),
            "horses": norm_horses,
            "key": ticket_key(kind, norm_horses),
            "odds": float(odds),
            "popularity": int(popularity) if pd.notna(popularity) else None,
        }
        if pd.notna(odds_max) and float(odds_max) > 0:
            item["odds_max"] = float(odds_max)
            item["odds"] = (float(odds) + float(odds_max)) / 2.0
        items.append(item)
    return items


def normalize_jra_odds_payload(race_id: int, raw_by_request: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    odds: Dict[str, List[Dict[str, Any]]] = {}
    for request_kind, payload in raw_by_request.items():
        root = ((payload.get("data") or {}).get("odds") or {})
        if request_kind == "tansho_fukusho":
            if "1" in root:
                odds["tansho"] = _normalize_odds_items("tansho", root.get("1"))
            if "2" in root:
                odds["fukusho"] = _normalize_odds_items("fukusho", root.get("2"))
            continue
        kind = request_kind
        type_code = ODDS_REQUEST_TYPES.get(request_kind)
        odds_obj = root.get(type_code) if isinstance(root, dict) and type_code in root else root
        odds[kind] = _normalize_odds_items(kind, odds_obj)

    return {
        "race_id": int(race_id),
        "source_url": netkeiba_result_url(int(race_id)),
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "odds": {k: v for k, v in odds.items() if v},
    }


def scrape_jra_all_odds(race_id: int) -> Dict[str, Any]:
    raw_by_request: Dict[str, Dict[str, Any]] = {}
    for request_kind, type_code in ODDS_REQUEST_TYPES.items():
        raw_by_request[request_kind] = fetch_jra_odds_payload(int(race_id), type_code)
    return normalize_jra_odds_payload(int(race_id), raw_by_request)


def save_odds_cache(payload: Dict[str, Any]) -> str:
    path = odds_cache_path_for(int(payload["race_id"]))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def load_odds_cache(race_id: int) -> Optional[Dict[str, Any]]:
    path = odds_cache_path_for(int(race_id))
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def actual_odds_multiplier(payload: Optional[Dict[str, Any]], kind: str, horses: Iterable[int]) -> Optional[float]:
    if not payload:
        return None
    base = normalize_ticket_kind(kind)
    key = ticket_key(base, horses)
    for item in (payload.get("odds") or {}).get(base, []):
        if item.get("key") == key:
            odds = parse_float(item.get("odds"))
            if pd.notna(odds) and float(odds) > 0:
                return float(odds)
    return None


def parse_rank(value: str):
    text = str(value or "").strip()
    return float(int(text)) if text.isdigit() else np.nan


def parse_race_meta(soup: BeautifulSoup, race_id: int) -> Dict[str, Any]:
    meta: Dict[str, Any] = {"venue": str(race_id)[4:6]}
    meta["venue_name"] = VENUE_NAMES.get(meta["venue"], meta["venue"])

    title_el = soup.select_one(".RaceName")
    if title_el:
        meta["race_name"] = _text(title_el)

    data_text = " ".join(
        _text(el)
        for el in soup.select(".RaceData01, .RaceData02, .RaceList_NameBox, .RaceList_DataItem")
    )
    page_text = _text(soup)
    text = f"{data_text} {page_text}".strip()

    m = re.search(r"(\d+)R", text)
    if m:
        meta["round_no"] = float(m.group(1))
    else:
        race_id_text = str(race_id)
        try:
            meta["round_no"] = float(int(race_id_text[-2:]))
        except Exception:
            meta["round_no"] = np.nan

    m = re.search(r"(芝|ダート|ダ|障害|障)\s*(右|左|直線|直)?\s*(\d{3,4})m", text)
    if m:
        meta["surface"] = {"芝": "turf", "ダ": "dirt", "ダート": "dirt", "障": "jump", "障害": "jump"}.get(m.group(1), "other")
        meta["direction"] = {"直線": "直"}.get(m.group(2), m.group(2) or "?")
        meta["distance"] = float(m.group(3))

    m = re.search(r"天候[:：]\s*([^\s/]+)", text)
    if m:
        meta["weather"] = m.group(1)
    m = re.search(r"馬場[:：]\s*([^\s/]+)", text)
    if m:
        meta["going"] = m.group(1)

    m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", text)
    if m:
        meta["date"] = f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

    class_match = re.search(r"(新馬|未勝利|1勝クラス|2勝クラス|3勝クラス|オープン|OP|L|G1|G2|G3|GI|GII|GIII)", text)
    if class_match:
        meta["race_class"] = class_match.group(1)

    return meta


def _horse_id_from_row(row: Any) -> Optional[int]:
    link = row.select_one("a[href*='/horse/']")
    if not link:
        return None
    m = re.search(r"/horse/(\d+)", link.get("href", ""))
    return int(m.group(1)) if m else None


def _id_from_link(row: Any, kind: str) -> float:
    link = row.select_one(f"a[href*='/{kind}/']")
    if not link:
        return np.nan
    m = re.search(rf"/{kind}/(?:result/recent/)?(\d+)/?", link.get("href", ""))
    return float(int(m.group(1))) if m else np.nan


def parse_netkeiba_entries(soup: BeautifulSoup, race_id: int, meta: Dict[str, Any]) -> pd.DataFrame:
    table = soup.select_one("table.Shutuba_Table") or soup.select_one("table.RaceTable01")
    if table is None:
        raise ValueError("出馬表テーブルを取得できませんでした。URLが出馬表ページか確認してください。")

    headers = _table_headers(table)
    rows = []
    temporary_no = 1
    for tr in table.select("tr"):
        cells = tr.find_all(["td", "th"])
        if len(cells) < 6:
            continue
        texts = [_text(c) for c in cells]
        horse_link = tr.select_one("a[href*='/horse/']")
        horse_name = (
            _text(horse_link)
            or _text(tr.select_one(".HorseName"))
            or (texts[3] if len(texts) > 3 else "")
        )
        if not horse_name or horse_name == "馬名":
            continue

        horse_no_text = _text(tr.select_one(".Umaban")) or (texts[1] if len(texts) > 1 else "")
        horse_no = parse_float(horse_no_text)
        horse_no_confirmed = pd.notna(horse_no)
        if pd.isna(horse_no):
            # Future races can list registered horses before frame/horse numbers
            # are finalized. Keep those rows predict-able with a temporary order.
            horse_no = float(temporary_no)
        temporary_no += 1

        frame_no = parse_float(_text(tr.select_one(".Waku")) or texts[0])

        sex_age_text = _text(tr.select_one(".Barei"))
        for t in texts:
            if sex_age_text:
                break
            if re.match(r"^[牡牝セ騙]\d+", t):
                sex_age_text = t
                break
        sex, age = parse_sex_age(sex_age_text)

        weights = [parse_float(t) for t in texts]
        weight_carry = np.nan
        for v in weights:
            if 45 <= v <= 65:
                weight_carry = v
                break

        body_weight, body_diff = np.nan, np.nan
        weight_text = _text(tr.select_one(".Weight"))
        if weight_text:
            body_weight, body_diff = parse_body_weight(weight_text)
        for t in texts:
            if pd.notna(body_weight):
                break
            if re.search(r"\d+\([+\-]?\d+\)", t):
                body_weight, body_diff = parse_body_weight(t)
                break

        numeric_tail = [parse_float(t) for t in texts[-5:]]
        odds_source = ""
        odds = _float_from_selectors(tr, [
            ".Popular span[id^='odds-']",
            "td.Popular",
        ])
        if pd.notna(odds):
            odds_source = "確定"

        if pd.isna(odds):
            odds, odds_label = _cell_float_by_header(cells, headers, ["予想オッズ", "単勝オッズ", "オッズ"], ["人気"])
            odds_source = _odds_source_from_label(odds_label)

        if pd.isna(odds):
            odds = _float_from_selectors(tr, [
                "td[class*='Yoso'][class*='Odds']",
                "td[class*='Pred'][class*='Odds']",
                "td[class*='Forecast'][class*='Odds']",
                "td[class*='Odds']",
            ])
            if pd.notna(odds):
                odds_source = "予想"

        popularity = _float_from_selectors(tr, [
            ".Popular_Ninki span",
            "td.Popular_Ninki",
        ])
        if pd.isna(popularity):
            popularity, pop_label = _cell_float_by_header(cells, headers, ["予想人気", "人気"])
        else:
            pop_label = "人気"
        if pd.isna(popularity):
            popularity = _float_from_selectors(tr, [
                "td[class*='Yoso'][class*='Ninki']",
                "td[class*='Pred'][class*='Ninki']",
                "td[class*='Forecast'][class*='Ninki']",
                "td[class*='Ninki']",
            ])
        for v in numeric_tail:
            if pd.notna(odds):
                break
            if pd.notna(v) and 1.0 <= v <= 999.9:
                odds = v
                odds_source = "表"
                break
        for v in reversed(numeric_tail):
            if pd.notna(popularity):
                break
            if pd.notna(v) and 1 <= v <= 30 and float(v).is_integer():
                popularity = v
                break
        odds_key = np.nan
        odds_span = tr.select_one("span[id^='odds-1_']")
        if odds_span:
            m = re.search(r"odds-1_(\d+)", odds_span.get("id", ""))
            if m:
                odds_key = float(int(m.group(1)))

        row = {
            "race_id": int(race_id),
            "horse_id": _horse_id_from_row(tr),
            "horse_no": float(horse_no),
            "horse_no_confirmed": bool(horse_no_confirmed),
            "frame_no": frame_no,
            "馬名": horse_name,
            "sex": sex,
            "age": age,
            "weight_carry": weight_carry,
            "body_weight": body_weight,
            "body_weight_diff": body_diff,
            "odds": odds,
            "odds_source": odds_source or ("予想" if "予想" in str(pop_label) else ""),
            "odds_key": odds_key,
            "popularity": popularity,
            "jockey_id": _id_from_link(tr, "jockey"),
            "trainer_id": _id_from_link(tr, "trainer"),
            "surface": meta.get("surface", "other"),
            "direction": meta.get("direction", "?"),
            "distance": meta.get("distance", np.nan),
            "venue": str(meta.get("venue", str(race_id)[4:6])),
            "weather": meta.get("weather"),
            "going": meta.get("going"),
            "race_class": meta.get("race_class"),
            "round_no": meta.get("round_no", np.nan),
            "date": pd.to_datetime(meta.get("date"), errors="coerce"),
            "rank": np.nan,
        }
        rows.append(row)

    if not rows:
        raise ValueError("出走馬を取得できませんでした。")

    df = pd.DataFrame(rows).sort_values("horse_no").reset_index(drop=True)
    df["field_size"] = float(len(df))
    if df["popularity"].isna().all() and df["odds"].notna().any():
        df["popularity"] = df["odds"].rank(method="min")
    return df


def parse_netkeiba_result_entries(soup: BeautifulSoup, race_id: int, meta: Dict[str, Any]) -> pd.DataFrame:
    lap_pace = parse_lap_times_from_soup(soup)
    table = soup.select_one("table.RaceTable01")
    if table is None:
        raise ValueError("結果テーブルを取得できませんでした。")

    rows = []
    for tr in table.select("tr.HorseList, tr.FirstDisplay"):
        cells = tr.find_all(["td", "th"])
        if len(cells) < 15:
            continue
        texts = [_text(c) for c in cells]
        rank = parse_rank(texts[0])
        horse_no = parse_float(texts[2])
        horse_link = tr.select_one("a[href*='/horse/']")
        horse_name = _text(horse_link) if horse_link else texts[3]
        if not horse_name or pd.isna(horse_no):
            continue

        sex, age = parse_sex_age(texts[4])
        passing_first, passing_last, passing_mean = parse_passing(texts[12])
        body_weight, body_diff = parse_body_weight(texts[14])
        rows.append({
            "race_id": int(race_id),
            "horse_id": _horse_id_from_row(tr),
            "jockey_id": _id_from_link(tr, "jockey"),
            "trainer_id": _id_from_link(tr, "trainer"),
            "time_index": np.nan,
            "round_no": meta.get("round_no", np.nan),
            "race_class": meta.get("race_class"),
            "surface": meta.get("surface", "other"),
            "direction": meta.get("direction", "?"),
            "distance": meta.get("distance", np.nan),
            "weather": meta.get("weather"),
            "going": meta.get("going"),
            "sex": sex,
            "age": age,
            "body_weight": body_weight,
            "body_weight_diff": body_diff,
            "rank": rank,
            "passing_first": passing_first,
            "passing_last": passing_last,
            "passing_mean": passing_mean,
            "agari": parse_float(texts[11]),
            "odds": parse_float(texts[10]),
            "popularity": parse_float(texts[9]),
            "weight_carry": parse_float(texts[5]),
            "frame_no": parse_float(texts[1]),
            "horse_no": float(horse_no),
            "date": pd.to_datetime(meta.get("date"), errors="coerce"),
            "venue": str(meta.get("venue", str(race_id)[4:6])),
            "馬名": horse_name,
            "race_front_pace": lap_pace.get("race_front_pace", np.nan),
            "race_back_pace":  lap_pace.get("race_back_pace",  np.nan),
            "race_pace_diff":  lap_pace.get("race_pace_diff",  np.nan),
        })

    if not rows:
        raise ValueError("結果が未確定、または着順データを取得できませんでした。")

    df = pd.DataFrame(rows).sort_values("horse_no").reset_index(drop=True)
    df["field_size"] = float(len(df))
    return df


def _history_data() -> pd.DataFrame:
    global _HISTORY_CACHE
    if _HISTORY_CACHE is not None:
        return _HISTORY_CACHE
    with gzip.open(DATA_PKL, "rb") as f:
        _HISTORY_CACHE = pickle.load(f)
    _HISTORY_CACHE["date"] = pd.to_datetime(_HISTORY_CACHE["date"], errors="coerce")
    return _HISTORY_CACHE


def _mean_numeric(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return np.nan
    values = pd.to_numeric(df[col], errors="coerce")
    return float(values.mean()) if values.notna().any() else np.nan


def _max_numeric(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return np.nan
    values = pd.to_numeric(df[col], errors="coerce")
    return float(values.max()) if values.notna().any() else np.nan


def _last_value(df: pd.DataFrame, col: str) -> Any:
    if col not in df.columns or df.empty:
        return np.nan
    values = df[col].dropna()
    return values.iloc[-1] if len(values) else np.nan


def _rate_from_rank(df: pd.DataFrame, op: str) -> float:
    if "rank" not in df.columns or df.empty:
        return np.nan
    ranks = pd.to_numeric(df["rank"], errors="coerce").dropna()
    if ranks.empty:
        return np.nan
    if op == "win":
        return float((ranks == 1).mean())
    return float((ranks <= 3).mean())


def _id_value(value: Any) -> Optional[int]:
    v = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return int(v) if pd.notna(v) else None


def _person_stats(hist: pd.DataFrame, col: str, person_id: Optional[int]) -> Dict[str, float]:
    if person_id is None or col not in hist.columns:
        return {"runs": np.nan, "win_rate": np.nan, "top3_rate": np.nan}
    sub = hist[pd.to_numeric(hist[col], errors="coerce") == person_id]
    return {
        "runs": float(len(sub)),
        "win_rate": _rate_from_rank(sub, "win"),
        "top3_rate": _rate_from_rank(sub, "top3"),
    }


def _matches_value(series: pd.Series, value: Any) -> pd.Series:
    if value is None or pd.isna(value):
        return pd.Series(False, index=series.index)
    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric_value):
        return pd.to_numeric(series, errors="coerce") == numeric_value
    return series.astype(str) == str(value)


def _filtered_rank_stats(hist: pd.DataFrame, filters: Dict[str, Any]) -> Dict[str, float]:
    sub = hist
    for col, value in filters.items():
        if col not in sub.columns:
            return {"runs": np.nan, "win_rate": np.nan, "top3_rate": np.nan}
        sub = sub[_matches_value(sub[col], value)]
    return {
        "runs": float(len(sub)),
        "win_rate": _rate_from_rank(sub, "win"),
        "top3_rate": _rate_from_rank(sub, "top3"),
    }


def _person_stats_filtered(hist: pd.DataFrame, col: str, person_id: Optional[int], filters: Dict[str, Any]) -> Dict[str, float]:
    if person_id is None or col not in hist.columns:
        return {"runs": np.nan, "win_rate": np.nan, "top3_rate": np.nan}
    sub = hist[pd.to_numeric(hist[col], errors="coerce") == person_id]
    return _filtered_rank_stats(sub, filters)


def _future_feature_row(row: pd.Series, horse_hist: pd.DataFrame, hist_before: pd.DataFrame) -> Dict[str, Any]:
    horse_hist = horse_hist.sort_values("date")
    current_date = pd.to_datetime(row.get("date"), errors="coerce")
    current_distance = pd.to_numeric(pd.Series([row.get("distance")]), errors="coerce").iloc[0]
    current_surface = row.get("surface")
    current_venue = row.get("venue")

    rec: Dict[str, Any] = {
        "horse_id": row.get("horse_id"),
        "horse_runs": float(len(horse_hist)),
        "horse_avg_rank": _mean_numeric(horse_hist, "rank"),
        "horse_win_rate": _rate_from_rank(horse_hist, "win"),
        "horse_top3_rate": _rate_from_rank(horse_hist, "top3"),
        "horse_avg_time_idx": _mean_numeric(horse_hist, "time_index"),
        "horse_best_time_idx": _max_numeric(horse_hist, "time_index"),
        "horse_avg_agari": _mean_numeric(horse_hist, "agari"),
        "sire": _last_value(horse_hist, "sire"),
        "broodmare_sire": _last_value(horse_hist, "broodmare_sire"),
    }

    if horse_hist.empty or pd.isna(current_date):
        rec["days_since_last"] = np.nan
    else:
        last_date = pd.to_datetime(horse_hist["date"].iloc[-1], errors="coerce")
        rec["days_since_last"] = float((current_date - last_date).days) if pd.notna(last_date) else np.nan

    for lag in (1, 2, 3):
        if len(horse_hist) >= lag:
            prev = horse_hist.iloc[-lag]
            field_size = pd.to_numeric(pd.Series([prev.get("field_size")]), errors="coerce").iloc[0]
            prev_distance = pd.to_numeric(pd.Series([prev.get("distance")]), errors="coerce").iloc[0]
            passing_first = pd.to_numeric(pd.Series([prev.get("passing_first")]), errors="coerce").iloc[0]
            passing_last = pd.to_numeric(pd.Series([prev.get("passing_last")]), errors="coerce").iloc[0]
            denom = field_size - 1 if pd.notna(field_size) and field_size != 1 else np.nan
            rec[f"horse_rank_lag{lag}"] = pd.to_numeric(pd.Series([prev.get("rank")]), errors="coerce").iloc[0]
            rec[f"horse_time_idx_lag{lag}"] = pd.to_numeric(pd.Series([prev.get("time_index")]), errors="coerce").iloc[0]
            rec[f"horse_agari_lag{lag}"] = pd.to_numeric(pd.Series([prev.get("agari")]), errors="coerce").iloc[0]
            rec[f"horse_passing_first_lag{lag}"] = passing_first
            rec[f"horse_passing_last_lag{lag}"] = passing_last
            rec[f"horse_field_size_lag{lag}"] = field_size
            rec[f"horse_distance_lag{lag}"] = prev_distance
            rec[f"horse_passing_first_rate_lag{lag}"] = (passing_first - 1) / denom if pd.notna(denom) and pd.notna(passing_first) else np.nan
            rec[f"horse_passing_last_rate_lag{lag}"] = (passing_last - 1) / denom if pd.notna(denom) and pd.notna(passing_last) else np.nan
            rec[f"horse_passing_gain_lag{lag}"] = passing_first - passing_last if pd.notna(passing_first) and pd.notna(passing_last) else np.nan
            rec[f"horse_distance_diff_lag{lag}"] = current_distance - prev_distance if pd.notna(current_distance) and pd.notna(prev_distance) else np.nan
        else:
            for col in (
                "rank", "time_idx", "agari", "passing_first", "passing_last",
                "field_size", "distance", "passing_first_rate", "passing_last_rate",
                "passing_gain", "distance_diff",
            ):
                rec[f"horse_{col}_lag{lag}"] = np.nan

    rank_lags = [rec.get(f"horse_rank_lag{lag}") for lag in (1, 2, 3)]
    rank_values = pd.to_numeric(pd.Series(rank_lags), errors="coerce")
    rec["horse_recent_avg_rank3"] = float(rank_values.mean()) if rank_values.notna().any() else np.nan
    rec["horse_recent_top3_rate3"] = float((rank_values <= 3).mean())

    front_values = pd.to_numeric(pd.Series([rec.get(f"horse_passing_first_rate_lag{lag}") for lag in (1, 2, 3)]), errors="coerce")
    closing_values = pd.to_numeric(pd.Series([rec.get(f"horse_passing_gain_lag{lag}") for lag in (1, 2, 3)]), errors="coerce")
    rec["horse_front_style"] = float(front_values.mean()) if front_values.notna().any() else np.nan
    rec["horse_closing_style"] = float(closing_values.mean()) if closing_values.notna().any() else np.nan
    diff1 = rec.get("horse_distance_diff_lag1")
    rec["horse_shorter_than_last"] = float(pd.notna(diff1) and diff1 < 0)
    rec["horse_longer_than_last"] = float(pd.notna(diff1) and diff1 > 0)
    closing = rec.get("horse_closing_style")
    rec["closer_short_distance_risk"] = float(max(0.0, closing) if pd.notna(closing) and pd.notna(current_distance) and current_distance <= 1400 else 0.0)
    front = rec.get("horse_front_style")
    rec["front_short_distance_fit"] = float((1 - min(1.0, max(0.0, front))) if pd.notna(front) and pd.notna(current_distance) and current_distance <= 1400 else 0.0)
    rec["closer_long_distance_fit"] = float(max(0.0, closing) if pd.notna(closing) and pd.notna(current_distance) and current_distance >= 1800 else 0.0)
    rec["front_long_distance_risk"] = float((1 - min(1.0, max(0.0, front))) if pd.notna(front) and pd.notna(current_distance) and current_distance >= 1800 else 0.0)

    horse_condition_stats = {
        "horse_dist": {"distance": current_distance},
        "horse_surface": {"surface": current_surface},
        "horse_venue": {"venue": current_venue},
        "horse_course": {"venue": current_venue, "surface": current_surface, "distance": current_distance},
    }
    for prefix, filters in horse_condition_stats.items():
        stats = _filtered_rank_stats(horse_hist, filters)
        rec[f"{prefix}_runs"] = stats["runs"]
        rec[f"{prefix}_win_rate"] = stats["win_rate"]
        rec[f"{prefix}_top3_rate"] = stats["top3_rate"]

    jockey_stats = _person_stats(hist_before, "jockey_id", _id_value(row.get("jockey_id")))
    trainer_stats = _person_stats(hist_before, "trainer_id", _id_value(row.get("trainer_id")))
    rec["jockey_runs"] = jockey_stats["runs"]
    rec["jockey_win_rate"] = jockey_stats["win_rate"]
    rec["jockey_top3_rate"] = jockey_stats["top3_rate"]
    rec["trainer_win_rate"] = trainer_stats["win_rate"]
    rec["trainer_top3_rate"] = trainer_stats["top3_rate"]
    person_filters = {
        "venue": {"venue": current_venue},
        "course": {"venue": current_venue, "surface": current_surface, "distance": current_distance},
    }
    jockey_id = _id_value(row.get("jockey_id"))
    trainer_id = _id_value(row.get("trainer_id"))
    for suffix, filters in person_filters.items():
        stats = _person_stats_filtered(hist_before, "jockey_id", jockey_id, filters)
        rec[f"jockey_{suffix}_runs"] = stats["runs"]
        rec[f"jockey_{suffix}_win_rate"] = stats["win_rate"]
        rec[f"jockey_{suffix}_top3_rate"] = stats["top3_rate"]
        stats = _person_stats_filtered(hist_before, "trainer_id", trainer_id, filters)
        rec[f"trainer_{suffix}_runs"] = stats["runs"]
        rec[f"trainer_{suffix}_win_rate"] = stats["win_rate"]
        rec[f"trainer_{suffix}_top3_rate"] = stats["top3_rate"]
    return rec


def enrich_with_history(df: pd.DataFrame) -> pd.DataFrame:
    if "horse_id" not in df.columns or df["horse_id"].dropna().empty:
        return df
    hist = _history_data()
    race_dates = pd.to_datetime(df.get("date"), errors="coerce").dropna()
    race_date = race_dates.min() if len(race_dates) else pd.NaT
    hist_before = hist[pd.to_datetime(hist["date"], errors="coerce") < race_date].copy() if pd.notna(race_date) else hist.copy()

    horse_ids = pd.to_numeric(df["horse_id"], errors="coerce").dropna().astype("int64")
    horse_hist_all = hist_before[pd.to_numeric(hist_before["horse_id"], errors="coerce").isin(horse_ids)].copy()
    feature_rows = []
    for _, row in df.iterrows():
        horse_id = _id_value(row.get("horse_id"))
        horse_hist = (
            horse_hist_all[pd.to_numeric(horse_hist_all["horse_id"], errors="coerce") == horse_id]
            if horse_id is not None else horse_hist_all.iloc[0:0]
        )
        feature_rows.append(_future_feature_row(row, horse_hist, hist_before))

    features = pd.DataFrame(feature_rows)
    merged = df.merge(features, on="horse_id", how="left", suffixes=("", "_hist"))
    for col in list(merged.columns):
        if col.endswith("_hist"):
            base = col[:-5]
            if base in merged.columns:
                merged[base] = merged[base].combine_first(merged[col])
            else:
                merged[base] = merged[col]
            merged = merged.drop(columns=[col])
    return merged


def fetch_html(url: str) -> str:
    session = requests.Session()
    # Windows/proxy環境で HTTPS_PROXY=http://127.0.0.1:9 のような壊れた設定が
    # 入っていると取得に失敗するため、アプリのスクレイピングでは環境プロキシを無視する。
    session.trust_env = False
    headers = dict(DEFAULT_HEADERS)
    # neteiba はリファラーなしで直接アクセスすると弾くことがある。
    # 事前にトップページを取得してクッキーをセットし、Referer を付与する。
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        session.get(origin, headers=headers, timeout=15)
        headers["Referer"] = origin + "/"
    except Exception:
        pass
    res = session.get(url, headers=headers, timeout=30)
    res.raise_for_status()
    res.encoding = res.apparent_encoding or res.encoding
    return res.text


def scrape_race(source: str) -> RaceData:
    race_id = race_id_from_source(source)
    url = str(source).strip() if str(source).strip().startswith("http") else netkeiba_url(race_id)
    html = fetch_html(url)
    soup = BeautifulSoup(html, "lxml")
    meta = parse_race_meta(soup, race_id)
    rows = parse_netkeiba_entries(soup, race_id, meta)
    rows = fill_odds_from_api(rows, race_id)
    rows = enrich_with_history(rows)
    probe = RaceData(race_id=race_id, source_url=url, fetched_at="", meta=meta, rows=rows)
    if _is_incomplete_current_or_future_race(probe):
        raise ValueError(_incomplete_race_message(len(rows)))
    warnings = []
    matched = int(pd.to_numeric(rows.get("horse_runs"), errors="coerce").fillna(0).gt(0).sum()) if "horse_runs" in rows.columns else 0
    if matched < len(rows):
        warnings.append(f"履歴特徴量を補完できた馬は {matched}/{len(rows)} 頭です。未補完の特徴量はNaNで予想します。")
    if rows["odds"].isna().all():
        warnings.append("単勝オッズが未取得です。オッズはモデル確率から推定します。")
    elif "odds_source" in rows.columns and rows["odds_source"].astype(str).eq("予想").any():
        warnings.append("確定オッズが未取得の馬は、出馬表の予想オッズ・予想人気を使用しています。")
    race = RaceData(
        race_id=race_id,
        source_url=url,
        fetched_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        meta=meta,
        rows=rows,
        warnings=warnings,
    )
    race.cache_path = save_cache(race)
    return race


def scrape_result_race(race_id: int) -> pd.DataFrame:
    url = netkeiba_result_url(race_id)
    html = fetch_html(url)
    soup = BeautifulSoup(html, "lxml")
    meta = parse_race_meta(soup, int(race_id))
    rows = parse_netkeiba_result_entries(soup, int(race_id), meta)
    if rows["rank"].notna().sum() == 0:
        raise ValueError("結果が未確定です。")
    return rows


def load_or_scrape(source: str, force: bool = False) -> RaceData:
    race_id = race_id_from_source(source)
    if not force:
        cached = load_cache(race_id)
        if cached is not None:
            if _is_incomplete_current_or_future_race(cached):
                return scrape_race(source)
            missing = REQUIRED_HISTORY_FEATURES.difference(cached.rows.columns)
            if missing and "horse_id" in cached.rows.columns:
                try:
                    cached.rows = enrich_with_history(cached.rows)
                    cached.cache_path = save_cache(cached)
                except Exception as exc:
                    cached.warnings = list(cached.warnings or [])
                    cached.warnings.append(f"履歴特徴量の再計算に失敗しました: {exc}")
            return cached
    return scrape_race(source)


def _emit_progress(progress: Optional[Callable[[Dict[str, Any]], None]], **payload):
    if progress is not None:
        progress(payload)


def _history_feature_columns(columns: Iterable[str]) -> set[str]:
    prefixes = (
        "horse_runs", "horse_avg_", "horse_best_time_idx", "horse_win_rate", "horse_top3_rate",
        "days_since_last", "jockey_runs", "jockey_win_rate", "jockey_top3_rate",
        "trainer_win_rate", "trainer_top3_rate", "horse_dist_",
        "horse_surface_", "field_size", "horse_rank_lag",
        "horse_time_idx_lag", "horse_agari_lag", "horse_passing_",
        "horse_field_size_lag", "horse_distance_lag", "horse_distance_diff_lag",
        "horse_recent_", "horse_front_style", "horse_closing_style",
        "horse_shorter_than_last", "horse_longer_than_last",
        "closer_short_distance_risk", "front_short_distance_fit",
        "closer_long_distance_fit", "front_long_distance_risk",
        "horse_venue_", "horse_course_", "jockey_venue_",
        "jockey_course_", "trainer_venue_", "trainer_course_",
    )
    return {c for c in columns if c.startswith(prefixes)}


def _fill_pedigree_from_history(new_rows: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    out = new_rows.copy()
    for col in ("sire", "broodmare_sire"):
        if col not in out.columns:
            out[col] = np.nan
    if "horse_id" not in hist.columns:
        return out
    ped_cols = [c for c in ["horse_id", "date", "sire", "broodmare_sire"] if c in hist.columns]
    ped = (
        hist[ped_cols]
        .dropna(subset=["horse_id"])
        .sort_values("date" if "date" in hist.columns else "horse_id")
        .drop_duplicates("horse_id", keep="last")
    )
    out = out.merge(ped, on="horse_id", how="left", suffixes=("", "_hist"))
    for col in ("sire", "broodmare_sire"):
        hist_col = f"{col}_hist"
        if hist_col in out.columns:
            out[col] = out[col].combine_first(out[hist_col])
            out = out.drop(columns=[hist_col])
    return out


def append_results_to_processed(result_rows: pd.DataFrame) -> Dict[str, Any]:
    if result_rows.empty:
        return {"rows_added": 0, "races_added": 0, "backup_path": "", "saved_path": DATA_PKL}

    with gzip.open(DATA_PKL, "rb") as f:
        hist = pickle.load(f)

    result_rows = result_rows.copy()
    result_rows = result_rows[~result_rows["race_id"].isin(set(hist["race_id"].astype("int64")))]
    if result_rows.empty:
        return {"rows_added": 0, "races_added": 0, "backup_path": "", "saved_path": DATA_PKL}

    result_rows = _fill_pedigree_from_history(result_rows, hist)
    history_cols = _history_feature_columns(hist.columns)
    base_cols = [c for c in hist.columns if c not in history_cols]

    for col in base_cols:
        if col not in result_rows.columns:
            result_rows[col] = np.nan
    combined = pd.concat([hist[base_cols], result_rows[base_cols]], ignore_index=True)

    for col in ["race_id"]:
        combined[col] = pd.to_numeric(combined[col], errors="coerce").astype("int64")
    for col in ["horse_id", "jockey_id", "trainer_id"]:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce").astype("Int32")
    for col in ["date"]:
        combined[col] = pd.to_datetime(combined[col], errors="coerce")
    for col in combined.columns:
        if col in {"race_id", "horse_id", "jockey_id", "trainer_id", "date"}:
            continue
        if col not in {"race_class", "surface", "direction", "weather", "going", "sex", "venue", "sire", "broodmare_sire"}:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    from preprocess import add_history_features

    combined = add_history_features(combined, verbose=False)
    ordered = [c for c in hist.columns if c in combined.columns] + [c for c in combined.columns if c not in hist.columns]
    combined = combined[ordered]

    for col in hist.columns:
        if col in combined.columns:
            try:
                if str(hist[col].dtype) == "category":
                    combined[col] = combined[col].astype("category")
                elif str(hist[col].dtype).startswith("float"):
                    combined[col] = pd.to_numeric(combined[col], errors="coerce").astype(hist[col].dtype)
                elif str(hist[col].dtype).startswith("Int"):
                    combined[col] = pd.to_numeric(combined[col], errors="coerce").astype(hist[col].dtype)
            except Exception:
                pass

    backup_path = os.path.join(
        os.path.dirname(DATA_PKL),
        f"processed_all_backup_{datetime.now():%Y%m%d_%H%M%S}.pkl.gz",
    )
    shutil.copy2(DATA_PKL, backup_path)
    with gzip.open(DATA_PKL, "wb") as f:
        pickle.dump(combined, f, protocol=pickle.HIGHEST_PROTOCOL)

    global _HISTORY_CACHE
    _HISTORY_CACHE = combined
    return {
        "rows_added": int(len(result_rows)),
        "races_added": int(result_rows["race_id"].nunique()),
        "backup_path": backup_path,
        "saved_path": DATA_PKL,
        "latest_date": pd.to_datetime(combined["date"], errors="coerce").max().date().isoformat(),
        "total_rows": int(len(combined)),
    }


def race_ids_for_year(year: int) -> List[int]:
    with gzip.open(DATA_PKL, "rb") as f:
        df = pickle.load(f)
    dates = pd.to_datetime(df["date"], errors="coerce")
    sub = (
        df.loc[dates.dt.year == int(year), ["date", "race_id"]]
        .assign(
            date=dates[dates.dt.year == int(year)],
            race_id=lambda x: pd.to_numeric(x["race_id"], errors="coerce"),
        )
        .dropna(subset=["date", "race_id"])
        .drop_duplicates("race_id")
        .sort_values(["date", "race_id"])
    )
    return [int(rid) for rid in sub["race_id"].astype("int64").tolist()]


def fetch_market_data_for_year(
    year: int,
    fetch_payouts: bool = True,
    fetch_odds: bool = False,
    force: bool = False,
    limit: Optional[int] = None,
    sleep_seconds: float = 0.25,
    progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """
    Fetch historical market data for races already present in processed_all.

    Payouts are used to settle simulations with actual published returns.
    Odds are stored separately and, when present, can be used by betting logic
    instead of formula-based estimates.
    """
    if should_cancel is None:
        should_cancel = lambda: False
    if not fetch_payouts and not fetch_odds:
        fetch_payouts = True

    race_ids = race_ids_for_year(int(year))
    total_found = len(race_ids)
    if limit is not None and int(limit) > 0:
        race_ids = race_ids[:int(limit)]

    total = max(1, len(race_ids))
    payout_ok = payout_skip = payout_fail = 0
    odds_ok = odds_skip = odds_fail = 0
    updates: List[Dict[str, Any]] = []

    _emit_progress(
        progress,
        phase="prepare",
        current=0,
        total=total,
        message=f"{year}年の登録済みレース {total_found}R から取得対象 {len(race_ids)}R を準備しました",
    )

    for index, rid in enumerate(race_ids, start=1):
        if should_cancel():
            break

        item = {
            "race_id": int(rid),
            "payout": "対象外",
            "odds": "対象外",
            "message": "",
        }

        if fetch_payouts:
            payout_path = payout_cache_path_for(rid)
            if os.path.exists(payout_path) and not force:
                payout_skip += 1
                item["payout"] = "既存"
            else:
                try:
                    payload = scrape_result_payouts(rid)
                    save_payout_cache(payload)
                    payout_ok += 1
                    item["payout"] = "取得"
                except Exception as exc:
                    payout_fail += 1
                    item["payout"] = "失敗"
                    item["message"] = str(exc)

        if fetch_odds:
            odds_path = odds_cache_path_for(rid)
            if os.path.exists(odds_path) and not force:
                odds_skip += 1
                item["odds"] = "既存"
            else:
                try:
                    payload = scrape_jra_all_odds(rid)
                    save_odds_cache(payload)
                    odds_ok += 1
                    item["odds"] = "取得"
                except Exception as exc:
                    odds_fail += 1
                    item["odds"] = "失敗"
                    item["message"] = (item["message"] + " / " if item["message"] else "") + str(exc)

        updates.append(item)
        _emit_progress(
            progress,
            phase="download",
            current=index,
            total=total,
            message=(
                f"{rid} market data ({index}/{len(race_ids)}) "
                f"払戻 {item['payout']} / オッズ {item['odds']}"
            ),
        )
        if sleep_seconds and float(sleep_seconds) > 0 and index < len(race_ids):
            time.sleep(float(sleep_seconds))

    cancelled = should_cancel()
    _emit_progress(
        progress,
        phase="done" if not cancelled else "cancelled",
        current=len(updates),
        total=total,
        message=(
            f"市場データ取得{'停止' if cancelled else '完了'}: "
            f"払戻 取得{payout_ok} 既存{payout_skip} 失敗{payout_fail} / "
            f"オッズ 取得{odds_ok} 既存{odds_skip} 失敗{odds_fail}"
        ),
    )
    return {
        "mode": "market_data",
        "year": int(year),
        "total_found": total_found,
        "target_races": len(race_ids),
        "processed": len(updates),
        "cancelled": cancelled,
        "fetch_payouts": bool(fetch_payouts),
        "fetch_odds": bool(fetch_odds),
        "payout_ok": payout_ok,
        "payout_skip": payout_skip,
        "payout_fail": payout_fail,
        "odds_ok": odds_ok,
        "odds_skip": odds_skip,
        "odds_fail": odds_fail,
        "payout_dir": PAYOUT_DIR,
        "odds_dir": ODDS_DIR,
        "updates": updates,
    }


def auto_update_missing(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: Optional[int] = 0,
    skip_cached: bool = True,
    refresh_lookups: bool = True,
    progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Search race-list pages from the local dataset's next day through today,
    scrape completed results that are absent from processed_all, then append
    them to the processed training dataset with the same history features.
    """
    known_ids, latest_date = processed_race_ids_and_latest_date()
    start = parse_date(start_date)
    end = parse_date(end_date) or date.today()

    if start is None:
        start = (latest_date + timedelta(days=1)) if latest_date else end
    if start > end:
        start = end

    max_updates = None if limit is None or int(limit) <= 0 else int(limit)
    scan_dates = list(iter_dates(start, end))
    if not scan_dates:
        scan_dates = [end]
    date_total = len(scan_dates)
    discovered: List[int] = []
    missing: List[int] = []
    date_results: List[Dict[str, Any]] = []
    scanned_dates = 0
    skipped_existing = 0
    stopped_by_limit = False

    _emit_progress(
        progress,
        phase="search",
        current=0,
        total=date_total,
        message=f"{start.isoformat()} から {end.isoformat()} まで開催候補日を検索します",
    )

    for date_index, d in enumerate(scan_dates, start=1):
        scanned_dates += 1
        try:
            race_ids = discover_race_ids_for_date(d)
            date_results.append({"date": d.isoformat(), "ok": True, "races": len(race_ids), "message": ""})
        except Exception as exc:
            date_results.append({"date": d.isoformat(), "ok": False, "races": 0, "message": str(exc)})
            _emit_progress(
                progress,
                phase="search",
                current=date_index,
                total=date_total,
                message=f"{d.isoformat()} の検索に失敗: {exc}",
            )
            continue

        for rid in race_ids:
            discovered.append(rid)
            if rid in known_ids:
                skipped_existing += 1
                continue
            missing.append(rid)
            if max_updates is not None and len(missing) >= max_updates:
                stopped_by_limit = True
                break
        _emit_progress(
            progress,
            phase="search",
            current=date_index,
            total=date_total,
            message=f"{d.isoformat()} を検索: {len(race_ids)}R発見 / 更新候補 {len(missing)}R",
        )
        if stopped_by_limit:
            break

    updates = []
    result_frames: List[pd.DataFrame] = []
    update_total = max(1, len(missing))
    _emit_progress(
        progress,
        phase="download",
        current=0,
        total=update_total,
        message=f"未登録 {len(missing)}R の結果ページを取得します",
    )
    for update_index, rid in enumerate(missing, start=1):
        try:
            rows = scrape_result_race(rid)
            result_frames.append(rows)
            item = {
                "ok": True,
                "race_id": int(rid),
                "rows": len(rows),
                "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "message": "結果を取得しました",
            }
        except Exception as exc:
            item = {
                "ok": False,
                "race_id": rid,
                "rows": 0,
                "fetched_at": "",
                "message": str(exc),
            }
        updates.append(item)
        _emit_progress(
            progress,
            phase="download",
            current=update_index,
            total=update_total,
            message=f"{rid} の結果を取得中 ({update_index}/{len(missing)})",
        )

    ok_count = sum(1 for item in updates if item.get("ok"))
    failed_count = len(updates) - ok_count
    dataset_update = {
        "rows_added": 0,
        "races_added": 0,
        "backup_path": "",
        "saved_path": DATA_PKL,
    }
    if result_frames:
        _emit_progress(
            progress,
            phase="save",
            current=0,
            total=1,
            message="学習データを再構築して保存します",
        )
        dataset_update = append_results_to_processed(pd.concat(result_frames, ignore_index=True))
        _emit_progress(
            progress,
            phase="save",
            current=1,
            total=1,
            message=f"学習データ保存完了: {dataset_update.get('races_added', 0)}R / {dataset_update.get('rows_added', 0)}行追加",
        )
    lookup_update = {
        "updated": False,
        "reason": "学習データの追加がないため未実行",
    }
    if refresh_lookups and dataset_update.get("rows_added", 0):
        _emit_progress(
            progress,
            phase="lookup",
            current=0,
            total=10,
            message="オッズ・人気なしモデル用 lookup を更新します",
        )
        try:
            from no_market_v4_lookups import regenerate_v4_lookups
            lookup_update = regenerate_v4_lookups(progress=progress)
        except Exception as exc:
            lookup_update = {
                "updated": False,
                "reason": "lookup 更新に失敗",
                "error": str(exc),
            }
            _emit_progress(
                progress,
                phase="lookup",
                current=10,
                total=10,
                message=f"lookup 更新に失敗: {exc}",
            )
    elif not refresh_lookups:
        lookup_update = {
            "updated": False,
            "reason": "refresh_lookups=False",
        }
    _emit_progress(
        progress,
        phase="done",
        current=1,
        total=1,
        message=f"更新完了: 取得成功 {ok_count}R / 失敗 {failed_count}R",
    )

    return {
        "mode": "auto",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "scanned_dates": scanned_dates,
        "discovered_races": len(set(discovered)),
        "skipped_existing": skipped_existing,
        "missing_races": len(missing),
        "updated": ok_count,
        "failed": failed_count,
        "dataset_updated": bool(dataset_update.get("rows_added", 0)),
        "rows_added": dataset_update.get("rows_added", 0),
        "races_added": dataset_update.get("races_added", 0),
        "backup_path": dataset_update.get("backup_path", ""),
        "saved_path": dataset_update.get("saved_path", DATA_PKL),
        "latest_date": dataset_update.get("latest_date", latest_date.isoformat() if latest_date else ""),
        "total_rows": dataset_update.get("total_rows"),
        "lookup_updated": bool(lookup_update.get("updated")),
        "lookup_update": lookup_update,
        "stopped_by_limit": stopped_by_limit,
        "limit": max_updates,
        "date_results": date_results,
        "updates": updates,
    }
