"""Additive admission rules for an unchanged-row subset of a verified corpus.

This module deliberately does not import or alter the frozen quality pipeline.
URL routing is evidence of an unreviewed source family, not a language label or
proof that every quarantined document was machine translated.
"""
import json
import math
import re
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlsplit

POLICY_PATH = Path(__file__).resolve().parents[1] / "configs/h200_admission_policy.json"
_DATED_LINE = re.compile(r"(?m)^\s*\d{1,2}[./-]\d{1,2}[./-]\d{4}\s+\d{1,2}:\d{2}\b")
_CYRILLIC_WORD = re.compile(r"[а-яёіў]+")
_MENU_COUNT = re.compile(r"\(\d[\d\s]*\)")
_HOST = re.compile(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?\Z")


def load_admission_policy(path=None):
    """Load a validated policy; malformed or unsupported rules fail closed."""
    policy = json.loads(Path(path or POLICY_PATH).read_text(encoding="utf-8"))
    if policy.get("schema") != "belka-h200-admission-policy-v1":
        raise ValueError("unsupported admission policy schema")
    if policy.get("independent_native_review") is not False:
        raise ValueError("admission policy cannot claim native review")
    excluded = policy["excluded_sources"]
    if not isinstance(excluded, dict) or any(not isinstance(k, str) or not k or not isinstance(v, str) or not v for k, v in excluded.items()):
        raise ValueError("invalid excluded source policy")
    reviewed = policy["reviewed_content_quarantine"]
    if not isinstance(reviewed, dict):
        raise ValueError("invalid reviewed content policy")
    for digest, evidence in reviewed.items():
        if (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or
                not isinstance(evidence, dict) or set(evidence) != {"source", "sample_report", "sample_id", "reason"} or
                any(not isinstance(evidence[k], str) or not evidence[k] for k in ("source", "sample_report", "reason")) or
                type(evidence["sample_id"]) is not int or evidence["sample_id"] < 1 or evidence["sample_report"] not in policy["evidence_files"]):
            raise ValueError("invalid reviewed content evidence")
    route = policy["multilingual_routing"]
    template = policy["templates"]
    for values in (policy["url_sources"], policy["evidence_files"],
                   *(route[k] for k in ("host_prefixes", "path_prefixes", "query_keys", "query_values", "local_tld_exceptions")),
                   *(template[k] for k in ("read_more_markers", "archive_path_segments", "share_tail_markers", "category_menu_markers", "game_ui_markers", "wiki_control_tail_markers", "wiki_control_markers", "google_play_ui_markers")), policy["ocr_fragmentation"]["common_short_words"]):
        if not isinstance(values, list) or not values or len(values) != len(set(values)) or any(not isinstance(v, str) or not v for v in values):
            raise ValueError("invalid admission policy list")
    for mapping in (policy["blocked_hosts"], route["publisher_exceptions"]):
        if not isinstance(mapping, dict) or any(not _HOST.fullmatch(host) or host != host.lower() or ".." in host or not isinstance(reason, str) or not reason for host, reason in mapping.items()):
            raise ValueError("invalid admission host policy")
    for key in ("raw_wiki_minimum_template_chars", "raw_wiki_minimum_fields", "read_more_minimum_occurrences", "dated_index_minimum_entries", "share_tail_minimum_short_lines", "share_tail_short_line_maximum_words", "category_menu_minimum_counts", "game_ui_minimum_distinct_markers", "game_ui_maximum_chars", "wiki_control_minimum_distinct_markers", "google_play_ui_minimum_distinct_markers", "google_play_ui_maximum_chars"):
        if type(template[key]) is not int or template[key] <= 0:
            raise ValueError("invalid admission template count: " + key)
    for key in ("raw_wiki_minimum_fraction", "share_tail_minimum_fraction", "category_menu_minimum_fraction", "game_ui_minimum_fraction", "wiki_control_minimum_tail_fraction", "google_play_ui_minimum_fraction"):
        value = template[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError("invalid admission template fraction: " + key)
    if not isinstance(template["paginated_archive_path_segment"], str) or not re.fullmatch(r"[a-z]+", template["paginated_archive_path_segment"]):
        raise ValueError("invalid paginated archive path segment")
    ocr = policy["ocr_fragmentation"]
    for key in ("minimum_cyrillic_words", "maximum_short_word_length", "minimum_long_word_length"):
        if type(ocr[key]) is not int or ocr[key] <= 0:
            raise ValueError("invalid admission OCR count: " + key)
    for key in ("minimum_short_word_fraction", "minimum_non_function_short_word_fraction", "maximum_long_word_fraction"):
        if type(ocr[key]) not in (int, float) or not math.isfinite(ocr[key]) or not 0 < ocr[key] <= 1:
            raise ValueError("invalid admission OCR fraction: " + key)
    if (type(ocr["maximum_mean_word_length"]) not in (int, float) or
            not math.isfinite(ocr["maximum_mean_word_length"]) or ocr["maximum_mean_word_length"] <= 0 or
            ocr["maximum_short_word_length"] >= ocr["minimum_long_word_length"]):
        raise ValueError("invalid admission OCR word lengths")
    clone = policy["clone_route"]
    if set(clone) != {"host_tld", "first_path_segment"} or any(not isinstance(v, str) or not v or not re.fullmatch(r"[a-z]+", v) for v in clone.values()):
        raise ValueError("invalid clone route")
    return policy


def _host_matches(host, domain):
    return host == domain or host.endswith("." + domain)


def _raw_template_chars(text, minimum_fields):
    """Count disjoint complete or unterminated outer {{...}} field templates."""
    depth = 0
    start = None
    total = 0
    position = 0
    while position < len(text) - 1:
        pair = text[position:position + 2]
        if pair == "{{":
            if depth == 0:
                start = position
            depth += 1
            position += 2
        elif pair == "}}" and depth:
            depth -= 1
            position += 2
            if depth == 0 and text.count("|", start, position) >= minimum_fields:
                total += position - start
        else:
            position += 1
    if depth and text.count("|", start) >= minimum_fields:
        total += len(text) - start
    return total


def admission_reason(row, policy):
    """Return the first whole-row quarantine reason, otherwise None.

    The input row and its text/metadata are never mutated. A URL exception only
    bypasses the multilingual-route gate; it does not bypass template checks.
    """
    text = row.get("text")
    source = row.get("source")
    if not isinstance(text, str) or not text or not isinstance(source, str) or not source:
        raise ValueError("admission requires nonempty text and source")
    if source in policy["excluded_sources"]:
        return "legacy_source_failed_repeated_content_review"
    reviewed = policy["reviewed_content_quarantine"].get(row.get("content_sha256"))
    if reviewed is not None:
        if source != reviewed["source"]:
            raise ValueError("reviewed content source mismatch")
        return "reviewed_material_content_defect"
    parts = []
    if source in policy["url_sources"]:
        url = row.get("source_url")
        if not isinstance(url, str) or not url:
            return "missing_source_url_for_admission"
        try:
            parsed = urlsplit(url)
            host = (parsed.hostname or "").lower().rstrip(".")
            if parsed.scheme not in {"http", "https"} or not host or parsed.username is not None:
                return "invalid_source_url_for_admission"
            parts = [p.casefold() for p in unquote(parsed.path).split("/") if p]
            query = [(k.casefold(), v.casefold()) for k, v in parse_qsl(parsed.query, max_num_fields=128)]
        except (ValueError, UnicodeError):
            return "invalid_source_url_for_admission"
        if any(_host_matches(host, domain) for domain in policy["blocked_hosts"]):
            return "reviewed_problem_source_family"
        clone = policy["clone_route"]
        if host.endswith("." + clone["host_tld"]) and parts and parts[0] == clone["first_path_segment"]:
            return "unreviewed_translated_news_route"
        route = policy["multilingual_routing"]
        # www is a transport alias, not the language-routing label.
        routed_host = host.removeprefix("www.")
        multilingual = (routed_host.split(".")[0] in route["host_prefixes"] or
                        bool(parts and parts[0] in route["path_prefixes"]) or
                        any(k in route["query_keys"] and v in route["query_values"] for k, v in query))
        excepted = (any(host.endswith("." + tld) for tld in route["local_tld_exceptions"]) or
                    any(_host_matches(host, domain) for domain in route["publisher_exceptions"]))
        if multilingual and not excepted:
            return "unreviewed_multilingual_route"
    cfg = policy["templates"]
    if (len(parts) >= 3 and parts[-2] == cfg["paginated_archive_path_segment"] and parts[-1].isdigit() and
            any(p in cfg["archive_path_segments"] for p in parts[:-2])):
        return "paginated_article_archive"
    if "{{" in text:
        amount = _raw_template_chars(text, cfg["raw_wiki_minimum_fields"])
        if amount >= cfg["raw_wiki_minimum_template_chars"] and amount / len(text) >= cfg["raw_wiki_minimum_fraction"]:
            return "dominant_unexpanded_wiki_template"
    lower = text.casefold()
    if (sum(lower.count(marker) for marker in cfg["read_more_markers"]) >= cfg["read_more_minimum_occurrences"] and
            any(p in cfg["archive_path_segments"] for p in parts)):
        return "article_teaser_archive"
    if lower.count("tags:") >= cfg["dated_index_minimum_entries"] and len(_DATED_LINE.findall(text)) >= cfg["dated_index_minimum_entries"]:
        return "dated_tagged_news_index"
    for marker in cfg["category_menu_markers"]:
        offset = lower.find(marker)
        if (offset >= 0 and (len(lower) - offset) / len(lower) >= cfg["category_menu_minimum_fraction"] and
                len(_MENU_COUNT.findall(lower[offset:])) >= cfg["category_menu_minimum_counts"]):
            return "dominant_category_selection_menu"
    if len(text) <= cfg["game_ui_maximum_chars"]:
        found = [marker for marker in cfg["game_ui_markers"] if marker in lower]
        if (len(found) >= cfg["game_ui_minimum_distinct_markers"] and
                sum(lower.count(marker) * len(marker) for marker in found) / len(lower) >= cfg["game_ui_minimum_fraction"]):
            return "dominant_game_navigation_controls"
    if len(text) <= cfg["google_play_ui_maximum_chars"]:
        found = [marker for marker in cfg["google_play_ui_markers"] if marker in lower]
        if (len(found) >= cfg["google_play_ui_minimum_distinct_markers"] and
                sum(lower.count(marker) * len(marker) for marker in found) / len(lower) >= cfg["google_play_ui_minimum_fraction"]):
            return "dominant_google_play_navigation_controls"
    ocr = policy["ocr_fragmentation"]
    words = _CYRILLIC_WORD.findall(lower)
    if len(words) >= ocr["minimum_cyrillic_words"]:
        short = [word for word in words if len(word) <= ocr["maximum_short_word_length"]]
        common = set(ocr["common_short_words"])
        if (len(short) / len(words) >= ocr["minimum_short_word_fraction"] and
                sum(word not in common for word in short) / len(words) >= ocr["minimum_non_function_short_word_fraction"] and
                sum(len(word) >= ocr["minimum_long_word_length"] for word in words) / len(words) <= ocr["maximum_long_word_fraction"] and
                sum(map(len, words)) / len(words) <= ocr["maximum_mean_word_length"]):
            return "extreme_short_token_fragmentation"
    for marker in cfg["wiki_control_tail_markers"]:
        offset = lower.find(marker)
        if (offset >= 0 and (len(lower) - offset) / len(lower) >= cfg["wiki_control_minimum_tail_fraction"] and
                sum(label in lower[offset:] for label in cfg["wiki_control_markers"]) >= cfg["wiki_control_minimum_distinct_markers"]):
            return "substantial_wiki_navigation_tail"
    for marker in cfg["share_tail_markers"]:
        offset = lower.find(marker)
        if offset < 0 or (len(lower) - offset) / len(lower) < cfg["share_tail_minimum_fraction"]:
            continue
        tail = lower[offset + len(marker):]
        short_lines = sum(0 < len(line.split()) <= cfg["share_tail_short_line_maximum_words"] for line in tail.splitlines())
        if short_lines >= cfg["share_tail_minimum_short_lines"]:
            return "dominant_related_links_tail"
    return None
