import copy
import json

import pytest

from data_pipeline.h200_admission import admission_reason, load_admission_policy


def row(url="https://example.com/article", source="fineweb2", text=None):
    return dict(source=source, source_url=url, text=text or "Беларускі тэхнічны тэкст пра налады камп'ютара і працу праграмы Microsoft Teams.")


@pytest.mark.parametrize("url", [
    "https://be.example.com/article", "https://www.by.example.com/article",
    "https://example.com/be/article", "https://example.com/%62e/article",
    "https://example.com/article?lang=be", "https://example.com/article?HL=BE",
])
def test_unreviewed_multilingual_routes(url):
    assert admission_reason(row(url), load_admission_policy()) == "unreviewed_multilingual_route"


@pytest.mark.parametrize("url", [
    "https://example.ru/article", "https://example.com/article",
    "https://techking.by/article", "https://be.editorial.by/article",
    "https://be.wikipedia.org/wiki/Example", "https://spring96.org/be/news/123",
    "https://charter97.org/be/news/123", "https://belsat.eu/be/article",
])
def test_routing_does_not_ban_country_tlds_or_belarusian_technical_prose(url):
    original = row(url)
    before = copy.deepcopy(original)
    assert admission_reason(original, load_admission_policy()) is None
    assert original == before


def test_domain_boundaries_and_explicit_block_override_local_exception():
    policy = load_admission_policy()
    assert admission_reason(row("https://be.wikipedia.org.evil.test/x"), policy) == "unreviewed_multilingual_route"
    assert admission_reason(row("https://notferryto.com/x"), policy) is None
    assert admission_reason(row("https://www.ferryto.com/be/x"), policy) == "reviewed_problem_source_family"
    policy["blocked_hosts"]["bad.example.by"] = "fixture confirmed malformed prose"
    assert admission_reason(row("https://bad.example.by/be/x"), policy) == "reviewed_problem_source_family"


def test_clone_route_is_narrow_and_decoded():
    policy = load_admission_policy()
    assert admission_reason(row("https://example.ru/%6eaviny/article"), policy) == "unreviewed_translated_news_route"
    assert admission_reason(row("https://example.ru/science/article"), policy) is None
    assert admission_reason(row("https://example.by/naviny/article"), policy) is None


def test_missing_url_is_explicit_for_url_sources_only():
    policy = load_admission_policy()
    assert admission_reason(row(None), policy) == "missing_source_url_for_admission"
    assert admission_reason(row(None, "fixture-prose"), policy) is None
    assert admission_reason(row("http://[broken"), policy) == "invalid_source_url_for_admission"


def test_archive_requires_multiple_controls_and_archive_path():
    policy = load_admission_policy()
    text = "\n".join(["Кароткі анонс матэрыялу. Читать далее »"] * 3)
    assert admission_reason(row("https://news.by/topics/video/", text=text), policy) == "article_teaser_archive"
    assert admission_reason(row("https://news.by/article", text=text), policy) is None
    assert admission_reason(row("https://news.by/topics/video/", text="Матэрыял. Читать далее »"), policy) is None


def test_dated_tagged_teaser_index_without_url():
    policy = load_admission_policy()
    text = "\n".join(f"{day:02}.09.2018 12:30 Кароткі анонс. tags: навіны" for day in [21, 22, 23])
    assert admission_reason(row(None, "fixture-prose", text), policy) == "dated_tagged_news_index"
    assert admission_reason(row(None, "fixture-prose", text.replace("tags:", "тэма:")), policy) is None


def test_template_dominance_preserves_small_templates_inside_prose():
    policy = load_admission_policy()
    template = "{{Адміністрацыйная адзінка " + "|Назва вельмі доўгага параметра = " * 15 + "}}"
    assert admission_reason(row(None, "be_x_oldwiki", template + " Беларуская вёска."), policy) == "dominant_unexpanded_wiki_template"
    assert admission_reason(row(None, "be_x_oldwiki", template + " Беларуская вёска і яе гісторыя." * 100), policy) is None
    assert admission_reason(row(None, "be_x_oldwiki", "{{Цытата|Беларускі тэкст}} і літаратурны аналіз."), policy) is None
    assert admission_reason(row(None, "be_x_oldwiki", template[:-2]), policy) == "dominant_unexpanded_wiki_template"


def test_large_related_link_tail_and_small_footer():
    policy = load_admission_policy()
    body = "Беларуская змястоўная гісторыя. " * 15
    tail = "Падзяліцеся гэтым артыкулам:\n" + "Новая асобная навіна пра іншую падзею\n" * 20
    assert admission_reason(row("https://news.by/article", text=body + tail), policy) == "dominant_related_links_tail"
    assert admission_reason(row("https://news.by/article", text=body * 20 + tail), policy) is None


@pytest.mark.parametrize("bad", [0, -1, True, float("nan"), 1.1])
def test_invalid_policy_fractions_fail_closed(tmp_path, bad):
    policy = load_admission_policy()
    policy["templates"]["raw_wiki_minimum_fraction"] = bad
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy))
    with pytest.raises(ValueError):
        load_admission_policy(path)


def test_invalid_rows_fail_instead_of_silent_admission():
    with pytest.raises(ValueError):
        admission_reason(dict(text=None, source="fineweb2"), load_admission_policy())


def test_paginated_category_archive_does_not_need_a_specific_read_more_spelling():
    policy = load_admission_policy()
    text = "Загаловак. Кароткі анонс (далей…)\n" * 2
    assert admission_reason(row("https://budzma.org/category/konkurs/page/5/", text=text), policy) == "paginated_article_archive"
    assert admission_reason(row("https://news.by/article/page/5/", text=text), policy) is None


def test_confirmed_inkmilk_family_not_all_ru_hosts():
    policy = load_admission_policy()
    assert admission_reason(row("http://inkmilk.ru/08_21_7956/"), policy) == "reviewed_problem_source_family"
    assert admission_reason(row("http://another.ru/08_21_7956/"), policy) is None


def test_short_dominant_raw_infobox():
    policy = load_admission_policy()
    text = "{{Асоба " + "|Вельмі доўгае поле = " * 11 + "}}"
    assert 200 < len(text) < 300
    assert admission_reason(row(None, "fixture-prose", text + " Беларуская біяграфія."), policy) == "dominant_unexpanded_wiki_template"


def test_category_count_menu_requires_marker_counts_and_dominance():
    policy = load_admission_policy()
    menu = "Выберите рубрику " + "Адукацыя (123) Культура (30) Навіны (45) " * 9
    assert admission_reason(row(None, "fixture-prose", "Змястоўны артыкул. " * 10 + menu), policy) == "dominant_category_selection_menu"
    assert admission_reason(row(None, "fixture-prose", "Змястоўны артыкул. " * 100 + menu), policy) is None
    assert admission_reason(row(None, "fixture-prose", menu.replace("Выберите рубрику", "Статыстычныя дадзеныя")), policy) is None


def test_game_controls_do_not_remove_substantive_game_articles():
    policy = load_admission_policy()
    controls = "\n".join(policy["templates"]["game_ui_markers"])
    assert admission_reason(row(None, "fixture-prose", "Назва гульні.\n" + controls), policy) == "dominant_game_navigation_controls"
    article = "Гэта гісторыя распрацоўкі гульні і яе мастацкіх асаблівасцей. " * 40
    assert admission_reason(row(None, "fixture-prose", article + controls), policy) is None


def test_wiki_controls_only_when_tail_is_substantial():
    policy = load_admission_policy()
    controls = "Асабістыя прылады\n" + "\n".join(policy["templates"]["wiki_control_markers"])
    assert admission_reason(row(None, "fixture-prose", "Кароткі артыкул. " * 10 + controls), policy) == "substantial_wiki_navigation_tail"
    assert admission_reason(row(None, "fixture-prose", "Змястоўны артыкул. " * 100 + controls), policy) is None


def test_raw_template_five_fields_still_needs_absolute_and_relative_dominance():
    policy = load_admission_policy()
    template = "{{Загаловак " + "|Назва доўгага змястоўнага параметра = значэнне " * 5 + "}}"
    assert admission_reason(row(None, "bewikisource_full", template + " Кароткі ўрывак."), policy) == "dominant_unexpanded_wiki_template"
    assert admission_reason(row(None, "bewikisource_full", template + " Змястоўная беларуская літаратурная проза." * 50), policy) is None
    assert admission_reason(row(None, "bewikisource_full", "{{Назва|а|б|в|г|д}} і літаратурны аналіз."), policy) is None


def test_google_play_controls_require_a_dense_distinct_cluster():
    policy = load_admission_policy()
    controls = "\n".join(policy["templates"]["google_play_ui_markers"])
    assert admission_reason(row(None, "fixture-prose", controls + " Апісанне праграмы."), policy) == "dominant_google_play_navigation_controls"
    article = "Беларускі тэхнічны артыкул тлумачыць працу мабільнага прыкладання і наладжванне сістэмы. " * 60
    assert admission_reason(row(None, "fixture-prose", article + controls), policy) is None
    assert admission_reason(row(None, "fixture-prose", "У праграме ёсць мае падпіскі і выбар рэдакцыі. Перакласці апісанне можна самастойна."), policy) is None


def test_extreme_syllable_spacing_quarantines_only_long_dominant_pattern():
    policy = load_admission_policy()
    fragment = "ма лень кія дзе ці па чы на юць раз маў ляць і на ву ча юц ца "
    assert admission_reason(row(None, "fixture-prose", fragment * 15), policy) == "extreme_short_token_fragmentation"
    assert admission_reason(row(None, "fixture-prose", fragment), policy) is None
    prose = "Маленькія дзеці паступова пачынаюць размаўляць і навучаюцца разумець навакольны свет. " * 40
    assert admission_reason(row(None, "fixture-prose", prose + fragment), policy) is None
    # A long poem with short function words is not evidence of split syllables.
    poem = "Я і ты, мы і ён,\nУ нас ёсць мой дом.\nІ яна, і яны,\nМы ўсе тут, мы ўсе там.\n" * 12
    assert admission_reason(row(None, "bewikisource_full", poem), policy) is None
    examples = "Прыклад: ма-ла-ко. Гэта паказвае падзел слова на склады; наступны прыклад тлумачыць гукі. " * 20
    assert admission_reason(row(None, "bewikibooks_full", examples), policy) is None


@pytest.mark.parametrize("field,bad", [("minimum_cyrillic_words", 0), ("maximum_short_word_length", True), ("minimum_short_word_fraction", 1.2), ("maximum_long_word_fraction", float("nan")), ("maximum_mean_word_length", -1)])
def test_invalid_ocr_policy_fails_closed(tmp_path, field, bad):
    policy = load_admission_policy()
    policy["ocr_fragmentation"][field] = bad
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy))
    with pytest.raises(ValueError):
        load_admission_policy(path)


@pytest.mark.parametrize("source", ["cc100-be", "bewikiquote"])
def test_failed_legacy_source_is_quarantined_whole_without_mutation(source):
    policy = load_admission_policy()
    original = row(None, source, "Змястоўны беларускі артыкул без бачнай памылкі.")
    before = copy.deepcopy(original)
    assert admission_reason(original, policy) == "legacy_source_failed_repeated_content_review"
    assert original == before
    assert admission_reason(row(None, "fixture-prose", original["text"]), policy) is None


def test_reviewed_content_quarantine_is_exact_and_source_bound():
    policy = load_admission_policy()
    digest, evidence = next(iter(policy["reviewed_content_quarantine"].items()))
    example = row(None, evidence["source"])
    example["content_sha256"] = digest
    assert admission_reason(example, policy) == "reviewed_material_content_defect"
    example["content_sha256"] = "0" * 64
    assert admission_reason(example, policy) is None
    example["content_sha256"] = digest
    example["source"] = "different-source"
    with pytest.raises(ValueError, match="source mismatch"):
        admission_reason(example, policy)


def test_reviewed_content_requires_a_bound_evidence_report(tmp_path):
    policy = load_admission_policy()
    next(iter(policy["reviewed_content_quarantine"].values()))["sample_report"] = "unbound.json"
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="reviewed content evidence"):
        load_admission_policy(path)
