from data_pipeline.detect_belarusian import detect_belarusian


def test_accepts_belarusian_text():
    text = "Гэта беларускі тэкст, які мае літары ў, і, ё. Тут ёсць мова, краіна, чалавек і звесткі."
    res = detect_belarusian(text, allow_short=True)
    assert res.score >= 4
    assert res.decision == "accept"


def test_rejects_english_dominant_text():
    text = "This is an English paragraph about model data and meeting workspace privacy without Belarusian text."
    res = detect_belarusian(text, allow_short=True)
    assert res.decision in {"reject", "quarantine"}
    assert res.latin_ratio > 0.5


def test_russian_without_belarusian_markers_not_accepted():
    text = "Это русский текст который должен быть отброшен потому что он не содержит белорусских маркеров."
    res = detect_belarusian(text, allow_short=True)
    assert res.decision != "accept"
