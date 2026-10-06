"""Preparing explicit artifacts must not redirect subsequent runtime calls."""
import os

import pytest

from test_training_plan import prepared_fixture


@pytest.mark.parametrize('selected', [None, 'previous-runtime-base'])
def test_preparation_preserves_callers_artifact_selection(monkeypatch, request, selected):
    if selected is None:
        monkeypatch.delenv('NANOCHAT_BASE_DIR', raising=False)
    else:
        monkeypatch.setenv('NANOCHAT_BASE_DIR', selected)
    base, plan = request.getfixturevalue('prepared_fixture')
    assert plan['prepared_base'] == str(base)
    assert (base/'tokenizer/TOKENIZER_TRAINING_MANIFEST.json').is_file()
    assert os.environ.get('NANOCHAT_BASE_DIR') == selected
