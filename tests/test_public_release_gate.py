import json
import sys

import pytest

import tools.validate_public_release as release


@pytest.fixture
def release_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(release, 'REPO', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['validate_public_release.py', '--dry-run'])
    tracked = list(release.REQUIRED_FILES)
    for name in tracked:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}\n')
    monkeypatch.setattr(release, 'tracked_files', lambda: tracked)
    return tmp_path, tracked


def test_release_requires_metadata_but_not_local_eval_payload(release_tree, capsys):
    root, _ = release_tree
    assert not list((root/'eval/datasets/belarusianglue').glob('*/*.jsonl'))
    assert release.main() == 0
    assert json.loads(capsys.readouterr().out)['ready'] is True
    # Acquired, ignored data may be present locally without becoming release payload.
    local = root/'eval/datasets/belarusianglue/besls/test.jsonl'
    local.parent.mkdir()
    local.write_text('{"sentence":"fixture","label":1}\n')
    assert release.main() == 0


@pytest.mark.parametrize('name', ['eval/datasets/belarusianglue/MANIFEST.json',
                                 'tools/acquire_belarusianglue.py'])
def test_release_rejects_missing_reconstruction_inputs(release_tree, capsys, name):
    root, _ = release_tree
    (root/name).unlink()
    assert release.main() == 1
    assert 'missing required file: '+name in json.loads(capsys.readouterr().out)['problems']


@pytest.mark.parametrize('name', ['eval/datasets/belarusianglue/besls/test.jsonl',
                                 'eval/datasets/belarusianglue/raw/bewic/test.arrow',
                                 'eval/datasets/belarusianglue/bewic/validation.jsonl.gz'])
def test_release_rejects_tracked_third_party_eval_payload(release_tree, capsys, name):
    _, tracked = release_tree
    tracked.append(name)
    assert release.main() == 1
    assert 'tracked raw data: '+name in json.loads(capsys.readouterr().out)['problems']
