from tools.validate_runtime_test_report import validate
import xml.etree.ElementTree as ET
import pytest

def test_missing_dependency_and_unexecuted_roundtrip_fail(tmp_path):
    report=tmp_path/'tests.xml'
    report.write_text('<testsuites><testsuite><testcase name="test_native_export_roundtrip"><skipped message="could not import torch"/></testcase></testsuite></testsuites>')
    failure=validate(report)
    assert any('unexpected skip' in row for row in failure)
    assert any('required CPU integration' in row for row in failure)
    report.write_text('<testsuites><testsuite><testcase name="test_native_export_roundtrip"/><testcase name="gpu"><skipped message="requires H200 hardware"/></testcase></testsuite></testsuites>')
    assert validate(report)==[]

@pytest.mark.parametrize('classname,name,reason,allowed', [
    ('tests.test_tokenizer_isolation', 'test_d8_workspace_exists',
     'requires local .workspace d8 build artifacts (not present on clean clone)', True),
    ('tests.test_tokenizer_isolation', 'test_d8_workspace_exists', 'could not import torch', False),
    ('tests.test_other', 'test_d8_workspace_exists',
     'requires local .workspace d8 build artifacts (not present on clean clone)', False),
    ('tests.test_tokenizer_isolation', 'test_other',
     'requires local .workspace d8 build artifacts (not present on clean clone)', False),
])
def test_local_artifact_skip_is_limited_to_exact_test(tmp_path, classname, name, reason, allowed):
    suite=ET.Element('testsuite')
    ET.SubElement(suite, 'testcase', name='test_native_export_roundtrip')
    case=ET.SubElement(suite, 'testcase', classname=classname, name=name)
    ET.SubElement(case, 'skipped', message=reason)
    report=tmp_path/'tests.xml'
    ET.ElementTree(suite).write(report)
    failures=validate(report)
    if allowed:
        assert failures==[]
    else:
        assert any('unexpected skip' in row for row in failures)
