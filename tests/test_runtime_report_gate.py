from tools.validate_runtime_test_report import validate

def test_missing_dependency_and_unexecuted_roundtrip_fail(tmp_path):
    report=tmp_path/'tests.xml'
    report.write_text('<testsuites><testsuite><testcase name="test_native_export_roundtrip"><skipped message="could not import torch"/></testcase></testsuite></testsuites>')
    failure=validate(report)
    assert any('unexpected skip' in row for row in failure)
    assert any('required CPU integration' in row for row in failure)
    report.write_text('<testsuites><testsuite><testcase name="test_native_export_roundtrip"/><testcase name="gpu"><skipped message="requires H200 hardware"/></testcase></testsuite></testsuites>')
    assert validate(report)==[]
