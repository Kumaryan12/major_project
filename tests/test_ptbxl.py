import pytest

from mi_localization.ptbxl import classify_record, parse_scp_codes


MI_CODES = {"AMI", "ALMI", "ASMI", "IMI", "ILMI", "IPMI"}
ABNORMAL = MI_CODES | {"STTC", "CD"}


def test_parse_scp_codes_is_literal_only():
    assert parse_scp_codes("{'AMI': 100.0, 'SR': 0.0}") == {"AMI": 100.0, "SR": 0.0}
    with pytest.raises((ValueError, SyntaxError)):
        parse_scp_codes("__import__('os').system('echo unsafe')")


def test_strict_record_labels():
    assert classify_record({"AMI": 100.0}, MI_CODES, ABNORMAL)[0] == "AMI"
    assert classify_record({"NORM": 100.0, "SR": 0.0}, MI_CODES, ABNORMAL)[0] == "HC"
    assert classify_record({"AMI": 100.0, "IMI": 0.0}, MI_CODES, ABNORMAL)[1] == "multiple_mi_codes"
    assert classify_record({"NORM": 100.0, "STTC": 0.0}, MI_CODES, ABNORMAL)[0] == "EXCLUDE"
    assert classify_record({"IPMI": 100.0}, MI_CODES, ABNORMAL)[1] == "non_target_mi"
