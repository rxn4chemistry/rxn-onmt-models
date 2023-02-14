from freezegun import freeze_time

from rxn.onmt_utils.utils import log_file_name_from_time


@freeze_time("2018-03-23 15:54:22")
def test_log_file_name_from_time() -> None:
    assert log_file_name_from_time() == "20180323-1554.log"
    assert log_file_name_from_time("dummy_prefix") == "dummy_prefix-20180323-1554.log"
