from pathlib import Path

import pytest
from freezegun import freeze_time
from rxn.utilities.files import named_temporary_path

from rxn_onmt_utils.utils import (
    ensure_directory_exists_and_is_empty,
    log_file_name_from_time,
)


@freeze_time("2018-03-23 15:54:22")
def test_log_file_name_from_time() -> None:
    assert log_file_name_from_time() == "20180323-1554.log"
    assert log_file_name_from_time("dummy_prefix") == "dummy_prefix-20180323-1554.log"


def test_ensure_directory_exists_and_is_empty() -> None:
    path: Path
    with named_temporary_path() as path:
        # Calling the function on the path creates it as a directory
        assert not path.exists()
        ensure_directory_exists_and_is_empty(path)
        assert path.exists()
        assert path.is_dir()

        # Calling it a second time does nothing
        ensure_directory_exists_and_is_empty(path)

        # Creating a file inside it and calling it again -> will fail
        (path / "foo").touch()
        with pytest.raises(RuntimeError):
            ensure_directory_exists_and_is_empty(path)
