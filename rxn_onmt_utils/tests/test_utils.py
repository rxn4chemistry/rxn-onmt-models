from pathlib import Path

import pytest
from freezegun import freeze_time
from rxn.utilities.files import named_temporary_path

from rxn_onmt_utils.utils import (
    ensure_directory_exists_and_is_empty,
    get_multiplier,
    get_multipliers,
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


def test_get_multipliers() -> None:
    assert get_multipliers(1, 1) == (1, 1)
    assert get_multipliers(123, 123) == (1, 1)
    assert get_multipliers(1, 12) == (12, 1)
    assert get_multipliers(12, 1) == (1, 12)
    assert get_multipliers(3, 27) == (9, 1)
    assert get_multipliers(27, 3) == (1, 9)

    assert get_multipliers(2, 3) == (3, 2)
    assert get_multipliers(10, 6) == (3, 5)

    invalid_pairs = [
        (0, 0),
        (-1, -1),
        (-1, 1),
    ]
    for a, b in invalid_pairs:
        with pytest.raises(ValueError):
            _ = get_multipliers(a, b)


def test_get_multiplier() -> None:
    assert get_multiplier(1, 1) == 1
    assert get_multiplier(123, 123) == 1
    assert get_multiplier(1, 12) == 12
    assert get_multiplier(6, 12) == 2
    assert get_multiplier(3, 27) == 9

    invalid_pairs = [
        (0, 0),
        (-1, -1),
        (-1, 1),
        (2, 1),
        (2, 3),
        (3, 2),
    ]
    for a, b in invalid_pairs:
        with pytest.raises(ValueError):
            _ = get_multiplier(a, b)
