from tobac_flow import io
from datetime import datetime
import os


def test_find_glm_blobs():
    assert (
        len(io.find_glm_blobs(datetime(2018, 6, 19, 19))) == 180
    ), "Error running test of find_glm_blobs(), wrong number of blobs detected"


def test_find_glm_files():
    blobs = io.find_glm_blobs(datetime(2018, 6, 19, 19))
    io.download_blob(blobs[0], save_dir="./", replicate_path=False)
    files = io.find_glm_files(
        datetime(2018, 6, 19, 19),
        save_dir="./",
        replicate_path=False,
        download_missing=False,
    )
    assert (
        len(files) == 1
    ), "Error running test of find_glm_files(), wrong number of files detected"
    for f in files:
        os.remove(f)


def test_find_abi_files():
    test_date = datetime(2000, 1, 1, 12)
    files = io.find_abi_files(
        test_date,
        view="C",
        channel=13,
        save_dir="./",
        replicate_path=False,
        download_missing=True,
    )
    assert (
        len(files) == 1
    ), "Error running test of find_abi_files(), wrong number of files detected"
    for f in files:
        os.remove(f)


def test_get_goes_date():
    test_date = datetime(2000, 1, 1, 12)
    assert [
        io.get_goes_date(f.name)
        for f in io.find_abi_blobs(test_date, view="C", channel=1)
    ][
        0
    ] == test_date, (
        "Error running test of get_goes_date(), file date does not match expected date"
    )


def test_download_blobs():
    test_date = datetime(2000, 1, 1, 12)
    blobs = io.find_abi_blobs(test_date, view="C", channel=13)
    blob = blobs[0]
    blob_name = blob.name
    io.download_blob(blob, save_dir="./", replicate_path=False, clobber=True)
    assert os.path.exists(
        "./" + os.path.split(blob_name)[-1]
    ), "Error running test of download_goes_blobs(), file not located"
    os.remove("./" + os.path.split(blob_name)[-1])


def test_find_abi_blobs():
    assert (
        len(
            sum(
                [
                    io.find_abi_blobs(datetime(2000, 1, 1, 12), view="C", channel=i + 1)
                    for i in range(16)
                ],
                [],
            )
        )
        == 15
    ), "Error running test of find_abi_blobs(), wrong number of blobs detected"
    assert (
        len(
            sum(
                [
                    io.find_abi_blobs(datetime(2000, 1, 1, 12), view="F", channel=i + 1)
                    for i in range(16)
                ],
                [],
            )
        )
        == 1
    ), "Error running test of find_abi_blobs(), wrong number of blobs detected"
