"""Shared test fixtures."""

import shutil
from pathlib import Path

import pytest

FIXTURES_RAW = Path(__file__).parent / "fixtures" / "raw"

SHUFERSAL_PRICE_HEADER = (
    "found_folder,file_name,chainid,subchainid,storeid,bikoretno,priceupdatetime,"
    "itemcode,lastsaledatetime,itemtype,itemname,manufacturename,manufacturecountry,"
    "manufactureitemdescription,unitqty,quantity,unitofmeasure,bisweighted,"
    "qtyinpackage,itemprice,unitofmeasureprice,allowdiscount,itemstatus"
)

# Constructed rows in Shufersal's column and value format. Store 681 belongs
# to another Shufersal sub-chain; store 611 is a Yesh Hesed branch. Shufersal
# timestamps carry no milliseconds.
SHUFERSAL_PRICE_ROWS = (
    "app_data/dumps/Shufersal,Price7290027600007-005-681-20261114-010000.xml,"
    "7290027600007,005,681,9,2026-11-14T00:10:00,3574661098531,2026-11-13T11:29:55,1,"
    "תח.גוף ניוטרוגינה מ400,ש.שסטוביץ,GR,תח.גוף ניוטרוגינה מ400,מיליליטר,400.00,"
    "100 מיליליטר,0,1,37.90,9.48,1,1",
    ",Price7290027600007-006-611-20261114-010000.xml,,006,611,,,3838824438696,"
    "2026-11-13T11:00:09,,קרם סופרם-צבע לשיער-6-0,סלובניה,SI,קרם סופרם-צבע לשיער-6-0,"
    "יחידות,1.00,יחידות,,,49.90,49.90,,",
    ",,,,,,,7290000000015,2026-11-13T12:00:00,,פריט לדוגמה,'',IL,פריט לדוגמה,,,,,,12.50,12.50,,",
)


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """The real sample files plus a constructed Shufersal price file.

    No real Shufersal price sample is available, so the Yesh Hesed price path
    runs on constructed rows paired with the real Shufersal store file.
    """
    raw = tmp_path / "raw"
    shutil.copytree(FIXTURES_RAW, raw)
    (raw / "shufersal" / "price_file_shufersal.csv").write_text(
        "\n".join([SHUFERSAL_PRICE_HEADER, *SHUFERSAL_PRICE_ROWS]) + "\n", encoding="utf-8"
    )
    return raw
