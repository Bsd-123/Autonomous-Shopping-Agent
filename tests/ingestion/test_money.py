"""Tests for the money rules: integer agorot, per-kilogram weighed prices, no unit price."""

from datetime import UTC, datetime
from decimal import Decimal

import polars as pl
import pytest

from shopping_agent.ingestion.adapters import ADAPTERS, PRICE_SCHEMA
from shopping_agent.ingestion.canonical import SCHEMAS, build_canonical, write_canonical
from shopping_agent.ingestion.money import (
    MoneyError,
    is_money_column,
    require_integer_money,
    weighed_not_per_kilogram,
)

VERSION = "2026-11-15"
RUN_AT = datetime(2026, 11, 15, 3, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def tables(raw_dir_module) -> dict[str, pl.DataFrame]:
    return build_canonical(raw_dir_module, VERSION, RUN_AT)


@pytest.fixture(scope="module")
def raw_dir_module(request) -> object:
    return request.config.rootpath / "tests" / "fixtures" / "raw"


class TestIntegerAgorot:
    def test_price_agorot_is_an_integer_column(self, tables):
        assert tables["prices"].schema["price_agorot"] == pl.Int64()
        assert tables["prices"]["price_agorot"].null_count() == 0

    def test_no_money_column_is_a_float(self, tables):
        for name, table in tables.items():
            for column, dtype in table.schema.items():
                if is_money_column(column):
                    assert dtype.is_integer(), f"{name}.{column}"

    def test_the_declared_schemas_follow_the_rule_too(self):
        empty = {name: pl.DataFrame(schema=schema) for name, schema in SCHEMAS.items()}

        require_integer_money(empty)

    def test_prices_are_exact(self, tables):
        # 1.50 shekels is 150 agorot, with no float in between.
        muffin = tables["prices"].filter(product_key="osher_ad:7290006775405").row(0, named=True)

        assert muffin["price_agorot"] == 150
        assert type(muffin["price_agorot"]) is int

    def test_the_written_files_hold_integers(self, raw_dir_module, tmp_path):
        folder = write_canonical(raw_dir_module, tmp_path / "canonical", VERSION, RUN_AT)

        # One file, or one per chain when prices are partitioned: read whichever exists.
        prices = pl.concat(
            [pl.read_parquet(file) for file in sorted(folder.rglob("prices.parquet"))]
        )

        assert prices.schema["price_agorot"] == pl.Int64()
        assert not any(dtype.is_float() for dtype in prices.schema.values())


class TestRequireIntegerMoney:
    def frame(self, **columns) -> dict[str, pl.DataFrame]:
        return {"prices": pl.DataFrame(columns)}

    def test_integer_agorot_pass(self):
        require_integer_money(self.frame(price_agorot=[590], price_flag=["ok"], lat=[32.08]))

    @pytest.mark.parametrize(
        "values",
        [[5.9], [Decimal("5.90")], ["590"]],
        ids=["float", "decimal", "text"],
    )
    def test_a_money_column_that_is_not_an_integer_is_rejected(self, values):
        with pytest.raises(MoneyError, match="prices.price_agorot must be whole agorot"):
            require_integer_money(self.frame(price_agorot=values))

    @pytest.mark.parametrize("column", ["price", "sale_price", "basket_cost"])
    def test_a_money_column_is_named_with_agorot(self, column):
        with pytest.raises(MoneyError, match="named with '_agorot'"):
            require_integer_money(self.frame(**{column: [590]}))

    @pytest.mark.parametrize("column", ["unit_price", "unit_price_agorot", "Unit_Price_Per_Litre"])
    def test_a_unit_price_column_is_rejected(self, column):
        with pytest.raises(MoneyError, match="a unit price is not stored"):
            require_integer_money(self.frame(price_agorot=[590], **{column: [59]}))

    def test_the_run_stops_when_a_table_breaks_the_rule(self, tables):
        broken = dict(tables)
        broken["prices"] = tables["prices"].with_columns(unit_price=pl.col("price_agorot") / 1000)

        with pytest.raises(MoneyError, match="prices.unit_price"):
            require_integer_money(broken)

    def test_other_columns_are_not_money(self):
        for column in ("price_flag", "lat", "base_quantity", "rows_in", "size_value"):
            assert not is_money_column(column)
        for column in ("price_agorot", "total_agorot", "price", "travel_cost"):
            assert is_money_column(column)


class TestNoUnitPrice:
    def test_no_table_has_a_unit_price_column(self, tables):
        for name, table in tables.items():
            assert not [column for column in table.columns if "unit_price" in column], name

    def test_no_schema_declares_one(self):
        for schema in SCHEMAS.values():
            assert not [
                column for column in schema.names() if "unit" in column and "price" in column
            ]


class TestWeighedProducts:
    def test_weighed_products_are_marked(self, tables):
        products = tables["products"]

        assert products["is_weighed"].null_count() == 0
        assert products.filter("is_weighed").height > 0
        assert products.filter(product_key="maayan_2000:116")["is_weighed"].item() is True

    def test_a_weighed_product_carries_its_per_kilogram_price(self, tables):
        # White potatoes at Maayan 2000, store 18: published at 5.90 per kilogram.
        potatoes = tables["prices"].filter(product_key="maayan_2000:116", store_id="maayan_2000:18")

        assert potatoes["price_agorot"].unique().to_list() == [590]

    def test_the_weight_field_does_not_change_the_price(self, raw_dir_module):
        # The potatoes row has quantity 0.15; the price is still the one per kilogram.
        adapter = next(adapter for adapter in ADAPTERS if adapter.chain == "maayan_2000")
        row = adapter.read_prices(raw_dir_module).filter(item_code="116").row(0, named=True)

        assert (row["quantity"], row["unit"], row["is_weighed"]) == (0.15, "קילוגרם", True)
        assert row["price"] == Decimal("5.90")

    def test_weighed_prices_in_the_samples_are_plausible_per_kilogram(self, tables):
        weighed = tables["prices"].join(
            tables["products"].filter("is_weighed"), on="product_key", how="semi"
        )

        assert weighed.height > 0
        # Between 1 and 400 shekels per kilogram.
        assert weighed["price_agorot"].is_between(100, 40_000).all()


class TestWeighedNotPerKilogram:
    def listings(self, *rows: tuple) -> pl.DataFrame:
        columns = ("chain", "item_code", "name", "is_weighed", "unit_of_measure", "price")
        data = [dict(zip(columns, row, strict=True)) for row in rows]
        return pl.DataFrame(data, schema={name: PRICE_SCHEMA[name] for name in columns}).select(
            pl.all(),
            quantity=pl.lit(1.0),
            unit=pl.lit(None, dtype=pl.String),
        )

    def test_only_weighed_products_stated_in_grams_are_listed(self):
        prices = self.listings(
            ("maayan_2000", "1", "פילה לברק", True, "100 גרם", Decimal("49.90")),
            ("maayan_2000", "2", "חזה עוף", True, "1 קילוגרם", Decimal("46.90")),
            ("yesh_hesed", "3", "גזר ארוז", True, "1קילוגרם", Decimal("3.90")),
            ("osher_ad", "4", "אבוקדו", True, "יחידה", Decimal("5.90")),
            ("osher_ad", "5", "חומוס", True, 'ק"ג', Decimal("39.90")),
            ("yesh_hesed", "6", "גבינה", False, "100 גרם", Decimal("9.90")),
            ("maayan_2000", "7", "בשר", True, None, Decimal("84.90")),
        )

        listed = weighed_not_per_kilogram(prices)

        assert listed["item_code"].to_list() == ["1"]
        assert listed.columns == [
            "chain",
            "item_code",
            "name",
            "quantity",
            "unit",
            "unit_of_measure",
            "price",
        ]

    def test_a_product_listed_in_many_stores_appears_once(self):
        row = ("maayan_2000", "1", "פילה לברק", True, "100 גרם", Decimal("49.90"))

        assert weighed_not_per_kilogram(self.listings(row, row, row)).height == 1

    def test_the_samples_can_be_checked(self, raw_dir_module):
        prices = pl.concat([adapter.read_prices(raw_dir_module) for adapter in ADAPTERS])

        listed = weighed_not_per_kilogram(prices)

        assert listed.height <= prices.filter("is_weighed").height
        assert listed["unit_of_measure"].str.contains("גרם").all()
