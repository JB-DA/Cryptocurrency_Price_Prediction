-- Exported from QuickDBD: https://www.quickdatabasediagrams.com/
-- NOTE! If you have used non-SQL datatypes in your design, you will have to change these here.


CREATE TABLE "assets" (
    "asset_id" VARCHAR   NOT NULL,
    "name" VARCHAR   NOT NULL,
    "type_is_crypto" INT   NOT NULL,
    "data_quote_start" VARCHAR   NOT NULL,
    "data_quote_end" VARCHAR   NOT NULL,
    "data_orderbook_start" VARCHAR   NOT NULL,
    "data_orderbook_end" VARCHAR   NOT NULL,
    "data_trade_start" VARCHAR   NOT NULL,
    "data_trade_end" VARCHAR   NOT NULL,
    "data_quote_count" VARCHAR   NOT NULL,
    "data_trade_count" VARCHAR   NOT NULL,
    "data_symbols_count" INT   NOT NULL,
    "volume_1hrs_usd" FLOAT   NOT NULL,
    "volume_1day_usd" FLOAT   NOT NULL,
    "volume_1mth_usd" FLOAT   NOT NULL,
    "price_usd" FLOAT   NOT NULL,
    CONSTRAINT "pk_assets" PRIMARY KEY (
        "asset_id"
     )
);

CREATE TABLE "periods" (
    "period_id" VARCHAR   NOT NULL,
    "length_seconds" INT   NOT NULL,
    "length_months" INT   NOT NULL,
    "unit_count" INT   NOT NULL,
    "unit_name" VARCHAR   NOT NULL,
    "display_name" VARCHAR   NOT NULL
);

CREATE TABLE "current_rates" (
    "time" VARCHAR   NOT NULL,
    "asset_id_base" VARCHAR   NOT NULL,
    "asset_id_quote" VARCHAR   NOT NULL,
    "rate" FLOAT   NOT NULL
);

CREATE TABLE "exchanges" (
    "exchange_id" VARCHAR   NOT NULL,
    "website" VARCHAR   NOT NULL,
    "name" VARCHAR   NOT NULL,
    "data_start" VARCHAR   NOT NULL,
    "data_end" VARCHAR   NOT NULL,
    "data_quote_start" VARCHAR   NOT NULL,
    "data_quote_end" VARCHAR   NOT NULL,
    "data_orderbook_start" VARCHAR   NOT NULL,
    "data_orderbook_end" VARCHAR   NOT NULL,
    "data_trade_start" VARCHAR   NOT NULL,
    "data_trade_end" VARCHAR   NOT NULL,
    "data_symbols_count" INT   NOT NULL,
    "volume_1hrs_usd" FLOAT   NOT NULL,
    "volume_1day_usd" FLOAT   NOT NULL,
    "volume_1mth_usd" FLOAT   NOT NULL
);

CREATE TABLE "historic_trades" (
    "asset_id" VARCHAR   NOT NULL,
    "time_period_start" VARCHAR   NOT NULL,
    "time_period_end" VARCHAR   NOT NULL,
    "time_open" VARCHAR   NOT NULL,
    "time_close" VARCHAR   NOT NULL,
    "price_open" FLOAT   NOT NULL,
    "price_high" FLOAT   NOT NULL,
    "price_low" FLOAT   NOT NULL,
    "price_close" FLOAT   NOT NULL,
    "volume_traded" FLOAT   NOT NULL,
    "trades_count" INT   NOT NULL
);

ALTER TABLE "assets" ADD CONSTRAINT "fk_assets_asset_id" FOREIGN KEY("asset_id")
REFERENCES "historic_trades" ("asset_id");

ALTER TABLE "current_rates" ADD CONSTRAINT "fk_current_rates_asset_id_base" FOREIGN KEY("asset_id_base")
REFERENCES "assets" ("asset_id");

