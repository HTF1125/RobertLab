"""ROBERT"""
from datetime import datetime
from functools import lru_cache
import pandas as pd
import pandas_datareader as pdr


__data__ = dict(
    OecdUsLei="USALOLITONOSTSAM",
    UsUnemploymentRate="UNRATE",
    TransportAirPPI="PCU481481",
    TransportDeepSeaPPI="PCU4831114831115",
    UsM2="M2SL",
    UsPce="PCEPI",
    UsCpiAllUrban="CPIAUCSL",
    UsFedEffectiveRate="FEDFUNDS",
    UsRealGDP="GDPC1",
    UsRealGDPGovConsumtpi="GCEC1"
)


@lru_cache()
def get_data(meta: str) -> pd.DataFrame:
    """get fred data"""
    return pdr.DataReader(
        name=meta, data_source="fred", start=datetime(1900, 1, 1)
    ).astype(float)


def get_oecd_us_leading_indicator() -> pd.DataFrame:
    return get_data(meta=__data__["OecdUsLei"])


def get_us_unemployment_rate() -> pd.DataFrame:
    return get_data(meta=__data__["UsUnemploymentRate"])


def get_us_m2() -> pd.DataFrame:
    return get_data(meta=__data__["UsM2"])


def get_us_pce() -> pd.DataFrame:
    return get_data(meta=__data__["UsPce"])


def get_us_cpi_all_urban() -> pd.DataFrame:
    return get_data(meta=__data__["UsCpiAllUrban"])


def get_transport_air_ppi() -> pd.DataFrame:
    return get_data(meta=__data__["TransportAirPPI"])


def get_transport_deep_sea_ppi() -> pd.DataFrame:
    return get_data(meta=__data__["TransportDeepSeaPPI"])


def get_us_fed_effective_rate() -> pd.DataFrame:
    return get_data(meta=__data__["UsFedEffectiveRate"])


