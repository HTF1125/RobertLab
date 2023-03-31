


import urllib.request

def access_to_internet(host='http://google.com'):
    try:
        urllib.request.urlopen(host) #Python 3.x
        return True
    except:
        return False




from src import db
import pandas as pd
import sqlalchemy as sa
import yfinance as yf
import pandas_datareader as pdr

db.client.conn.create_all()

from datetime import date, timedelta
from dateutil import parser

today = date.today()
last_working_day = today - timedelta(days=today.weekday() % 7 or 1)

with db.SessionContext() as session:
    for meta in db.models.Meta.query(session=session).all():
        print(meta.id)

        latest_record = (
            session.query(sa.func.max(db.models.EquityDailyBar.date))
            .select_from(db.models.EquityDailyBar)
            .join(db.models.Meta, db.models.Meta.id == db.models.EquityDailyBar.meta_id)
            .filter(db.models.Meta.ticker == meta.ticker)
            .scalar()
        )

        if latest_record:
            if latest_record >= last_working_day:
                continue

        source = db.models.Source.query(meta_id=meta.id).one_or_none()

        if not source or source.source == "NOTSET":
            continue

        if source.source == "yahoo":
            ticker = yf.Ticker(meta.ticker)
            hist = ticker.history(period="max", auto_adjust=False).reset_index()

        elif source.source == "naver":
            hist = (
                pdr.DataReader(meta.ticker, "naver", latest_record or "1900-1-1")
                .astype(float)
                .reset_index()
            )

        else:
            continue

        if hist.empty:
            continue

        cols = {
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
            "Dividends": "dvds",
            "Stock Splits": "splits",
        }
        hist = hist.rename(columns=cols)
        hist["open"] = hist["open"].round(2)
        hist["high"] = hist["high"].round(2)
        hist["low"] = hist["low"].round(2)
        hist["close"] = hist["close"].round(2)
        hist["pre_close"] = hist["close"].shift(1).bfill()
        hist["meta_id"] = meta.id
        if source.source == "yahoo":
            hist["tot_return"] = (hist["close"] + hist["dvds"]) / hist["pre_close"] - 1
            hist["tot_return"] = hist["tot_return"].round(10)
        elif source.source == "naver":
            hist["tot_return"] = hist["close"].pct_change().fillna(0).round(10)

        if latest_record:
            hist = hist.loc[hist.date > parser.parse(str(latest_record))]
        print(hist)
        db.models.EquityDailyBar.insert(hist, session=session)
        session.commit()
