from src import db
import yfinance as yf

db.client.conn.create_all()

with db.SessionContext() as session:
    
    for meta in db.models.Meta.query(session=session).all():
        
        source = db.models.Source.query(meta_id = meta.id).one_or_none()
        
        if source.source == "NOTSET": continue
        
        if source.source == "yahoo":
            print(meta.ticker)
            ticker = yf.Ticker(meta.ticker)
            hist = ticker.history(period="max", auto_adjust=False)
            hist.columns = hist.columns.str.replace(" ", "_")
            hist.columns = hist.columns.str.lower()
            hist.index.name = "asofdate"
            hist['open'] = hist['open'].round(2)
            hist['high'] = hist['high'].round(2)
            hist['low'] = hist['low'].round(2)
            hist['close'] = hist['close'].round(2)
            hist['tot_return'] = hist['close'].add(hist['dividends']).pct_change().fillna(0).round(5)
            hist['adj_return'] = hist['adj_close'].pct_change().fillna(0).round(5)
            hist = hist.reset_index()
            hist["meta_id"] = meta.id + 100
            db.models.EquityDailyBar.insert(hist[["meta_id", "asofdate", "open", "high", "low", "close"]])
