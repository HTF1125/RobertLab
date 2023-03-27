import os
import sys
import pytz
import uvicorn
import argsparse
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../..")))

from src import app

uvicorn.run(app)


def is_during_trading_session(zone="Asia/Seoul") -> bool:
    """_summary_

    Args:
        zone (str, optional): _description_. Defaults to "Asia/Seoul".

    Returns:
        bool: _description_
    """
    now = datetime.now(tz=pytz.timezone(zone=zone))
    trading_start = now.replace(hour=9, minute=0, second=0, microsecond=0)
    trading_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return trading_start <= now <= trading_end


