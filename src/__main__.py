import os
import sys
import pytz
import uvicorn
from datetime import datetime
from dateutil import parser

sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../..")))

from src import app

# uvicorn.run(app)


def is_during_trading_session(zone="Asia/Seoul", start="9:00", end="15:30") -> bool:
    """_summary_

    Args:
        zone (str, optional): _description_. Defaults to "Asia/Seoul".

    Returns:
        bool: True if in trading session.
    """
    tz=pytz.timezone(zone=zone)
    now = datetime.now(tz=tz)
    start = parser.parse(now.strftime("%Y-%m-%d") + " " + start).astimezone(tz)
    end = parser.parse(now.strftime("%Y-%m-%d") + " " + end).astimezone(tz)
    return start <= now <= end


def is_trading_session_kr() -> bool:
    """this is passthrough function"""
    return is_during_trading_session(zone="Asia/Seoul", start="9:00", end="15:30")

def is_trading_session_us() -> bool:
    """this is passthrough function"""
    return is_during_trading_session(zone="America/New_York", start="9:30", end="16:00")

print(is_trading_session_kr())
print(is_trading_session_us())
