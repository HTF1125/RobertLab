from datetime import datetime
from dateutil import parser
import pytz


def is_during_trading_session(zone="Asia/Seoul", start="9:00", end="15:30") -> bool:
    """_summary_

    Args:
        zone (str, optional): _description_. Defaults to "Asia/Seoul".

    Returns:
        bool: True if in trading session.
    """
    timezone = pytz.timezone(zone=zone)
    now = datetime.now(tz=timezone)
    start = parser.parse(now.strftime("%Y-%m-%d") + " " + start).astimezone(timezone)
    end = parser.parse(now.strftime("%Y-%m-%d") + " " + end).astimezone(timezone)
    return start <= now <= end


def is_trading_session_kr() -> bool:
    """this is passthrough function"""
    return is_during_trading_session(zone="Asia/Seoul", start="9:00", end="15:30")


def is_trading_session_us() -> bool:
    """this is passthrough function"""
    return is_during_trading_session(zone="America/New_York", start="9:30", end="16:00")
