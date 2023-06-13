
import requests

def is_iterable(param):
    try:
        iter(param)
        return True
    except TypeError:
        return False


def has_internet_connection(url="http://www.google.com/", timeout=5):
    """_summary_

    Args:
        url (str, optional): _description_. Defaults to 'http://www.google.com/'.
        timeout (int, optional): _description_. Defaults to 5.

    Returns:
        _type_: _description_
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return True
    except requests.HTTPError as error:
        print("HTTP error: ", error)
    except requests.ConnectionError as error:
        print("Connection error: ", error)
    except requests.Timeout as error:
        print("Timeout error: ", error)
    except requests.RequestException as error:
        print("Request exception: ", error)
    print("Internet connection is inactive")
    return False


def dict_to_signature_string(data):
    # Sort dictionary by keys
    sorted_data = sorted(data.items())

    # Convert key-value pairs to a list of strings
    pairs = [f"{key}={value}" for key, value in sorted_data]

    # Join the list of strings with a delimiter
    signature_string = "&".join(pairs)

    return signature_string




# class Backtest:
#     def __call__(self, func) -> Callable:
#         def wrapper(cls, **kwargs):
#             name = kwargs.pop("name", f"Strategy-{len(cls.strategies)}")
#             if name in cls.strategies:
#                 warnings.warn(message=f"{name} already backtested.")
#                 return
#             strategy = Strategy(
#                 prices=kwargs.pop("prices", cls.prices),
#                 start=kwargs.pop("start", cls.start),
#                 end=kwargs.pop("end", cls.end),
#                 frequency=kwargs.pop("frequency", cls.frequency),
#                 commission=kwargs.pop("commission", cls.commission),
#                 shares_frac=kwargs.pop("shares_frac", cls.shares_frac),
#                 rebalance=partial(func, cls, **kwargs),
#             )

#             cls.strategies[name] = strategy
#             return strategy

#         return wrapper