import sys


def terminal_progress(
    current_bar: int,
    total_bar: int,
    prefix: str = "",
    suffix: str = "",
    bar_length: int = 50,
) -> None:
    """
    Calls in a loop to create a terminal progress bar.

    Args:
        current_bar (int): Current iteration.
        total_bar (int): Total iteration.
        prefix (str, optional): Prefix string. Defaults to ''.
        suffix (str, optional): Suffix string. Defaults to ''.
        bar_length (int, optional): Character length of the bar.
            Defaults to 50.

    References:
        https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a
    """
    # Calculate the percent completed.
    percents = current_bar / float(total_bar)
    # Calculate the length of bar.
    filled_length = int(round(bar_length * current_bar / float(total_bar)))
    # Fill the bar.
    block = "█" * filled_length + "-" * (bar_length - filled_length)
    # Print new line.
    sys.stdout.write(f"\r{prefix} |{block}| {percents:.2%} {suffix}")

    if current_bar == total_bar:
        sys.stdout.write("\n")
    sys.stdout.flush()
