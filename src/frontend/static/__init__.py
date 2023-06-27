"""ROBERT"""

import os
from typing import List


def all_filenames() -> List[str]:
    directory = os.path.dirname(__file__)
    # Get all filenames in the directory
    filenames = os.listdir(directory)
    return [
        os.path.join(directory, filename)
        for filename in filenames
        if not filename.startswith("__")
    ]
