import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../..")))

from src import app

import uvicorn

uvicorn.run(app)
