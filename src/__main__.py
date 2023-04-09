import os
import sys
import argparse
from datetime import datetime
import uvicorn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../..")))



####################################################################################################
# parse arguments
parse = argparse.ArgumentParser(description="Client Interface")
parse.add_argument("-s", "--script", default=None)
parse.add_argument("-d", "--date", default=datetime.today())
args = parse.parse_args()


if not args.script:
    uvicorn.run(app="src.api.main:app", reload=True)
