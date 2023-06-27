"""ROBERT"""
import os
import time
import logging
from fastapi import FastAPI, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from .dependencies import get_session


logger = logging.getLogger(__name__)
directory = os.path.dirname(os.path.abspath(__file__))

################################################################################
# create fastapi instance.
app = FastAPI(
    title="RoboAdvisor",
    description="Robo-Advisor for Wealth Management",
    version="0.0.1",
    openapi_url="/api/v1/openapi.json",
)

################################################################################
# define allowed origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

################################################################################
# append web html
if os.path.exists(os.path.join(directory, "/web/build/static")):
    app.mount(
        path="/static",
        app=StaticFiles(directory=os.path.join(directory, "/web/build/static")),
        name="static",
    )


@app.middleware(middleware_type="http")
async def add_process_time_header(request, call_next):
    start = time.time()
    response = await call_next(request)
    process_time = time.time() - start
    response.headers["X-Process-Time"] = str(f"{process_time:0.4f} secs")
    return response


################################################################################
# Catch-all route for React Router to handle
@app.get("/{path:path}")
async def catch_all(path: str):
    logger.error(path)
    return FileResponse(
        path=os.path.join(os.path.join(directory, "/web/build/index.html")),
        media_type="text/html",
    )
