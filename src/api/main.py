from pydantic import BaseModel
from datetime import date
from typing import Optional, List
import os
from fastapi import FastAPI, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from src.config import APISRC_FOLDER
from src.database import client


####################################################################################################
# create fastapi instance.
app = FastAPI(
    title="RoboAdvisor",
    description="Robo-Advisor for Wealth Management",
    version="0.0.1",
    contact={"name": "robert", "email": "roberthan1125@gmail.com"},
    openapi_url="/api/v1/openapi.json",
)


####################################################################################################
# define allowed origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if os.path.exists(APISRC_FOLDER + "/build"):
    app.mount(
        "/static", StaticFiles(directory=APISRC_FOLDER + "/build/static"), name="static"
    )

####################################################################################################
# append additional rountes to fastapi
if os.path.exists("../sphinx"):
    app.mount(
        path="../sphinx",
        app=StaticFiles(directory="../sphinx/docs/build/html", html=True),
        name="sphinx",
    )
