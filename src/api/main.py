"""ROBERT"""
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import FileResponse
from src.config import TOP_FOLDER
from src import database

####################################################################################################
# create fastapi instance.
app = FastAPI(
    title="RoboAdvisor",
    description="Robo-Advisor for Wealth Management",
    version="0.0.1",
    contact={"name": "robert", "email": "hantianfeng@outlook.com"},
    openapi_url="/api/v1/openapi.json",
)

####################################################################################################
# define allowed origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

####################################################################################################
# append web html
if os.path.exists(TOP_FOLDER + "/web/build/static"):
    app.mount(
        path="/static",
        app=StaticFiles(directory=TOP_FOLDER + "/web/build/static"),
        name="static"
    )

# append docs html
if os.path.exists(TOP_FOLDER + "/docs/build/static"):
    app.mount(
        path=TOP_FOLDER + "/docs/build/static",
        app=StaticFiles(directory="../sphinx/docs/build/html", html=True),
        name="sphinx",
    )



@app.get("/test")
def test():
    return {"test":"hello"}






####################################################################################################
# Catch-all route for React Router to handle
@app.get("/{path:path}")
async def catch_all(path: str):
    return FileResponse(
        path = os.path.join(TOP_FOLDER, "web/build/index.html"),
        media_type="text/html"
    )