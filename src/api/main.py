"""ROBERT"""
import os
import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import FileResponse
from ..config import TOP_FOLDER

logger = logging.getLogger(__name__)

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


@app.get("/strategies/ew")
def ew():
    import yfinance as yf
    from src.core.strategies.strategies import backtest
    from src.core.portfolios import Optimizer

    @backtest
    def EW(strategy):
        """equal"""
        return Optimizer.from_prices(prices=strategy.reb_prices).uniform_allocation()

    result = EW(yf.download("SPY, AGG")["Adj Close"], start="2010-1-1")


    return {"value": result.value.to_dict()}



####################################################################################################
# Catch-all route for React Router to handle
@app.get("/{path:path}")
async def catch_all(path: str):
    logger.error(path)
    return FileResponse(
        path = os.path.join(TOP_FOLDER, "web/build/index.html"),
        media_type="text/html"
    )