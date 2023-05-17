"""ROBERT"""
import os
import logging
from typing import List
from fastapi import FastAPI, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from core.strategies import backtest
from core.portfolios import Optimizer
from config import SRC_FOLDER
from sqlmodel import Session
from sqlalchemy.exc import IntegrityError
from database import MetaBase, Meta
from .dependencies import get_session


logger = logging.getLogger(__name__)

####################################################################################################
# create fastapi instance.
app = FastAPI(
    title="RoboAdvisor",
    description="Robo-Advisor for Wealth Management",
    version="0.0.1",
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
if os.path.exists(SRC_FOLDER + "/web/build/static"):
    app.mount(
        path="/static",
        app=StaticFiles(directory=SRC_FOLDER + "/web/build/static"),
        name="static",
    )

# append docs html
if os.path.exists(SRC_FOLDER + "/docs/build/static"):
    app.mount(
        path=SRC_FOLDER + "/docs/build/static",
        app=StaticFiles(directory="../sphinx/docs/build/html", html=True),
        name="sphinx",
    )


@app.get("/strategies/ew")
def ew():
    import yfinance as yf

    @backtest
    def EW(strategy):
        """equal"""
        return Optimizer.from_prices(prices=strategy.reb_prices).uniform_allocation()

    result = EW(yf.download("SPY, AGG")["Adj Close"], start="2010-1-1")

    return {"value": result.value.to_dict()}


@app.post("/meta/create", response_model=Meta)
async def add_meta(meta_in: MetaBase, session: Session = Depends(get_session)):
    obj = Meta(ticker=meta_in.ticker, name=meta_in.name)
    session.add(obj)
    try:
        session.commit()
        session.refresh(obj)
        return obj
    except IntegrityError as exc:
        raise HTTPException(status_code=404, detail="Hero not found") from exc

@app.get("/meta", response_model=List[MetaBase])
async def meta(session: Session = Depends(get_session), limit: int = 10):
    return session.query(Meta).limit(limit).all()


####################################################################################################
# Catch-all route for React Router to handle
@app.get("/{path:path}")
async def catch_all(path: str):
    logger.error(path)
    return FileResponse(
        path=os.path.join(SRC_FOLDER, "web/build/index.html"), media_type="text/html"
    )
