



from typing import Optional
import pandas as pd




class GeometricBrownianMotion:


    def __init__(
        self,
        expected_return: pd.Series,
        expected_risk: pd.Series,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> None:
        self.expected_return = expected_return
        self.expected_risk = expected_risk
        self.start = start
        self.end = end


    




