from pydantic import BaseModel
from typing import List

class MethaneSchema(BaseModel):
    Crop: str
    Run_ID: int
    Day: int
    Temperature_C: float
    pH: float
    VFA_mgL: float
    VS_percent: float
    CN_Ratio: float
    Lignin_percent: float
    Daily_Methane_m3: float
