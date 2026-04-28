from pydantic import BaseModel
from typing import Optional

class CustomerData(BaseModel):
    customer_id: str
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: bool
    IsActiveMember: bool
    EstimatedSalary: float
    Exited: bool
    Complain: bool
    SatisfactionScore: int
    CardType: str
    PointEarned: int
    RiskScore: int
    BalancePerProduct: Optional[float] = None
    AgeRisk: Optional[bool] = None
    HighValueCustomer: Optional[bool] = None
    LowCreditRisk: Optional[bool] = None
    ComplainFlag: Optional[bool] = None
    LowSatisfaction: Optional[bool] = None

class PredictionRequest(BaseModel):
    task: str
    data: dict

class PredictionResponse(BaseModel):
    model: str
    customer_id: str
    probability: float
    prediction: int
    risk: str
    cached: bool = False