from pydantic import BaseModel

class CustomerData(BaseModel):
    Balance: float
    Tenure: int
    customer_id: object
    CreditScore: int
    SatisfactionScore: int
    Complain: int
    IsActiveMember: int

class PredictionRequest(BaseModel):
    task: str
    data: dict