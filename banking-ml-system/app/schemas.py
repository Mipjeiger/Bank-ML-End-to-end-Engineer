from pydantic import BaseModel

class CustomerData(BaseModel):
    Balance: float
    Tenure: int
    CreditScore: int
    SatisfactionScore: int
    Complain: int
    IsActiveMember: int