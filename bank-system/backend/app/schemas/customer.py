from pydantic import BaseModel

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
    BalancePerProduct: float
    AgeRisk: bool
    HighValueCustomer: bool
    LowCreditRisk: bool
    ComplainFlag: bool
    LowSatisfaction: bool

class PredictionRequest(BaseModel):
    task: str
    data: dict