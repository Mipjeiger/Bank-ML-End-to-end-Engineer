from pydantic import BaseModel

class Request(BaseModel):
    customer_id: str