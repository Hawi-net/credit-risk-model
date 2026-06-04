from pydantic import BaseModel, Field

class TransactionRequest(BaseModel):
    transactionid: str = Field(..., example="TX_1001")
    customerid: str = Field(..., example="CUST_882")
    amount: float = Field(..., example=12000.50)
    value: int = Field(..., example=12000)
    transactionstarttime: str = Field(..., example="2026-06-04T16:00:00Z")

class PredictionResponse(BaseModel):
    transaction_id: str
    risk_score_probability: float
    prediction_class: int
    credit_decision: str