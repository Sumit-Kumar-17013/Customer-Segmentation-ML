from fastapi import FastAPI , HTTPException
from pydantic import BaseModel , Field
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import uvicorn
from datetime import datetime , timezone
import time 


START_TIME = time.time()

model_path = 'customer_segmentation_pipe.pkl'

try:
    model = joblib.load(model_path)
except Exception as e:
    raise RuntimeError(
    f"Model Path Coudn't Load : {e}"
    )


app = FastAPI(
    title="Customer Segmentation API",
    description="API for predicting customer segments",
    version="1.0.0",
)

#cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials = True,
    allow_methods=["*"],
    allow_headers=["*"]
)


class ModelData(BaseModel):

    Age: float = Field(...,title="Customer Age",description="Age of the customer in years.",ge=18,le=100,examples=[45])

    Income: float = Field(...,title="Annual Income",description="Customer's annual income. Enter the original income value, not the standardized value.",ge=0,examples=[60000])

    Recency: float = Field(...,title="Recency",description="Number of days since the customer's last purchase.",ge=0,examples=[30])

    Customer_Tenure_Days: float = Field(...,title="Customer Tenure",description="Number of days the customer has been associated with the company.",ge=0,examples=[400])

    total_spend: float = Field(...,title="Total Customer Spending",description="Total amount spent by the customer across all product categories.",ge=0,examples=[500])

    total_purchase: float = Field(...,title="Total Purchases",description="Total number of purchases made by the customer across available purchase channels.",ge=0,examples=[15])

    total_campaigns: float = Field(...,title="Accepted Campaigns",description="Total number of marketing campaigns accepted by the customer.",ge=0,examples=[1])

    children: float = Field(...,title="Number of Children",description="Number of children in the customer's household.",ge=0,examples=[1])

    family_size: float = Field(...,title="Family Size",description="Total number of people represented in the customer's household.",ge=1,examples=[3])

    Education_Encoded: float = Field(...,title="Education Level (Encoded)",description="Numerical encoding of the customer's education level used by the clustering model.",ge=0,examples=[1])

    Living_With_Encoded: float = Field(...,title="Living Arrangement (Encoded)",description="Encoded living arrangement used by the model: 0 represents Alone and 1 represents Partner.",ge=0,le=1,examples=[1])

    NumWebVisitsMonth: float = Field(...,title="Monthly Web Visits",description="Number of visits made by the customer to the company's website during the month.",ge=0,examples=[5])




class PredictionRecord(BaseModel):
    cluster : int
    segment : str
    message : str


def get_segment_name(cluster : int) -> str:
    if(cluster == 0):
        return "Low-Value / Browsing Customers"
    elif(cluster == 1):
        return "High-Value Customers"
    else:
        return f"Customer Cluster {cluster}"


def get_segment_message(cluster : int) -> str:
    if(cluster == 0):
        return (
            "Customers with relatively lower spending and purchase "
            "activity but higher website engagement. "
            "Recommended actions: targeted discounts, bundles, "
            "product recommendations and conversion campaigns."
)
    elif(cluster == 1):
        return(
            "High-value customers with higher income, spending and "
            "purchase activity. Recommended actions: loyalty rewards, "
            "premium offers, retention campaigns and cross-selling."
)
    else:
        return "Customer belongs to a separate discovered segment."



@app.get("/")
def home():

    return {
        "Message": "Customer Segmentation Prediction API",
        "Status": "Running",
        "Endpoint": "POST /Prediction"
    }

@app.get("/Health")
def health():
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime": time.time() - START_TIME
    }

@app.post("/Prediction" , response_model=PredictionRecord)
def Prediction(data : ModelData):
        try:
            input_data = pd.DataFrame([{
                "Age" : data.Age,
                "Income" : data.Income,
                "Recency" : data.Recency,
                "Customer_Tenure_Days": data.Customer_Tenure_Days,
                "total_spend": data.total_spend,
                "total_purchase": data.total_purchase,
                "total_campaigns": data.total_campaigns,
                "children": data.children,
                "family_size": data.family_size,
                "Education_Encoded": data.Education_Encoded,
                "Living_With_Encoded": data.Living_With_Encoded,
                "NumWebVisitsMonth": data.NumWebVisitsMonth
        }]) 
            #prediction
            prediction_result = model.predict(input_data)

            cluster = int(prediction_result[0])

            segment = get_segment_name(cluster)
            message = get_segment_message(cluster)

            return{
                "cluster": cluster,    
                "segment": segment,   
                "message": message,
            }

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {str(e)}"
            )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )


