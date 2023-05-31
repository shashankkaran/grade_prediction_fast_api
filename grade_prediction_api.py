import pickle
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Load the pre-trained model
with open('final_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the input data model
class GradePredictionInput(BaseModel):
    PRNNO: str
    SUBJECT: str
    INSEM: int
    TW: int
    PR: int

# Define the FastAPI app
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Define the prediction endpoint
@app.post("/predict_grade")
def predict_grade(data: GradePredictionInput):
    dictionary = {'DM': 1, 'LD': 2, 'DSA': 3, 'OOP': 4, 'BCN': 5, 'M3': 6, 'PA': 7, 'DMSL': 8, 'CG': 9, 'SE': 10, 'M1': 11, 'PH': 12, 'SME': 13, 'BE': 14, 'PPS': 15, 'CH': 16, 'M2': 17, 'SM': 18, 'MECH': 19, 'PHY': 20}

    # Get the subject code from the dictionary
    subject_code = dictionary[data.SUBJECT]

    # Use the model to predict the grade
    output = model.predict([[data.INSEM, data.TW, data.PR, subject_code]])

    # Get the confidence level
    confidence = round(100 * output[0].max(), 2).item()

    # Get the predicted grade as an integer
    predicted_grade = int(output[0].item())

    # Define the grade labels
    grade_labels = {1: 'Distinction', 2: 'First Class', 3: 'Fail'}

    # Get the grade label based on the predicted grade
    grade = grade_labels.get(predicted_grade, 'Unknown')

    return {
        "predicted_grade": grade ,
        "confidence": confidence
    }

# Define the homepage
# @app.get("/")
# def home(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
