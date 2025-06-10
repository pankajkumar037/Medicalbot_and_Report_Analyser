from dotenv import load_dotenv
from PyPDF2 import PdfReader
import os
load_dotenv()
import google.generativeai as genai

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set in the environment variables.")


try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
except Exception as e:
    raise RuntimeError(f"Failed to configure Generative AI model: {str(e)}")


def analyse_medical_report(texts):

    try:
        prompt = f"""
        Analyze this medical report {texts}. Extract and summarize the following key information:
        - Patient Demographics (Name, Date of Birth, etc.)
        - Date of Report
        - Chief Complaint
        - History of Present Illness (if detailed)
        - Physical Examination Findings
        - Primary Diagnosis
        - Treatment Plan / Medications
        - Follow-up Instructions
        - Any notable lab results (if present and identifiable in tables/text).

        **Your anwer should be just summary  short and consise not a single world extra about anything gtreeting or anything .
        Present just this information in a structured, easy-to-read format with clear headings.
        Highlight any critical findings or unusual observations.
        """
        
        response = model.generate_content(prompt)
      
        return response.text

    except Exception as e:
        print(f"An error occurred during content generation: {e}")



