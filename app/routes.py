from flask import Blueprint, render_template, request, flash, session
from .services.prediction_service import PredictionService
import pandas as pd
import google.generativeai as genai
import os
from dotenv import load_dotenv
import markdown
import bleach

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

main_bp = Blueprint('main', __name__)

prediction_service = PredictionService()

# Helper: Convert Markdown to safe HTML
def md_to_html(text):
    if not text:
        return ""
    html = markdown.markdown(text, extensions=['nl2br'])
    allowed_tags = ['p', 'br', 'strong', 'em', 'ul', 'ol', 'li', 'h1', 'h2', 'h3']
    return bleach.clean(html, tags=allowed_tags, attributes={}, strip=True)

@main_bp.route('/', methods=['GET', 'POST'])
def index():
    # Initialize session
    if 'chat_history' not in session:
        session['chat_history'] = []
    if 'patient_context' not in session:
        session['patient_context'] = None

    risk_level = session.get('risk_level')
    probabilities = session.get('probabilities')
    input_data = session.get('input_data', {})
    raw_chat_history = session['chat_history']

    # Prepare rendered chat history (Markdown for AI only)
    rendered_chat = []
    for msg in raw_chat_history:
        content = msg['content']
        if msg['role'] == 'model':
            content = md_to_html(content)
        rendered_chat.append({"role": msg['role'], "content": content})

    if request.method == 'POST':
        if 'age' in request.form:  # New prediction
            try:
                data = {
                    "age": float(request.form['age']),
                    "sex": int(request.form['sex']),
                    "systolic_bp": float(request.form['systolic_bp']),
                    "cholesterol": float(request.form['cholesterol']),
                    "bmi": float(request.form['bmi']),
                    "smoking": int(request.form['smoking']),
                    "diabetes": int(request.form['diabetes']),
                    "resting_hr": float(request.form['resting_hr']),
                    "physical_activity": int(request.form['physical_activity']),
                    "family_history": int(request.form['family_history'])
                }

                df = pd.DataFrame([data])
                result = prediction_service.predict(df)

                risk_level = result['risk_level']
                probabilities = result['probabilities']

                session['risk_level'] = risk_level
                session['probabilities'] = probabilities
                session['input_data'] = data
                session['patient_context'] = data

                # Create initial context message
                context_prompt = f"""
                Patient Assessment:
                • Age: {data['age']}
                • Sex: {'Male' if data['sex'] == 1 else 'Female'}
                • BMI: {data['bmi']:.1f}
                • Cholesterol: {data['cholesterol']} mg/dL
                • Systolic BP: {data['systolic_bp']} mmHg
                • Smoking: {'Yes' if data['smoking'] else 'No'}
                • Diabetes: {'Yes' if data['diabetes'] else 'No'}
                • Physical Activity: {data['physical_activity']} days/week
                • Family History: {'Yes' if data['family_history'] else 'No'}

                Predicted Risk Level: **{risk_level}**

                Please give a warm, encouraging explanation and 3 practical lifestyle tips.
                Use simple language and formatting.
                """

                model = genai.GenerativeModel('gemini-2.5-flash')
                chat = model.start_chat(history=[])
                response = chat.send_message(context_prompt)
                initial_reply = response.text

                # Reset chat history
                session['chat_history'] = [
                    {"role": "user", "content": context_prompt},
                    {"role": "model", "content": initial_reply}
                ]

            except Exception as e:
                flash(f"Error: {str(e)}", "error")

        elif 'message' in request.form:  # Chat continuation
            user_message = request.form['message'].strip()
            if user_message and session.get('patient_context'):
                # Reconstruct chat from history
                history = []
                for msg in session['chat_history']:
                    history.append({
                        "role": msg['role'].replace('model', 'assistant'),
                        "parts": [msg['content']]
                    })

                model = genai.GenerativeModel('gemini-1.5-flash')
                chat = model.start_chat(history=history)
                response = chat.send_message(user_message)
                ai_reply = response.text

                # Append to history
                session['chat_history'].append({"role": "user", "content": user_message})
                session['chat_history'].append({"role": "model", "content": ai_reply})

    # Update rendered chat after any changes
    rendered_chat = []
    for msg in session['chat_history']:
        content = msg['content']
        if msg['role'] == 'model':
            content = md_to_html(content)
        rendered_chat.append({"role": msg['role'], "content": content})

    return render_template(
        'index.html',
        risk_level=risk_level,
        probabilities=probabilities,
        input_data=input_data,
        chat_history=rendered_chat
    )