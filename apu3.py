import streamlit as st
import pandas as pd
import requests
import json
import os
from dotenv import load_dotenv
import speech_recognition as sr
from gtts import gTTS
from groq import Groq
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.offline as pyo

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Load dataset
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Generate billing report using GROQ API
def generate_billing_report(patient_details):
    url = "https://api.groq.com/v1/generate"
    headers = {
        'Authorization': f'Bearer {GROQ_API_KEY}',
        'Content-Type': 'application/json'
    }
    prompt = (
        f"Generate a detailed billing report for the following patient:\n"
        f"Name: {patient_details['Name']}\n"
        f"Age: {patient_details['Age']}\n"
        f"Gender: {patient_details['Gender']}\n"
        f"Blood Type: {patient_details['Blood Type']}\n"
        f"Medical Condition: {patient_details['Medical Condition']}\n"
        f"Date of Admission: {patient_details['Date of Admission']}\n"
        f"Doctor: {patient_details['Doctor']}\n"
        f"Hospital: {patient_details['Hospital']}\n"
        f"Insurance Provider: {patient_details['Insurance Provider']}\n"
        f"Billing Amount: {patient_details['Billing Amount']}\n"
        f"Room Number: {patient_details['Room Number']}\n"
        f"Admission Type: {patient_details['Admission Type']}\n"
        f"Discharge Date: {patient_details['Discharge Date']}\n"
        f"Medication: {patient_details['Medication']}\n"
        f"Test Results: {patient_details['Test Results']}\n\n"
        f"Please include insights and findings from the billing analysis, summarizing the overall accuracy and efficiency of the billing process. Also, suggest any potential improvements for the patient care process."
    )
    payload = {
        "model": "text-davinci-003",
        "prompt": prompt,
        "max_tokens": 500
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        result = response.json()
        return result['choices'][0]['text']
    else:
        return "Error generating report from GROQ API."

# Generate health suggestions using GROQ API
def generate_health_suggestions(patient_details):
    url = "https://api.groq.com/v1/generate"
    headers = {
        'Authorization': f'Bearer {GROQ_API_KEY}',
        'Content-Type': 'application/json'
    }
    prompt = (
        f"Generate health suggestions, including food and exercise recommendations, for a patient with the following details:\n"
        f"Age: {patient_details['Age']}\n"
        f"Medical Condition: {patient_details['Medical Condition']}\n\n"
        f"Please provide detailed and personalized suggestions."
    )
    payload = {
        "model": "text-davinci-003",
        "prompt": prompt,
        "max_tokens": 500
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        result = response.json()
        return result['choices'][0]['text']
    else:
        return "Error generating health suggestions from GROQ API."

# Voice to text function
def voice_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        st.write("<h2 style='color: #00FFFF;'>Listening...</h2>", unsafe_allow_html=True)
        audio = r.listen(source)
        try:
            user_input = r.recognize_google(audio, language="en-US")
            st.write(f"<h2 style='color: #ff6347;'>You said:</h2><p style='color: #FFFFFF;'>{user_input}</p>", unsafe_allow_html=True)
            return user_input
        except sr.UnknownValueError:
            st.write("<h2 style='color: #f00;'>Sorry, I didn't understand that. Please try again.</h2>", unsafe_allow_html=True)
        except sr.RequestError as e:
            st.write(f"<h2 style='color: #f00;'>Error: {e}</h2>", unsafe_allow_html=True)
    return None

# Generate blog post using GROQ API
def generate_blog_post(topic):
    # Generate the blog post heading
    heading_prompt = PromptTemplate(
        input_variables=["topic"],
        template="Heading: Generate a catchy heading for a blog post about {topic}.",
    )
    heading = groq_chat(heading_prompt.format(topic=topic))["response"].split("Heading: ")[1].strip()

    # Generate the blog post introduction
    intro_prompt = PromptTemplate(
        input_variables=["topic"],
        template="Introduction: Write an engaging introduction for a blog post about {topic}.",
    )
    intro = groq_chat(intro_prompt.format(topic=topic))["response"].split("Introduction: ")[1].strip()

    # Generate the blog post content
    content_prompt = PromptTemplate(
        input_variables=["topic"],
        template="Content: Write the main content for a blog post about {topic}. Provide detailed information, examples, and insights.",
    )
    content = groq_chat(content_prompt.format(topic=topic))["response"].split("Content: ")[1].strip()

    # Generate the blog post summary
    summary_prompt = PromptTemplate(
        input_variables=["topic"],
        template="Summary: Summarize the key points of a blog post about {topic}.",
    )
    summary = groq_chat(summary_prompt.format(topic=topic))["response"].split("Summary: ")[1].strip()

    return heading, intro, content, summary

# Create a PDF bill for the patient
def create_pdf(patient_details, report):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Patient Billing Report", ln=True, align="C")
    pdf.ln(10)

    for key, value in patient_details.items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True, align="L")

    pdf.ln(10)
    pdf.multi_cell(0, 10, txt=report)
    return pdf

def main():
    st.title("AI Blog and Billing Generation System")

    # Add customization options to the sidebar
    st.sidebar.title('Select an LLM')
    model = st.sidebar.selectbox('Choose a model', ['mixtral-8x7b-32768', 'llama2-70b-4096'])
    conversational_memory_length = st.sidebar.slider('Conversational memory length:', 1, 10, value=5)
    memory = ConversationBufferWindowMemory(k=conversational_memory_length)

    # Session state variable
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    else:
        for message in st.session_state.chat_history:
            memory.save_context({'input': message['human']}, {'output': message['AI']})

    # Initialize Groq Langchain chat object and conversation
    global groq_chat
    groq_chat = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=model)
    conversation = ConversationChain(llm=groq_chat, memory=memory)

    # Display chatbot photo
    st.image("aiblog.png", caption="", use_column_width=True)

    st.sidebar.header("Upload CSV")
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

    if uploaded_file:
        data = load_data(uploaded_file)
        st.write(data)

        patient_name = st.text_input("Enter patient name")
        if st.button("Generate Bill"):
            patient_details = data[data['Name'].str.lower() == patient_name.lower()]
            if not patient_details.empty:
                patient_details = patient_details.iloc[0].to_dict()

                st.subheader("Patient Details")
                for key, value in patient_details.items():
                    st.write(f"**{key}:** {value}")

                report = generate_billing_report(patient_details)
                
                st.subheader("Detailed Billing Report")
                st.text(report)
                
                pdf = create_pdf(patient_details, report)
                pdf_output = f"{patient_details['Name'].replace(' ', '_')}_billing_report.pdf"
                pdf.output(pdf_output)

                with open(pdf_output, "rb") as pdf_file:
                    st.download_button(
                        label="Download Bill PDF",
                        data=pdf_file,
                        file_name=pdf_output,
                        mime="application/pdf"
                    )

                # Store patient details in session state
                st.session_state.patient_details = patient_details

                # Generate and display health suggestions automatically
                suggestions = generate_health_suggestions(patient_details)
                st.subheader("Health Suggestions")
                st.text(suggestions)

                # Display graphs
                fig, ax = plt.subplots()
                sns.histplot(data['Billing Amount'], bins=20, kde=True, ax=ax)
                ax.set_title('Distribution of Billing Amounts')
                st.pyplot(fig)

                fig, ax = plt.subplots()
                sns.countplot(x='Gender', data=data, ax=ax)
                ax.set_title('Patient Gender Distribution')
                st.pyplot(fig)

                fig, ax = plt.subplots()
                sns.countplot(y='Medical Condition', data=data, order=data['Medical Condition'].value_counts().index, ax=ax)
                ax.set_title('Frequency of Medical Conditions')
                st.pyplot(fig)
                
            else:
                st.error("Patient not found")

    if st.button("Speak"):
        user_input = voice_to_text()
        if user_input:
            if user_input.lower().startswith("generate blog post about"):
                topic = user_input[24:].strip()
                heading, intro, content, summary = generate_blog_post(topic)
                st.write(f"## Heading\n\n{heading}")
                st.write(f"## Introduction\n\n{intro}")
                st.write(f"## Content\n\n{content}")
                st.write(f"### Summary\n\n{summary}")
            else:
                response = conversation(user_input)
                message = {'human': user_input, 'AI': response['response']}
                st.session_state.chat_history.append(message)
                st.write(f"<h2 style='color: #FF69B4;'>Chatbot:</h2><p style='color: #FFFFFF;'>{response['response']}</p>", unsafe_allow_html=True)

                # Text-to-speech
                tts = gTTS(response['response'], lang="en")
                tts.save("response.mp3")

                # Automatic playback of the audio response
                st.audio("response.mp3", format="audio/mp3", start_time=0)

                # Automatically fetch patient details and generate suggestions
                if 'patient details' in response['response'].lower() or 'generate suggestions' in response['response'].lower():
                    # Fetch patient details from the last generated section
                    patient_details = st.session_state.patient_details
                    if patient_details:
                        suggestions = generate_health_suggestions(patient_details)
                        st.subheader("Health Suggestions")
                        st.text(suggestions)
                    else:
                        st.error("Patient details not found. Please generate a bill first.")
                else:
                    # Clear patient details if not relevant to the current conversation
                    st.session_state.patient_details = None

    if st.button("BOT"):
        if 'patient_details' in st.session_state and st.session_state.patient_details:
            patient_details = st.session_state.patient_details
            topic = f"the patient {patient_details['Name']}"
            heading, intro, content, summary = generate_blog_post(topic)
            st.write(f"## Heading\n\n{heading}")
            st.write(f"## Introduction\n\n{intro}")
            st.write(f"## Content\n\n{content}")
            st.write(f"### Summary\n\n{summary}")

            suggestions = generate_health_suggestions(patient_details)
            st.subheader("Health Suggestions")
            st.text(suggestions)
        else:
            st.error("No patient details found. Please generate a bill first.")

    user_question = st.text_area("Ask a question:")
    if user_question:
        response = conversation(user_question)
        message = {'human': user_question, 'AI': response['response']}
        st.session_state.chat_history.append(message)
        st.write(f"<h2 style='color: #FF69B4;'>Chatbot:</h2><p style='color: #FFFFFF;'>{response['response']}</p>", unsafe_allow_html=True)

        # Text-to-speech
        tts = gTTS(response['response'], lang="en")
        tts.save("response.mp3")

        # Automatic playback of the audio response
        st.audio("response.mp3", format="audio/mp3", start_time=0)

        # Automatically fetch patient details and generate suggestions
        if 'patient details' in response['response'].lower() or 'generate suggestions' in response['response'].lower():
            # Fetch patient details from the last generated section
            patient_details = st.session_state.patient_details
            if patient_details:
                suggestions = generate_health_suggestions(patient_details)
                st.subheader("Health Suggestions")
                st.text(suggestions)
            else:
                st.error("Patient details not found. Please generate a bill first.")
        else:
            # Clear patient details if not relevant to the current conversation
            st.session_state.patient_details = None

if __name__ == '__main__':
    main()
