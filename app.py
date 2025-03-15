from functools import wraps
import os
from bson import ObjectId
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, session, flash, jsonify, url_for
from flask_socketio import SocketIO, emit, join_room, leave_room
from itsdangerous import URLSafeTimedSerializer
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime
import bcrypt
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import random, string
from flask_mail import Mail, Message

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()

# App configuration
app.config.update(
    SECRET_KEY=os.getenv('FLASK_SECRET_KEY'),
    MONGO_URI=os.getenv('MONGO_URI'),
    ADMIN_EMAIL='misna5984@gmail.com',
    ADMIN_PASSWORD='123',
    MAIL_SERVER='smtp.gmail.com',
    MAIL_PORT=587,
    MAIL_USE_TLS=True,
    MAIL_USERNAME='misna5984@gmail.com',
    MAIL_PASSWORD=os.getenv('MAIL_PASSWORD'),
    MAIL_DEFAULT_SENDER=('Medical Navigator', 'misna5984@gmail.com')
)

# Initialize Flask-Mail and Serializer
mail = Mail(app)
serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])

# MongoDB Atlas Connection
try:
    uri = app.config['MONGO_URI']
    client = MongoClient(uri, server_api=ServerApi('1'))
    client.admin.command('ping')
    print("Successfully connected to MongoDB!")

    db = client['hospital_db']
    users_collection = db['users']
    chat_collection = db['chats']
    doctor_status_collection = db['doctor_status']
    chatbot_chats_collection = db['chatbot_chats']


    # Create indexes for collections
    def initialize_collections():
        # Users collection indexes
        users_collection.create_index('email', unique=True)

        # Chat collection indexes
        chat_collection.create_index('room_code', unique=True)
        chat_collection.create_index('doctor_email')
        chat_collection.create_index('user_email')
        chat_collection.create_index('created_at')

        # Doctor status collection indexes
        doctor_status_collection.create_index('doctor_email', unique=True)

        # Chatbot chats collection indexes
        chatbot_chats_collection.create_index('user_email')
        chatbot_chats_collection.create_index('timestamp')


    # Initialize collections
    initialize_collections()

    # Create admin user if it doesn't exist
    admin_user = users_collection.find_one({'email': 'misna5984@gmail.com'})
    if not admin_user:
        hashed_password = bcrypt.hashpw('123'.encode('utf-8'), bcrypt.gensalt())
        admin = {
            'name': 'Admin',
            'email': 'misna5984@gmail.com',
            'password': hashed_password,
            'role': 'admin',
            'status': 'approved',
            'email_verified': True
        }
        users_collection.insert_one(admin)
        print("Admin user created successfully!")

except Exception as e:
    print(f"Error connecting to MongoDB: {e}")


# Add this function after your database connection
def reset_chat_indexes():
    try:
        # Drop existing indexes
        chat_collection.drop_indexes()

        # Create new indexes
        chat_collection.create_index('room_code', unique=True, sparse=True)
        chat_collection.create_index('doctor_email')
        chat_collection.create_index('user_email')
        chat_collection.create_index('created_at')

        print("Chat indexes reset successfully")
    except Exception as e:
        print(f"Error resetting chat indexes: {str(e)}")


# Call this function after database connection
reset_chat_indexes()

import random
import re
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline


class DocumentQA:
    def __init__(self):
        self.model_name = "deepset/deberta-v3-large-squad2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)

        # Explicitly move model to device before creating pipeline
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device set to use {self.device}")
        self.model.to(self.device)

        # Create pipeline with explicit device
        self.qa_pipeline = pipeline('question-answering',
                                    model=self.model,
                                    tokenizer=self.tokenizer,
                                    device=0 if torch.cuda.is_available() else -1)

        # Define the medical context directly in the code
        self.MEDICAL_CONTEXT = """
Let me help you understand some common medical conditions and their treatments:

Arnica Montana (Pain, Bruise)

Indication: Used for muscle pain, bruising, and injury.
Dosage:
Adults: 30C or 200C potency, 2-3 pellets every 2-3 hours for pain.
Children: 30C potency, 2 pellets 2-3 times a day.
Antimonium Tartaricum (Cough)

Indication: Useful for chest congestion and productive cough.
Dosage:
Adults: 30C, 3 pellets every 2 hours for intense coughing.
Children: 6C potency, 2 pellets every 3 hours.
Berberis Vulgaris (Urinary Infection)

Indication: Helps with urinary tract infections, especially if there's pain while urinating.
Dosage:
Adults: 30C, 2 pellets every 3-4 hours until improvement.
Children: 6C potency, 2 pellets every 3 hours.
Bryonia Alba (Fever, Cough)

Indication: Used for dry, painful coughs and associated fever.
Dosage:
Adults: 30C potency, 3-4 pellets every 3-4 hours for fever and cough.
Children: 6C potency, 2 pellets every 3-4 hours.
Conium Maculatum (Dizziness)

Indication: Helps with dizziness, especially from neck problems.
Dosage:
Adults: 30C potency, 2-3 pellets 3 times a day.
Children: 6C potency, 2 pellets 2-3 times daily.
Causticum (Pain)

Indication: Helps with chronic pains like joint or nerve pain.
Dosage:
Adults: 30C or 200C potency, 2 pellets 2-3 times a day.
Children: 6C potency, 2 pellets daily.
Cinchona Officinalis (Gas Trouble)

Indication: Useful for indigestion and gas-related discomfort.
Dosage:
Adults: 30C potency, 3 pellets 2-3 times a day.
Children: 6C potency, 2 pellets every 3-4 hours.
Cina (Round Worm)

Indication: Helps with symptoms of intestinal worms like irritability, itching, and abdominal pain.
Dosage:
Adults: 30C potency, 2-3 pellets 3 times a day.
Children: 6C potency, 2 pellets twice a day.
Colocynthis (Stomach Ache)

Indication: Used for colicky abdominal pain, cramps, or spasms.
Dosage:
Adults: 30C potency, 3 pellets every 2-3 hours for pain relief.
Children: 6C potency, 2 pellets every 3 hours.
Cantharis (Urinary Infection)

Indication: Best for burning sensations during urination.
Dosage:
Adults: 30C potency, 2-3 pellets every 2 hours until relief.
Children: 6C potency, 2 pellets every 4-5 hours.
Carbo Vegetabilis (Gas Trouble)
Indication: Used for bloating, indigestion, and flatulence.
Dosage:
Adults: 30C potency, 3 pellets 2-3 times a day.
Children: 6C potency, 2 pellets twice daily.
Gelsemium Sempervirens (Fever)
Indication: Ideal for fevers with weakness and fatigue.
Dosage:
Adults: 30C potency, 3 pellets every 3 hours during fever.
Children: 6C potency, 2 pellets every 4 hours.
Dioscorea Villosa (Stomach Ache)
Indication: Helps with cramping, especially around the abdomen.
Dosage:
Adults: 30C potency, 3 pellets 2-3 times a day.
Children: 6C potency, 2 pellets 2-3 times a day.
Hydrocotyle Asiatica (Itching)
Indication: Used for rashes and itching caused by skin irritation.
Dosage:
Adults: 30C potency, 3 pellets every 2 hours until relief.
Children: 6C potency, 2 pellets 2-3 times a day.
Ipecacuanta (Asthma)
Indication: Helps with asthma, especially when there is excessive mucus production.
Dosage:
Adults: 30C potency, 3 pellets every 2-3 hours.
Children: 6C potency, 2 pellets every 3 hours.
Ledum Palustre (Pain, Wound)
Indication: Used for pain, especially after injuries or puncture wounds.
Dosage:
Adults: 30C potency, 3 pellets 2-3 times a day.
Children: 6C potency, 2 pellets 2-3 times a day.
Kalmia Latifolia (Pain)
Indication: Helps with acute pain in the joints and muscles.
Dosage:
Adults: 30C potency, 3 pellets every 4 hours.
Children: 6C potency, 2 pellets 2-3 times a day.
Thuja Occidentalis (Warts)
Indication: Used for wart removal, especially those that are recurring.
Dosage:
Adults: 30C potency, 3 pellets every 2 days.
Children: 6C potency, 2 pellets every 2-3 days.
Spigelia Anthelmintica (Headache)
Indication: Helpful for sharp, throbbing headaches, often on the left side.
Dosage:
Adults: 30C potency, 3 pellets every 3-4 hours.
Children: 6C potency, 2 pellets every 3 hours.
Sanguinaria Canadensis (Headache)
Indication: Used for headaches with nausea, often in the right side of the head.
Dosage:
Adults: 30C potency, 3 pellets 2-3 times a day.
Children: 6C potency, 2 pellets 2 times a day.
Sepia (Itching)
Indication: Helps with itching, especially around the genitals or groin.
Dosage:
Adults: 30C potency, 3 pellets every 3 hours.
Children: 6C potency, 2 pellets every 4 hours.
Ruta Graveolens (Pain)
Indication: Relieves pain from overuse of muscles, especially in the knees and wrists.
Dosage:
Adults: 30C potency, 3 pellets 3-4 times a day.
Children: 6C potency, 2 pellets 2-3 times a day.
Rules Crispus (Cough, Cold)
Indication: Useful for symptoms of the common cold, including cough and congestion.
Dosage:
Adults: 30C potency, 3 pellets every 3 hours.
Children: 6C potency, 2 pellets 2-3 times a day.
Podophyllum Peltatum (Diarrhea)
Indication: Helps with diarrhea, particularly when there's a sense of urgency.
Dosage:
Adults: 30C potency, 3 pellets every 2 hours until symptoms improve.
Children: 6C potency, 2 pellets 2 times a day.
Rauwolfia Serpentina Q (Blood Pressure)
Indication: Supports normal blood pressure levels.
Dosage:
Adults: 10-15 drops in water 2-3 times a day.
Children: 5-8 drops in water twice a day.
Syzygium Jambolanum Q (Diabetes)
Indication: Helps manage blood sugar levels.
Dosage:
Adults: 10-15 drops in water twice a day.
Children: 5-8 drops twice a day.
"""

        # Define common medical conditions with their homeopathic remedies
        self.condition_remedies = {
            'pain': ['Arnica Montana', 'Causticum', 'Kalmia Latifolia', 'Ruta Graveolens', 'Ledum Palustre'],
            'bruise': ['Arnica Montana'],
            'cough': ['Antimonium Tartaricum', 'Bryonia Alba', 'Rules Crispus'],
            'cold': ['Rules Crispus'],
            'urinary infection': ['Berberis Vulgaris', 'Cantharis'],
            'fever': ['Bryonia Alba', 'Gelsemium Sempervirens'],
            'dizziness': ['Conium Maculatum'],
            'gas trouble': ['Cinchona Officinalis', 'Carbo Vegetabilis'],
            'round worm': ['Cina'],
            'stomach ache': ['Colocynthis', 'Dioscorea Villosa'],
            'itching': ['Hydrocotyle Asiatica', 'Sepia'],
            'asthma': ['Ipecacuanta'],
            'wound': ['Ledum Palustre'],
            'warts': ['Thuja Occidentalis'],
            'headache': ['Spigelia Anthelmintica', 'Sanguinaria Canadensis'],
            'diarrhea': ['Podophyllum Peltatum'],
            'blood pressure': ['Rauwolfia Serpentina Q'],
            'diabetes': ['Syzygium Jambolanum Q']
        }

        # Create a list of all conditions for regex matching
        self.common_conditions = list(self.condition_remedies.keys()) + [
            'flu', 'allergies', 'anxiety', 'depression',
            'nausea', 'fatigue', 'sore throat', 'congestion', 'runny nose',
            'chest pain', 'shortness of breath', 'back pain', 'joint pain',
            'rash', 'insomnia', 'vomiting'
        ]

        # Prepare a regex pattern to match these conditions as whole words
        pattern = r'\b(?:' + '|'.join(re.escape(cond) for cond in self.common_conditions) + r')\b'
        self.conditions_pattern = re.compile(pattern, re.IGNORECASE)

        # Define common symptom combinations that might indicate specific conditions
        self.symptom_combinations = {
            ('fever', 'cough', 'fatigue'): 'flu',
            ('headache', 'fever'): 'possible infection',
            ('nausea', 'vomiting', 'diarrhea'): 'stomach bug',
            ('congestion', 'runny nose', 'sore throat'): 'common cold',
            ('cough', 'shortness of breath', 'chest pain'): 'respiratory illness',
            ('headache', 'nausea', 'dizziness'): 'migraine',
        }

        # Add friendly message templates
        self.greeting_templates = [
            "Hi there! I'm here to help with your health concerns.",
            "Hello! I understand you're not feeling well. Let me try to help.",
            "I'd be happy to provide some information about your symptoms.",
            "Thank you for sharing how you're feeling. Let's see what might help.",
            "I appreciate you reaching out about your health. Here's what might help."
        ]

        self.empathy_templates = [
            "I'm sorry to hear you're experiencing {}. That must be uncomfortable.",
            "It sounds like {} is bothering you. That's never pleasant.",
            "Dealing with {} can be difficult. I hope I can offer some helpful information.",
            "I understand {} can be quite troublesome. Let me share some information that might help.",
            "{} can really impact your day. Here's what might provide some relief."
        ]

        self.conclusion_templates = [
            "Remember, this is just general information. A healthcare provider can give you personalized advice for your specific situation.",
            "While these remedies might help, it's always best to consult with a healthcare professional for proper diagnosis and treatment.",
            "I hope this information helps! For a complete evaluation, please consider speaking with a healthcare provider.",
            "These suggestions are meant to complement, not replace, professional medical advice. Your health is important!",
            "Every person's health needs are unique, so consider this information as a starting point for a conversation with your healthcare provider."
        ]

        self.transition_phrases = [
            "For your symptoms, ",
            "Based on what you've shared, ",
            "To address these issues, ",
            "To help with your condition, ",
            "Some options to consider include "
        ]

    def safe_qa_pipeline(self, question, context):
        """
        Safely run QA pipeline with error handling and logging
        """
        try:
            # Ensure inputs are strings
            question = str(question)
            context = str(context)

            # Run pipeline
            result = self.qa_pipeline({
                'question': question,
                'context': context
            })

            # Ensure result is a dictionary
            if isinstance(result, list):
                result = result[0] if result else {}

            return result
        except Exception as e:
            print(f"Error in QA pipeline: {e}")
            return {'answer': '', 'score': 0.0}

    def get_remedy_info(self, condition):
        """Get homeopathic remedy information for a specific condition"""
        condition = condition.lower()

        # Check if we have remedies for this condition
        for key in self.condition_remedies:
            if key in condition or condition in key:
                remedies = self.condition_remedies[key]
                if remedies:
                    # Select a random remedy to recommend
                    selected_remedy = random.choice(remedies)
                    return f"{selected_remedy} is often used for {key}."

        return None

    def format_friendly_response(self, responses, conditions=None, potential_diagnosis=None):
        """Make the response more friendly, empathetic and informative"""
        # Start with a greeting
        greeting = random.choice(self.greeting_templates)
        formatted_answer = f"{greeting}\n\n"

        # Add empathy if conditions are detected
        if conditions and len(conditions) > 0:
            condition_list = ", ".join(conditions)
            empathy = random.choice(self.empathy_templates).format(condition_list)
            formatted_answer += f"{empathy}\n\n"

        # Add info about potential diagnosis if available
        if potential_diagnosis and 'combined_response' in responses:
            formatted_answer += f"Your symptoms of {', '.join(conditions)} may suggest {potential_diagnosis}. "
            formatted_answer += f"{responses['combined_response']}\n\n"

        # Add transition phrase
        formatted_answer += random.choice(self.transition_phrases)

        # Add homeopathic remedy suggestions
        remedy_suggestions = []
        for condition in conditions:
            remedy_info = self.get_remedy_info(condition)
            if remedy_info:
                remedy_suggestions.append(remedy_info)

        if remedy_suggestions:
            formatted_answer += "\n\n• " + "\n• ".join(remedy_suggestions)

        # Add individual symptom responses if no remedies found
        if not remedy_suggestions:
            symptom_responses = []
            for condition in conditions:
                if condition in responses:
                    symptom_responses.append(responses[condition])

            if symptom_responses:
                formatted_answer += " " + " ".join(symptom_responses)

        # Add concluding statement
        formatted_answer += f"\n\n{random.choice(self.conclusion_templates)}"

        return formatted_answer

    def extract_conditions(self, text):
        """Extracts conditions from the user's question using RegEx to match whole words."""
        matches = self.conditions_pattern.findall(text)
        extracted = [match.lower().strip() for match in matches]
        return extracted

    def check_symptom_combinations(self, conditions):
        """Check if the conditions match any known symptom combinations"""
        conditions_set = set(conditions)
        for combo, diagnosis in self.symptom_combinations.items():
            combo_set = set(combo)
            # Check if all symptoms in the combination are present
            if combo_set.issubset(conditions_set):
                return combo, diagnosis
        return None, None

    def get_answer(self, question, max_length=512):
        try:
            # Use our built-in medical context
            context = self.MEDICAL_CONTEXT

            # Truncate context if too long
            if len(self.tokenizer.encode(context)) > max_length:
                context = self.tokenizer.decode(
                    self.tokenizer.encode(context)[:max_length],
                    skip_special_tokens=True
                )

            # Extract potential medical conditions from question
            conditions = self.extract_conditions(question)

            # If no conditions detected, look for common symptom words
            if not conditions:
                symptom_words = ["hurts", "pain", "ache", "sore", "uncomfortable", "sick", "not feeling well"]
                for word in symptom_words:
                    if word in question.lower():
                        # Ask a clarifying question
                        return {
                            'answer': f"I notice you mentioned you're not feeling well. Could you share more specific symptoms you're experiencing? For example, do you have a headache, fever, cough, or other specific concerns?",
                            'score': 0.5,
                            'is_clarification': True
                        }

            # Check if these conditions match any known combinations
            symptom_combo, potential_diagnosis = self.check_symptom_combinations(conditions)

            # Dictionary to store responses for each condition and the combined condition
            responses = {}
            max_score = 0.0

            # If we have a potential diagnosis based on symptom combination, query for that
            if potential_diagnosis:
                combined_result = self.safe_qa_pipeline(
                    f"What is the remedy for {potential_diagnosis}?",
                    context
                )

                if combined_result and combined_result.get('score', 0) >= 0.1:
                    responses['combined_response'] = combined_result.get('answer', '')
                    max_score = max(max_score, combined_result.get('score', 0))

            # Also generate individual responses for each symptom
            for condition in conditions:
                result = self.safe_qa_pipeline(
                    f"What is the remedy for {condition}?",
                    context
                )

                if result and result.get('score', 0) >= 0.1:
                    responses[condition] = f"For {condition}, {result.get('answer', '')}"
                    max_score = max(max_score, result.get('score', 0))

            if not responses and conditions:
                # We have conditions but no good responses from the model
                # Just use our built-in remedy information
                friendly_answer = self.format_friendly_response({}, conditions, potential_diagnosis)
                return {
                    'answer': friendly_answer,
                    'score': 0.5,
                    'conditions_detected': conditions,
                    'potential_diagnosis': potential_diagnosis if potential_diagnosis else None
                }

            elif not responses:
                # No conditions detected and no good responses
                return {
                    'answer': "I'd love to help, but I need a bit more information about what you're experiencing. Could you please share any specific symptoms or health concerns you have? For example, are you experiencing pain, fever, cough, or something else?",
                    'score': 0.0,
                    'is_clarification': True
                }

            # Combine responses into a single friendly answer
            friendly_answer = self.format_friendly_response(responses, conditions)

            return {
                'answer': friendly_answer,
                'score': max_score,
                'conditions_detected': conditions,
                'potential_diagnosis': potential_diagnosis if potential_diagnosis else None
            }

        except Exception as e:
            print(f"Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'error': f"I apologize, but I encountered an issue while processing your question. Could you try rephrasing it? I'm here to help with any health concerns you might have.",
                'answer': None,
                'score': None
            }


# Example usage

    # Test with some example questions
    test_questions = [
        "I have a headache and fever. What should I do?",
        "I'm suffering from cough, fever, and fatigue. Any recommendations?",
        "My stomach hurts badly. Can you help?",
        "I have an itchy rash on my arm.",
        "I've been feeling dizzy lately."
    ]

qa_system = DocumentQA()


# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'email' not in session:
            flash('Please login first.')
            return redirect('/login')
        return f(*args, **kwargs)
    return decorated_function

# Registration route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            name = request.form['name']
            email = request.form['email']
            password = request.form['password']
            role = request.form['role']

            # Check if user already exists
            if users_collection.find_one({'email': email}):
                flash('Email already exists!')
                return redirect('/register')

            # Hash the password
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

            # Generate email verification token
            token = serializer.dumps(email, salt='email-verification')

            # Create verification link
            verification_link = url_for(
                'verify_email',
                token=token,
                _external=True
            )

            # Store user data
            user = {
                'name': name,
                'email': email,
                'password': hashed_password,
                'role': role,
                'status': 'pending' if role == 'doctor' else 'approved',
                'email_verified': False,  # Will be updated after verification
                'verification_token': token,
                'created_at': datetime.now()
            }

            # Add doctor-specific fields
            if role == 'doctor':
                user.update({
                    'specialization': request.form.get('specialization', ''),
                    'qualifications': request.form.get('qualifications', ''),
                    'experience': request.form.get('experience', ''),
                    'gender': request.form.get('gender', ''),
                    'about': request.form.get('about', '')
                })

            # Send verification email
            msg = Message(
                'Verify Your Medical Navigator Account',
                recipients=[email]
            )
            msg.html = render_template(
                'email/verification_email.html',
                name=name,
                verification_link=verification_link
            )
            mail.send(msg)

            # Insert user into database
            users_collection.insert_one(user)

            # Flash success message and redirect to login
            if role == 'doctor':
                flash('Registration successful! Please verify your email and wait for admin approval.')
            else:
                flash('Registration successful! Please verify your email before logging in.')

            return redirect('/login')  # Redirect to login page after registration

        except Exception as e:
            flash(f'An error occurred: {str(e)}')
            return redirect('/register')

    return render_template('register.html')








# Email verification route
@app.route('/verify-email/<token>')
def verify_email(token):
    try:
        email = serializer.loads(token, salt='email-verification', max_age=3600)  # Token expires in 1 hour

        # Update user's email verification status
        result = users_collection.update_one(
            {'email': email},
            {'$set': {'email_verified': True}}
        )

        if result.modified_count > 0:
            flash('Email verified successfully! You can now login.')
        else:
            flash('Invalid verification link or email already verified.')

    except Exception as e:
        flash('The verification link is invalid or has expired.')

    return redirect('/login')

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            email = request.form['email']
            password = request.form['password']

            user = users_collection.find_one({'email': email})

            if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
                # Skip email verification for admin
                if not user.get('email_verified', False) and user['role'] != 'admin':
                    flash('Please verify your email before logging in. Check your inbox for the verification link.')
                    return redirect('/login')

                # Check doctor's status
                if user['role'] == 'doctor':
                    if user['status'] == 'pending':
                        flash('Your account is pending approval from admin.')
                        return redirect('/login')
                    elif user['status'] == 'rejected':
                        flash('Your registration has been rejected. Please contact administration.')
                        return redirect('/login')

                session['email'] = email
                session['role'] = user['role']
                session['name'] = user['name']

                if user['role'] == 'admin':
                    return redirect('/admin_dashboard')
                elif user['role'] == 'doctor':
                    return redirect('/doctor_dashboard')
                else:
                    return redirect('/user_dashboard')
            else:
                flash('Invalid email or password!')

        except Exception as e:
            flash(f'An error occurred: {str(e)}')

    return render_template('login.html', year=datetime.now().year)

# Resend verification email route
@app.route('/resend-verification')
def resend_verification():
    if 'email' not in session:
        return redirect('/login')

    try:
        user = users_collection.find_one({'email': session['email']})

        if user and not user.get('email_verified', False):
            # Generate new token
            token = serializer.dumps(user['email'], salt='email-verification')

            # Update token in database
            users_collection.update_one(
                {'email': user['email']},
                {'$set': {'verification_token': token}}
            )

            # Create verification link
            verification_link = url_for(
                'verify_email',
                token=token,
                _external=True
            )

            # Send new verification email
            msg = Message(
                'Verify Your Medical Navigator Account',
                recipients=[user['email']]
            )
            msg.html = render_template(
                'email/verification_email.html',
                name=user['name'],
                verification_link=verification_link
            )
            mail.send(msg)

            flash('Verification email resent. Please check your inbox.')
        else:
            flash('Email already verified or user not found.')

    except Exception as e:
        flash(f'Error resending verification email: {str(e)}')

    return redirect('/login')


# Dashboard routes
@app.route('/')
def index():
    if 'email' not in session:
        return redirect('/login')

    if session['role'] == 'admin':
        return redirect('/admin_dashboard')
    elif session['role'] == 'doctor':
        return redirect('/doctor_dashboard')
    else:
        return redirect('/user_dashboard')


@app.route('/user_dashboard')
@login_required
def user_dashboard():
    if session['role'] != 'user':
        return redirect('/login')
    user_data = {
        'name': session.get('name'),
        'email': session.get('email')
    }
    return render_template('user_dashboard.html', user=user_data)


@app.route('/doctor_dashboard')
@login_required
def doctor_dashboard():
    if session['role'] != 'doctor':
        return redirect('/')

    try:
        doctor = users_collection.find_one({'email': session['email']})
        status = doctor_status_collection.find_one({'doctor_email': session['email']})
        doctor['is_active'] = status['is_active'] if status else False

        active_chats = list(chat_collection.aggregate([
            {
                '$match': {
                    'doctor_email': session['email'],
                    'status': 'active'
                }
            },
            {
                '$lookup': {
                    'from': 'users',
                    'localField': 'user_email',
                    'foreignField': 'email',
                    'as': 'patient'
                }
            },
            {
                '$unwind': '$patient'
            },
            {
                '$project': {
                    'room_code': 1,
                    'patient_name': '$patient.name',
                    'created_at': 1
                }
            }
        ]))

        return render_template('doctor_dashboard.html',
                               doctor=doctor,
                               active_chats=active_chats)

    except Exception as e:
        flash('Error loading dashboard')
        return redirect('/login')


@app.route('/toggle_doctor_status', methods=['POST'])
@login_required
def toggle_doctor_status():
    if session['role'] != 'doctor':
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        data = request.get_json()
        is_active = data.get('is_active', False)

        doctor_status_collection.update_one(
            {'doctor_email': session['email']},
            {'$set': {
                'doctor_email': session['email'],
                'is_active': is_active,
                'last_updated': datetime.now()
            }},
            upsert=True
        )

        return jsonify({'success': True})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Chat functionality
@app.route('/active_doctors')
@login_required
def active_doctors():
    doctors = list(users_collection.find(
        {'role': 'doctor', 'status': 'approved'},
        {'password': 0}
    ))

    for doctor in doctors:
        status = doctor_status_collection.find_one({'doctor_email': doctor['email']})
        doctor['is_active'] = status.get('is_active', False) if status else False

    return render_template('active_doctors.html', doctors=doctors)


def generate_room_code():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))


@app.route('/start-chat/<doctor_email>')
@login_required
def start_chat(doctor_email):
    if 'email' not in session:
        return redirect('/login')

    try:
        doctor = users_collection.find_one({'email': doctor_email})
        if not doctor:
            flash('Doctor not found')
            return redirect('/active_doctors')

        room_code = generate_room_code()

        # Get user details
        user = users_collection.find_one({'email': session['email']})
        user_name = user.get('name', 'Unknown User') if user else 'Unknown User'

        chat_data = {
            'room_code': room_code,
            'doctor_email': doctor_email,
            'doctor_name': doctor.get('name', 'Unknown Doctor'),
            'user_email': session['email'],
            'user_name': user_name,
            'created_at': datetime.now(),
            'status': 'active',
            'messages': []
        }

        chat_collection.insert_one(chat_data)
        return redirect(f'/chat/{room_code}')

    except Exception as e:
        flash(f'Error starting chat: {str(e)}')
        return redirect('/active_doctors')


@app.route('/chat/<room_code>')
@login_required
def chat(room_code):
    try:
        # Get chat details
        chat = chat_collection.find_one({'room_code': room_code})
        if not chat:
            flash('Invalid room code')
            return redirect('/active_doctors')

        if session['email'] not in [chat['doctor_email'], chat['user_email']]:
            flash('Unauthorized access')
            return redirect('/active_doctors')

        # Get other participant's details
        other_email = chat['user_email'] if session['email'] == chat['doctor_email'] else chat['doctor_email']
        other_user = users_collection.find_one({'email': other_email})

        # Get participant names
        doctor_name = chat.get('doctor_name', 'Unknown Doctor')
        user_name = chat.get('user_name', 'Unknown User')

        # Determine which name to display based on current user's role
        if session['role'] == 'doctor':
            other_participant_name = user_name
        else:
            other_participant_name = f"Dr. {doctor_name}"

        return render_template('chat.html',
                               room_code=room_code,
                               name=session['name'],
                               other_participant_name=other_participant_name,
                               chat=chat)

    except Exception as e:
        print(f"Error in chat route: {str(e)}")
        flash('Error accessing chat')
        return redirect('/active_doctors')


@app.route('/end_chat/<room_code>', methods=['POST'])
@login_required
def end_chat(room_code):
    if session['role'] != 'doctor':
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        chat = chat_collection.find_one({
            'room_code': room_code,
            'doctor_email': session['email']
        })

        if not chat:
            return jsonify({'error': 'Chat not found'}), 404

        chat_collection.update_one(
            {'room_code': room_code},
            {
                '$set': {
                    'status': 'ended',
                    'ended_at': datetime.now(),
                    'ended_by': session['email']
                }
            }
        )

        return jsonify({'success': True})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Chatbot routes
@app.route('/chatbot')
@login_required
def chatbot():
    return render_template('chatbot.html')


@app.route('/ask', methods=['POST'])
@login_required
def ask_question():
    try:
        data = request.get_json()
        question = data.get('question', '')

        if not question:
            return jsonify({'error': 'No question provided'}), 400

        result = qa_system.get_answer(question)

        chat_data = {
            'user_email': session['email'],
            'user_name': session['name'],
            'question': question,
            'answer': result['answer'],
            'confidence_score': result['score'],
            'timestamp': datetime.now()
        }
        chatbot_chats_collection.insert_one(chat_data)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500





# Admin routes
@app.route('/admin_dashboard')
@login_required
def admin_dashboard():
    if session['role'] != 'admin':
        return redirect('/')

    try:
        all_users = list(users_collection.find(
            {},
            {'password': 0}
        ))

        pending_doctors = list(users_collection.find(
            {
                'role': 'doctor',
                'status': 'pending'
            },
            {'password': 0}
        ))

        stats = {
            'total_users': users_collection.count_documents({'role': 'user'}),
            'total_doctors': users_collection.count_documents({'role': 'doctor'}),
            'pending_approvals': len(pending_doctors),
            'active_chats': chat_collection.count_documents({'status': 'active'})
        }

        admin_data = {
            'name': session.get('name'),
            'email': session.get('email'),
            'users': all_users,
            'pending_doctors': pending_doctors,
            'stats': stats
        }

        return render_template('admin_dashboard.html', admin=admin_data)

    except Exception as e:
        flash(f'Error loading dashboard: {str(e)}')
        return redirect('/login')

@app.route('/admin/approve_doctor/<email>')
@login_required
def approve_doctor(email):
    if session['role'] != 'admin':
        return redirect('/login')

    try:
        result = users_collection.update_one(
            {
                'email': email,
                'role': 'doctor'
            },
            {
                '$set': {
                    'status': 'approved',
                    'approved_at': datetime.now(),
                    'approved_by': session['email']
                }
            }
        )

        if result.modified_count > 0:
            # Send approval email
            user = users_collection.find_one({'email': email})
            if user:
                msg = Message(
                    'Doctor Registration Approved',
                    recipients=[email]
                )
                msg.html = render_template(
                    'email/doctor_approval.html',
                    name=user['name']
                )
                mail.send(msg)
            flash('Doctor approved successfully!')
        else:
            flash('Doctor not found or already approved!')

    except Exception as e:
        flash(f'Error approving doctor: {str(e)}')

    return redirect('/admin_dashboard')

@app.route('/admin/reject_doctor/<email>')
@login_required
def reject_doctor(email):
    if session['role'] != 'admin':
        return redirect('/login')

    try:
        result = users_collection.update_one(
            {
                'email': email,
                'role': 'doctor'
            },
            {
                '$set': {
                    'status': 'rejected',
                    'rejected_at': datetime.now(),
                    'rejected_by': session['email']
                }
            }
        )

        if result.modified_count > 0:
            # Send rejection email
            user = users_collection.find_one({'email': email})
            if user:
                msg = Message(
                    'Doctor Registration Status',
                    recipients=[email]
                )
                msg.html = render_template(
                    'email/doctor_rejection.html',
                    name=user['name']
                )
                mail.send(msg)
            flash('Doctor rejected successfully!')
        else:
            flash('Doctor not found or already rejected!')

    except Exception as e:
        flash(f'Error rejecting doctor: {str(e)}')

    return redirect('/admin_dashboard')

@app.route('/admin/delete_user/<email>')
@login_required
def delete_user(email):
    if session['role'] != 'admin':
        return redirect('/login')

    try:
        if email == app.config['ADMIN_EMAIL']:
            flash('Cannot delete admin account!')
            return redirect('/admin_dashboard')

        chat_collection.delete_many({
            '$or': [
                {'user_email': email},
                {'doctor_email': email}
            ]
        })

        doctor_status_collection.delete_one({'doctor_email': email})
        result = users_collection.delete_one({'email': email})

        if result.deleted_count > 0:
            flash('User and associated data deleted successfully!')
        else:
            flash('User not found!')

    except Exception as e:
        flash(f'Error deleting user: {str(e)}')

    return redirect('/admin_dashboard')


# Flask route for viewing chatbot conversations
@app.route('/admin/chatbot_chats')
@login_required
def view_chatbot_chats():
    if session.get('role') != 'admin':
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect('/')

    try:
        # Get all chats
        chats = list(chatbot_chats_collection.find().sort('timestamp', -1))
        print(f"Retrieved {len(chats)} chatbot conversations")

        # Process each chat to handle potential None values
        for chat in chats:
            # Convert ObjectId to string
            chat['_id'] = str(chat['_id'])

            # Ensure all fields exist and have appropriate values
            if 'question' not in chat:
                chat['question'] = 'No question recorded'
            if 'answer' not in chat:
                chat['answer'] = 'No answer recorded'

            # This is the key fix - handle None confidence scores
            if 'confidence_score' not in chat or chat['confidence_score'] is None:
                chat['confidence_score'] = 0.0

            if 'user_name' not in chat:
                chat['user_name'] = 'Anonymous User'
            if 'timestamp' not in chat:
                chat['timestamp'] = datetime.now()

        return render_template('admin_chatbot_chats.html', chats=chats)
    except Exception as e:
        print(f"Error in view_chatbot_chats: {str(e)}")
        flash(f'Error loading chatbot conversations: {str(e)}', 'danger')
        return redirect('/admin_dashboard')

@app.route('/admin/delete_chat/<chat_id>', methods=['POST'])
@login_required
def delete_chatbot_chat(chat_id):
    if session.get('role') != 'admin':
        return jsonify({'success': False, 'error': 'Access denied'}), 403

    try:
        # Convert string id to ObjectId
        from bson.objectid import ObjectId
        result = chatbot_chats_collection.delete_one({'_id': ObjectId(chat_id)})

        if result.deleted_count > 0:
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Chat not found'}), 404

    except Exception as e:
        print(f"Error deleting chat {chat_id}: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Socket.IO setup and events
socketio = SocketIO(app)

@socketio.on('join')
def on_join(data):
    room = data['room']
    join_room(room)
    emit('message', {
        'user': 'System',
        'message': f"{session['name']} has joined the chat"
    }, room=room)
# Replace the duplicate Socket.IO handlers with this single version
@socketio.on('message')
def handle_message(data):
    room = data['room']
    message = data['message']
    timestamp = datetime.now()

    try:
        # Save message to database
        chat_collection.update_one(
            {'room_code': room},
            {
                '$push': {
                    'messages': {
                        'sender': session['name'],
                        'sender_email': session['email'],
                        'sender_role': session['role'],
                        'message': message,
                        'timestamp': timestamp
                    }
                }
            }
        )

        # Emit message to room
        emit('message', {
            'user': session['name'],
            'message': message,
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'sender_role': session['role']
        }, room=room)

    except Exception as e:
        # Emit error message only to sender
        emit('error', {
            'message': 'Failed to send message. Please try again.'
        })
        print(f"Error saving chat message: {str(e)}")
@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully!')
    return redirect('/login')


@app.route('/doctor/chat_history', methods=['GET', 'POST'])
@login_required
def doctor_chat_history():
    if session['role'] != 'doctor':
        return redirect('/')

    try:
        search_query = request.args.get('search', '').strip()
        print(f"Doctor chat history - Search query: {search_query}")  # Debug print

        # Base query for doctor's chats
        base_query = {'doctor_email': session['email']}
        print(f"Doctor email: {session['email']}")  # Debug print

        if search_query:
            # Search for patient in users collection
            patient_regex = {'$regex': search_query, '$options': 'i'}
            patients = list(users_collection.find({
                '$or': [
                    {'name': patient_regex},
                    {'email': patient_regex}
                ]
            }))
            patient_emails = [p['email'] for p in patients]
            print(f"Found matching patients: {patient_emails}")  # Debug print

            # Add patient filter to base query
            base_query['user_email'] = {'$in': patient_emails}

        # Get chats
        chats = list(chat_collection.find(base_query).sort('created_at', -1))
        print(f"Found {len(chats)} chats for doctor")  # Debug print

        # Group chats by patient
        patients = {}
        for chat in chats:
            try:
                patient_email = chat.get('user_email')
                if patient_email:
                    if patient_email not in patients:
                        patient = users_collection.find_one({'email': patient_email})
                        patients[patient_email] = {
                            'name': patient.get('name', 'Unknown Patient') if patient else 'Unknown Patient',
                            'email': patient_email,
                            'chats': []
                        }

                    formatted_chat = {
                        'room_code': chat.get('room_code'),
                        'created_at': chat.get('created_at'),
                        'status': chat.get('status', 'unknown'),
                        'messages': chat.get('messages', []),
                        'message_count': len(chat.get('messages', [])),
                    }
                    patients[patient_email]['chats'].append(formatted_chat)
                    print(f"Added chat for patient {patient_email}")  # Debug print
            except Exception as e:
                print(f"Error processing chat: {str(e)}")  # Debug print
                continue

        print(f"Total patients with chats: {len(patients)}")  # Debug print

        return render_template('doctor_chat_history.html',
                               patients=patients,
                               search_query=search_query)

    except Exception as e:
        print(f"Error in doctor chat history: {str(e)}")
        flash('Error loading chat history')
        return redirect('/doctor_dashboard')
# Add this helper function
def format_timestamp(timestamp):
    """Format timestamp for display"""
    now = datetime.now()
    diff = now - timestamp

    if diff.days == 0:
        if diff.seconds < 60:
            return 'Just now'
        elif diff.seconds < 3600:
            minutes = diff.seconds // 60
            return f'{minutes} minute{"s" if minutes != 1 else ""} ago'
        else:
            hours = diff.seconds // 3600
            return f'{hours} hour{"s" if hours != 1 else ""} ago'
    elif diff.days == 1:
        return 'Yesterday'
    elif diff.days < 7:
        return f'{diff.days} days ago'
    else:
        return timestamp.strftime('%Y-%m-%d %H:%M')

# Update the view_doctor_chat route to use the timestamp formatter
@app.route('/doctor/view_chat/<room_code>')
@login_required
def view_doctor_chat(room_code):
    if session['role'] != 'doctor':
        return redirect('/')

    try:
        chat = chat_collection.find_one({
            'room_code': room_code,
            'doctor_email': session['email']
        })

        if not chat:
            flash('Chat not found')
            return redirect('/doctor/chat_history')

        # Get patient details
        patient = users_collection.find_one({'email': chat.get('user_email')})
        if patient:
            chat['user_name'] = patient.get('name', 'Unknown Patient')

        # Format timestamps for messages
        for message in chat.get('messages', []):
            if 'timestamp' in message:
                message['formatted_time'] = message['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            else:
                message['formatted_time'] = 'Unknown time'

        return render_template('view_chat.html', chat=chat)

    except Exception as e:
        print(f"Error viewing chat: {str(e)}")
        flash(f'Error viewing chat: {str(e)}')
        return redirect('/doctor/chat_history')


@app.route('/admin/chat_history', methods=['GET', 'POST'])
@login_required
def admin_chat_history():
    if session['role'] != 'admin':
        return redirect('/')

    try:
        search_query = request.args.get('search', '').strip()
        print(f"Search query: {search_query}")  # Debug print

        if search_query:
            # Search for patient in users collection
            patient_regex = {'$regex': search_query, '$options': 'i'}
            patients = list(users_collection.find({
                '$or': [
                    {'name': patient_regex},
                    {'email': patient_regex}
                ]
            }))
            patient_emails = [p['email'] for p in patients]
            print(f"Found patients: {patient_emails}")  # Debug print

            # Get chats for matching patients
            chats = list(chat_collection.find({
                'user_email': {'$in': patient_emails}
            }).sort('created_at', -1))
        else:
            # Get all chats if no search
            chats = list(chat_collection.find().sort('created_at', -1))

        print(f"Total chats found: {len(chats)}")  # Debug print

        # Group chats by doctor and patient
        doctors = {}
        for chat in chats:
            try:
                doctor_email = chat.get('doctor_email')
                patient_email = chat.get('user_email')

                print(f"Processing chat - Doctor: {doctor_email}, Patient: {patient_email}")  # Debug print

                if doctor_email and patient_email:
                    if doctor_email not in doctors:
                        doctor = users_collection.find_one({'email': doctor_email})
                        doctors[doctor_email] = {
                            'name': doctor.get('name', 'Unknown Doctor') if doctor else 'Unknown Doctor',
                            'patients': {}
                        }

                    if patient_email not in doctors[doctor_email]['patients']:
                        patient = users_collection.find_one({'email': patient_email})
                        doctors[doctor_email]['patients'][patient_email] = {
                            'name': patient.get('name', 'Unknown Patient') if patient else 'Unknown Patient',
                            'chats': []
                        }

                    doctors[doctor_email]['patients'][patient_email]['chats'].append(chat)
            except Exception as e:
                print(f"Error processing chat: {str(e)}")  # Debug print
                continue

        print(f"Grouped by doctors: {list(doctors.keys())}")  # Debug print

        return render_template('admin_chat_history.html',
                               doctors=doctors,
                               search_query=search_query)

    except Exception as e:
        print(f"Error in admin chat history: {str(e)}")
        flash('Error loading chat history')
        return redirect('/admin_dashboard')

@app.route('/admin/view_chat/<room_code>')
@login_required
def admin_view_chat(room_code):
    if session['role'] != 'admin':
        return redirect('/')

    try:
        # Get chat details
        chat = chat_collection.find_one({'room_code': room_code})

        if not chat:
            flash('Chat not found')
            return redirect('/admin/chat_history')

        # Get doctor details
        doctor = users_collection.find_one({'email': chat.get('doctor_email')})
        doctor_name = doctor.get('name', 'Unknown Doctor') if doctor else 'Unknown Doctor'

        # Get patient details
        patient = users_collection.find_one({'email': chat.get('user_email')})
        patient_name = patient.get('name', 'Unknown Patient') if patient else 'Unknown Patient'

        # Update chat object with names
        chat['doctor_name'] = doctor_name
        chat['user_name'] = patient_name

        # Format timestamps for messages
        for message in chat.get('messages', []):
            if 'timestamp' in message:
                message['formatted_time'] = message['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            else:
                message['formatted_time'] = 'Unknown time'

        print(f"Viewing chat - Doctor: {doctor_name}, Patient: {patient_name}")  # Debug print

        return render_template('admin_view_chat.html', chat=chat)

    except Exception as e:
        print(f"Error viewing chat: {str(e)}")  # Debug print
        flash(f'Error viewing chat: {str(e)}')
        return redirect('/admin/chat_history')


@app.route('/user/chat_history')
@login_required
def user_chat_history():
    if session['role'] != 'user':
        return redirect('/')

    try:
        # Get all chats for this user
        chats = list(chat_collection.find({
            'user_email': session['email']
        }).sort('created_at', -1))

        # Group chats by doctor
        doctors = {}
        for chat in chats:
            doctor_email = chat.get('doctor_email')
            if doctor_email:
                if doctor_email not in doctors:
                    # Get doctor details from users collection
                    doctor = users_collection.find_one({'email': doctor_email})
                    doctors[doctor_email] = {
                        'name': doctor.get('name', 'Unknown Doctor') if doctor else 'Unknown Doctor',
                        'email': doctor_email,
                        'specialization': doctor.get('specialization', 'Not specified') if doctor else 'Not specified',
                        'chats': []
                    }

                # Format chat data
                formatted_chat = {
                    'room_code': chat.get('room_code', ''),
                    'created_at': chat.get('created_at', datetime.now()),
                    'status': chat.get('status', 'unknown'),
                    'messages': chat.get('messages', []),
                    'message_count': len(chat.get('messages', [])),
                    'doctor_name': doctors[doctor_email]['name']
                }
                doctors[doctor_email]['chats'].append(formatted_chat)

        return render_template('user_chat_history.html', doctors=doctors)

    except Exception as e:
        print(f"Error in user chat history: {str(e)}")
        flash('Error loading chat history')
        return redirect('/user_dashboard')


@app.route('/user/view_chat/<room_code>')
@login_required
def view_user_chat(room_code):
    if session['role'] != 'user':
        return redirect('/')

    try:
        chat = chat_collection.find_one({
            'room_code': room_code,
            'user_email': session['email']
        })

        if not chat:
            flash('Chat not found')
            return redirect('/user/chat_history')

        # Get doctor details
        doctor = users_collection.find_one({'email': chat.get('doctor_email')})
        if doctor:
            chat['doctor_name'] = doctor.get('name', 'Unknown Doctor')
            chat['doctor_specialization'] = doctor.get('specialization', 'Not specified')

        # Format timestamps for messages
        for message in chat.get('messages', []):
            if 'timestamp' in message:
                message['formatted_time'] = message['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            else:
                message['formatted_time'] = 'Unknown time'

        return render_template('view_user_chat.html', chat=chat)

    except Exception as e:
        print(f"Error viewing chat: {str(e)}")
        flash(f'Error viewing chat: {str(e)}')
        return redirect('/user/chat_history')




if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use Render's PORT or default 5000
    socketio.run(app, RW_DEBUG=True)
