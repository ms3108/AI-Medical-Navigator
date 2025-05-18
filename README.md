# 🏥 AI Medical Navigator

**AI Medical Navigator** is a smart web-based platform designed to streamline medical consultations using artificial intelligence. Built with **Flask**, **MongoDB**, **HTML/CSS/JavaScript**, and integrated with **RoBERTa** for medical Q&A, this application connects users with homeopathic practitioners and allows real-time communication with healthcare providers.

## 🚀 Features

### 👤 User
- Register and log in securely.
- Interact with an AI medical chatbot (powered by RoBERTa).
- Get preliminary medicine suggestions for common ailments.
- Consult with certified homeopathic doctors in real time.
- View previous chats and consultation history.

### 🩺 Doctor
- Register and await admin approval.
- Log in to manage patient consultations.
- Real-time chat interface using Socket.IO.
- View list of assigned patients and previous chats.

### 🔐 Admin
- View all user and doctor accounts.
- Approve or reject doctor registrations.
- Monitor platform usage and communication history.

## 🛠️ Tech Stack

| Component        | Technology         |
|------------------|--------------------|
| Backend          | Flask (Python)     |
| Frontend         | HTML, CSS, JavaScript |
| Database         | MongoDB            |
| Real-Time Chat   | Socket.IO          |
| AI Chatbot       | RoBERTa (transformers) |
| Authentication   | Flask-Login        |
| Deployment       | (Render ) |

## 📂 Project Structure


├── static/
│ └── css/
│ └── style.css
├── templates/
│ ├── login.html
│ ├── dashboard.html
│ ├── chatbot.html
│ └── ...
├── app.py
├── chatbot.py
├── models.py
├── routes/
│ ├── user_routes.py
│ ├── doctor_routes.py
│ └── admin_routes.py
├── database/
│ └── init_db.py
└── README.md

## 🤖 AI Chatbot (RoBERTa)

- Utilizes pre-trained **RoBERTa** models fine-tuned for question-answering.
- Parses user symptoms and recommends basic homeopathic remedies.
- Offers a fallback to human consultation for complex queries.

## 📦 Installation

### Prerequisites

- Python 3.8+
- MongoDB
- pip

### Setup

```bash
git clone https://github.com/your-username/ai-medical-navigator.git
cd ai-medical-navigator
pip install -r requirements.txt
python app.py

