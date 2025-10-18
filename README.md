# 🏨 GuestSight  
**Multilingual, Aspect-Based Sentiment Analysis and Visual Sentiment Scores for Hotel Reviews**

**Author:** Sude Ebrar Çat  
**Advisor:** Tuğba Önal Süzek, PhD  
**Department of Computer Engineering, Muğla Sıtkı Koçman University**  
**Date:** 05/07/2025  

---

## 🌍 Project Overview

**GuestSight** is a multilingual, aspect-based sentiment analysis and recommendation system designed to analyze Turkish and English hotel reviews.  
The project integrates **Natural Language Processing (NLP)**, **Machine Learning (ML)**, and **Web Technologies** to deliver a comprehensive analytical and recommendation platform for both hotel managers and travelers.

GuestSight processes over **11,000+ reviews from 205 hotels in Muğla, Turkey**, providing:
- Actionable insights for **hotel owners** (data-driven service improvement)  
- Personalized recommendations for **travelers** (based on sentiment and preferences)  
- Visual sentiment scores and aspect-based analysis (cleanliness, service, and location)

---

## ✨ Key Features

- **Multilingual Sentiment Analysis:**  
  Accurately processes both Turkish and English hotel reviews using BERT-based transformer models.

- **Aspect-Based Classification:**  
  Categorizes feedback into key themes like *cleanliness*, *service quality*, and *location*.

- **Face Score Visualization:**  
  Converts complex review data into intuitive visual sentiment icons (😊 😐 ☹️).

- **Intelligent Recommendations:**  
  Hybrid recommendation engine combining **collaborative** and **content-based filtering**.

- **User-Friendly Interface:**  
  Clean, modern web interface built with **React.js** and **Material-UI** for easy interaction.

- **Fast & Scalable Backend:**  
  High-performance RESTful API developed using **FastAPI**, optimized for real-time results.

- **Seamless Accessibility:**  
  Users can explore hotel insights without registration — quick, simple, and interactive.

---

## 🧠 System Architecture

### Frontend
- **React.js 18.2.0** (Hooks & Functional Components)  
- **Material-UI (MUI) 5.15.6** for modern responsive design  
- **Axios** for API communication  
- **React Router DOM** for navigation and route protection  
- **React Query** for caching and intelligent data fetching  
- Deployed on **Vercel**

### Backend
- **FastAPI 0.109.2** — Asynchronous Python web framework  
- **Pydantic** for schema validation  
- **Uvicorn** as ASGI server  
- **Transformers 4.37.2 (Hugging Face)** for NLP models  
- **PyTorch** for model training and inference  
- **NLTK**, **spaCy**, **Gensim**, and **imbalanced-learn (SMOTE)** for preprocessing  
- Deployed on **Render**

### Data & Machine Learning
- **Data Source:** Google Places API (205 hotels, 11,441 reviews)  
- **Translation:** DeepL API for cross-language consistency  
- **Models:** BERTurk (Turkish) & BERT-base-uncased (English)  
- **Algorithms:** LDA topic modeling, PCA dimensionality reduction, and hybrid filtering  
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score

---

## 📊 System Workflow

1. **Data Collection** → Gather reviews via Google Places API  
2. **Preprocessing** → Clean, detect language, translate, and normalize data  
3. **Aspect Classification** → Identify key review categories  
4. **Sentiment Analysis** → Run multilingual BERT model for polarity detection  
5. **Feature Engineering** → Dimensionality reduction and standardization  
6. **Recommendation Engine** → Generate personalized hotel suggestions  
7. **Visualization** → Display results on a responsive web dashboard  

---

## 💻 Installation & Setup

### Prerequisites
- Python 3.8+  
- Node.js 18+  
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/SudeCat/GuestSight.git
cd GuestSight
```
### 2. Backend Setup
```bash
git clone https://github.com/SudeCat/GuestSight.git
cd GuestSight
```
### 3. Frontend Setup
```bash
cd ../NLP_NEW
npm install
npm start
```
### 5. Environment Variables
Create a .env file for:
```bash
DEEPL_API_KEY=your_deepl_key
GOOGLE_PLACES_API_KEY=your_google_places_key
```
## 🚀 Deployment

### Frontend
-**Frontend: Vercel** (Continuous Deployment via GitHub Actions)
-**Backend**: Render / Heroku
-**Storage**: Local + CSV-based review data-
-**CI/CD**: GitHub Actions automation

## 🎥 Demo & Live App

🔗 **Live Web App:** [GuestSight]([https://hotel-recommendation-kappa.vercel.app](https://hotel-recommendation-nine.vercel.app/))

🎬 **Demo Video:** [Watch on Google Drive](https://drive.google.com/file/d/1qEiNqLmG9HZJCtKa6pSqU8vXRm9e62UO/view?usp=sharing)

## End of the README

Feel free to explore each step in detail and get into the project workflow. If you have questions or suggestions, please open an issue or submit a pull request.
- **Email:** [catsudeebrar@gmail.com](mail)
