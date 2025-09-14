# 📧 Email Spam Detection

This project classifies emails as **Spam** or **Not Spam** using machine learning techniques.  
It includes data preprocessing, feature extraction, visualization, model training, evaluation, and **deployment as a web application** for real-time predictions.

## 🚀 Project Workflow

1. **Data Preprocessing**
   - Removed missing values and duplicates  
   - Cleaned text (lowercasing, punctuation removal)  
   - Tokenization & stopword removal  

2. **Exploratory Data Analysis**
   - Visualized spam vs. ham distribution  
   - Plotted word clouds, histograms, and bar charts  

3. **Feature Engineering**
   - Converted text to numerical form using TF-IDF  

4. **Model Building**
   - Trained models like Logistic Regression, Naive Bayes, Random Forest  
   - Compared performance of each model  

5. **Evaluation**
   - Accuracy, Precision, Recall, F1-Score  
   - Confusion Matrix  

6. **Deployment**
   - Built a web app using **Flask/Streamlit**  
   - User can enter email text and get prediction instantly  
   - Deployed locally / on cloud for real-time testing  

## 📊 Results
- Achieved **>95% accuracy** using Logistic Regression and Naive Bayes  
- Live web app predicts spam or ham for custom inputs  

## 🛠️ Tech Stack
- **Language:** Python  
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn  
- **Deployment:** Flask / Streamlit  

