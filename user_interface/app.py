import streamlit as st
import joblib

# Function to load model
def load_model(model_path):
    return joblib.load(model_path)

# Function to make predictions
def predict_news(model, vectorizer, news_text):
    news_vector = vectorizer.transform([news_text])
    return model.predict(news_vector)

# Streamlit UI
st.set_page_config(page_title="Fake News Detection System", page_icon="📰")
st.title("Fake News Detection System")
st.write("This is a simple web app to detect fake news using different models.")

st.sidebar.title("About")
st.sidebar.info(
    """
    This application is designed to detect fake news using different machine learning models: 
    **Decision Tree**, **Logistic Regression**, and **Support Vector Machine**.
    
    🔹 Select a model from the dropdown menu.  
    🔹 Enter a news article in the text box.  
    🔹 Click the **Predict** button to check if the news is **Real or Fake**.  

    This tool leverages Natural Language Processing (NLP) techniques to analyze news content 
    and classify it based on patterns learned from training data.
    
    **📌 Note:** The accuracy of the prediction depends on the dataset and model used for training.
    """
)


# Model selection
model_choice = st.selectbox(
    "Select the model", 
    ["Decision Tree", "Logistic Regression", "Support Vector Machine"]
)

# Text input for user to enter news article
news_text = st.text_area("Enter the news article content:")

# Mapping models to their respective files
model_paths = {
    "Decision Tree": "fake_news_detection_system_dt.pkl",
    "Logistic Regression": "fake_news_detection_system_logistic.pkl",
    "Support Vector Machine": "fake_news_detection_system_svm.pkl",
}

vectorizer_path = "vectorizer.pkl"

if st.button("Predict"):
    if news_text.strip():  # Ensure input is not empty
        # Load selected model and vectorizer
        model = load_model(model_paths[model_choice])
        vectorizer = load_model(vectorizer_path)
        
        # Make prediction
        prediction = predict_news(model, vectorizer, news_text)
        
        # Display result
        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.success("✅ Real News Detected!")
            
        else:
            st.error("⚠️ Fake News Detected!")
    else:
        st.warning("Please enter some text to analyze.")



# Footer
st.markdown("---")
st.markdown(
    """
    **Note:** This is a simple fake news detection system. It may not be 100% accurate.
    """
)
# Contact Information
st.markdown("### 📬 Connect with Me")
st.markdown(
    """
    - 🛠️ [GitHub](https://github.com/RimeshCdry/)  
    - ✉️ Email: rimeshcdry45@gmail.com  
    """
)
