# Customer Churn Prediction

## Project Overview  
This project is a **Customer Churn Prediction System** that leverages **Deep Learning** to identify customers who are likely to leave a bank. The system uses a **trained Artificial Neural Network (ANN)** to make predictions based on customer demographic and financial data.

## Objective  
The primary goal of this project is to build a **predictive model** that helps financial institutions **reduce churn rates** by identifying high-risk customers. This allows banks to take proactive measures to retain customers.

## Key Steps in the Project  
1. **Data Collection and Preprocessing**  
   - The dataset (`Churn_Modelling.csv`) was cleaned and preprocessed.  
   - Features such as `Geography`, `Gender`, and `CreditScore` were transformed using **Label Encoding** and **One-Hot Encoding**.  
   - Features were standardized using **StandardScaler** to improve model performance.  

2. **Model Development**  
   - An **Artificial Neural Network (ANN)** was built using **TensorFlow & Keras**.  
   - The model consists of input, hidden, and output layers optimized for binary classification.  
   - Hyperparameter tuning was performed to enhance the accuracy of the model.

3. **Model Training & Evaluation**  
   - The model was trained on historical customer data.  
   - It was evaluated using metrics like **accuracy, precision, recall, and F1-score**.  
   - The final trained model (`model.h5`) was saved for deployment.  

4. **Deployment with Streamlit**  
   - A **Streamlit web application** was developed for user interaction.  
   - Users input customer data, and the model predicts **churn probability**.  
   - The app dynamically displays whether a customer is likely to churn or not.

## Conclusions  
- The model successfully predicts customer churn with high accuracy.  
- Businesses can use this tool to **identify and retain high-risk customers**.  
- The deployed web app provides an easy-to-use interface for real-time churn prediction.

## Technologies Used  
- **Python** (Pandas, NumPy, Scikit-Learn)  
- **TensorFlow & Keras** (Deep Learning)  
- **Streamlit** (Web Application Framework)  
- **Pickle** (Model and Encoder Storage)  
- **GitHub** (Version Control & Deployment)  

## How to Run  
1. Clone the repository:  
   ```sh
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
2. Install dependencies:
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
3. Run the Streamlit app:
   streamlit run app.py
4. Open the web app in your browser and input customer details to predict churn.

## Future Work
- Improve model accuracy with advanced deep learning architectures.
- Integrate Explainable AI (XAI) for better model interpretability.
- Extend the dataset with real-time customer transaction data.
- Deploy the app using Docker & Cloud Platforms for scalability.
