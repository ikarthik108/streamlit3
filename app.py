import numpy as np
import pickle
import pandas as pd
#from flasgger import Swagger
#pip install
import streamlit as st 
import streamlit.components.v1 as components
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly
#pip install
from gtts import gTTS


from PIL import Image

from explainer import interpret_model
from shapcode import shap_explainer
from precision_recall_curve import plot_prec_recall_vs_tresh
#app=Flask(__name__)
#Swagger(app)

sampled_df=pd.read_csv("sampled.csv")


# =============================================================================
# sampled_df['Closeness_5'] = (sampled_df.LIMIT_BAL - sampled_df.BILL_AMT5) 
# sampled_df['Closeness_4'] = (sampled_df.LIMIT_BAL - sampled_df.BILL_AMT4) 
# sampled_df['Closeness_3'] = (sampled_df.LIMIT_BAL - sampled_df.BILL_AMT3)
# 
# 
# sampled_df['Closeness_2'] = (sampled_df.LIMIT_BAL - sampled_df.BILL_AMT2)
# sampled_df['Closeness_1'] = (sampled_df.LIMIT_BAL - sampled_df.BILL_AMT1)
# =============================================================================

feature_set=['PAY_1', 'PAY_2', 'BILL_AMT2','BILL_AMT3', 'PAY_AMT1','PAY_AMT2', 'AGE', 
              'Closeness_2','Closeness_3']



# =============================================================================
# import app1
# import app2
# import streamlit as st
# PAGES = {
#     "App1": app1,
#     "App2": app2
#     }
# st.sidebar.title('Navigation')
# selection = st.sidebar.radio("Go to", list(PAGES.keys()))
# page = PAGES[selection]
# page.app()
# =============================================================================


def model_selection(name):
    
    if name=='Voting Classifier(DT,KNN,LR)':
        model="voting1"+'.pkl'
    
    elif name=='Voting Classifier(DT,KNN,GNB)':
        model="voting2"+'.pkl'
    
    elif name=='StackingClassifier(KNN,RF,GNB)':
        model="stacking1"+'.pkl'
    
    elif name=='StackingClassifier(KNN,RF,LR)':
        model="stacking2"+'.pkl'

    elif name=='StackingClassifier(KNN,GNB,LR)':
        model="stacking3"+'.pkl'
                
    elif name=='StackingClassifier(RF,GNB,LR)':
        model="stacking4"+'.pkl'
        

   
    elif name=='Random Forest':
        model="rf.pkl"
        
    elif name=='CatBoost':
        model="catboost.pkl"
        
    
    pickle_in = open(model,"rb")
    classifier=pickle.load(pickle_in)
    return classifier

#@app.route('/')
def welcome():
    return "Welcome All"

def gauge_chart(default_probability,pot_probability):
 
  value1=default_probability
  value1=value1*100
  
  
  
  fig1 = go.Figure(go.Indicator(
    mode = "gauge+number+delta",
    value =float(value1),
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "Default Confidence ", 'font': {'size': 24}},
    delta = {'reference': 50, 'increasing': {'color': "RebeccaPurple"}},
    gauge = {
        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'bar': {'color': "darkblue"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [0, 25], 'color': 'cyan'},
            {'range': [25, 40], 'color': 'royalblue'}],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': 50}}))

  fig1.update_layout(paper_bgcolor = "lavender", font = {'color': "darkblue", 'family': "Arial"})
  
  value2=pot_probability
  value2=value2*100
  
  fig2 = go.Figure(go.Indicator(
    mode = "gauge+number+delta",
    value =float(value2),
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': " Timely Payment Confidence ", 'font': {'size': 24}},
    delta = {'reference': 50, 'increasing': {'color': "RebeccaPurple"}},
    gauge = {
        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'bar': {'color': "darkblue"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [0, 25], 'color': 'cyan'},
            {'range': [25, 40], 'color': 'royalblue'}],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': 50}}))

  fig2.update_layout(paper_bgcolor = "lavender", font = {'color': "darkblue", 'family': "Arial"})
  
  
  return fig1,fig2

 
def instance_values(PAY_1,PAY_2,BILL_AMT2,BILL_AMT3,PAY_AMT1,PAY_AMT2,AGE,Closeness_2,Closeness_3):
    values=list()
    values.append(PAY_1)
    values.extend((PAY_2,BILL_AMT2,BILL_AMT3,PAY_AMT1,PAY_AMT2,AGE,Closeness_2,Closeness_3))
    return values

#@app.route('/predict',methods=["Get"])
def predict_default(PAY_1,PAY_2,BILL_AMT2,BILL_AMT3,PAY_AMT1,PAY_AMT2,AGE,Closeness_2,Closeness_3,option):
    
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
    classifier=model_selection(option)
    int_features = [x for x in [PAY_1,PAY_2,BILL_AMT2,BILL_AMT3,PAY_AMT1,PAY_AMT2,AGE,Closeness_2,Closeness_3]]
    final_features = [np.array(int_features)]
    prediction=classifier.predict(final_features)
    default_probability=classifier.predict_proba(final_features)[:,1]
    default_probability=np.round(default_probability,2)
    pot_probability=classifier.predict_proba(final_features)[:,0]
    pot_probability=np.round(pot_probability,2)
    fig1,fig2=gauge_chart(default_probability,pot_probability)
    print(prediction)
    return prediction, default_probability,fig1,fig2




 
def main():
# =============================================================================
#     st.title("Default Predictor")
# =============================================================================
    st.markdown("<h1 style='text-align: center; color: red;'>Default Predictor</h1>", unsafe_allow_html=True)
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Credit Card Default Payment Prediction </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    
  
# =============================================================================
#     PAY_1 = st.text_input("PAY_1","Repayment Status in September")
#     PAY_2 = st.text_input("PAY_2","Repayment Status in August")
#     PAY_3 = st.text_input("PAY_3","Repayment Status in July")
#     PAY_4 = st.text_input("PAY_4","Repayment Status in June")
#     PAY_5 = st.text_input("PAY_5","Repayment Status in May")
#     PAY_6 = st.text_input("PAY_6","Repayment Status in April")
# =============================================================================
        
    NAME = st.text_input("Name of the Client")
    
    AGE = st.number_input("AGE")
    AGE=int(AGE)

    PAY_1 = st.selectbox(
    'PAY_1(REPAYMENT STATUS IN SEPTEMBER)',
     ("0 Months Delay","1 Month Delay","2 Months Delay","3 Months Delay","4 Months Delay",
      "5 Months Delay","6 Months Delay","7 Months Delay","8 Months Delay"))
    x=PAY_1.split(" ")
    PAY_1=int(x[0])

    PAY_2 = st.selectbox(
    'PAY_2(REPAYMENT STATUS IN AUGUST)',
     ("0 Months Delay","1 Month Delay","2 Months Delay","3 Months Delay","4 Months Delay",
      "5 Months Delay","6 Months Delay","7 Months Delay","8 Months Delay"))
    
    y=PAY_2.split(" ")
    PAY_2=int(y[0])
    
    

# =============================================================================
#     BILL_AMT1 = st.number_input("BILL_AMT1")
#     BILL_AMT1=int(BILL_AMT1)
# =============================================================================
    
    BILL_AMT2 = st.number_input("BILL_AMT2(BILL AMOUNT IN AUGUST)")
    BILL_AMT2=int(BILL_AMT2)

    PAY_AMT1 = st.number_input("PAY_AMT1(AMOUNT OF PREVIOUS PAYMENT IN SEPTEMBER)")
    PAY_AMT1=int(PAY_AMT1)

    BILL_AMT3 = st.number_input("BILL_AMT3(BILL AMOUNT IN JULY)")
    BILL_AMT3=int(BILL_AMT3)
  
    PAY_AMT2 = st.number_input("PAY_AMT2(PREVIOUS PAYMENT IN AUGUST)")
    PAY_AMT2=int(PAY_AMT2)
    



 

    
  


# =============================================================================
#     Closeness_1 = st.number_input("Closeness_1")
#     Closeness_2 = st.number_input("Closeness_2")
#     #Closeness_3 = st.text_input("Closeness_3","Repayment Status in April")
# 
#     Closeness_4 = st.number_input("Closeness_4")
# =============================================================================
    #Closeness_6 = st.text_input("Closeness_6","Repayment Status in April")
    LIMIT_BAL=st.number_input("LIMIT_BAL(CREDIT LIMIT GIVEN TO THE PERSON)")
    LIMIT_BAL=int(LIMIT_BAL)
   
    


    
    
    #Closeness_1=(LIMIT_BAL) - (BILL_AMT1)
    Closeness_2=(LIMIT_BAL) - (BILL_AMT2)
    Closeness_3=(LIMIT_BAL) - (BILL_AMT3)
 
    #Closeness_4=(LIMIT_BAL) - (BILL_AMT4)


 # top_12_rf=['PAY_1', 'PAY_2', 'BILL_AMT1', 'PAY_AMT1', 'AGE', 
               #'Closeness_1', 'PAY_AMT2', 'Closeness_4', 'BILL_AMT2', 'Closeness_2', 'Closeness_3', 'Closeness_6'] 
               
               
    
    
    option = st.selectbox(
    'Select A Classifier',
     ('CatBoost','Voting Classifier(DT,KNN,LR)', 'Voting Classifier(DT,KNN,GNB)',
      'StackingClassifier(KNN,RF,LR)','StackingClassifier(RF,GNB,LR)'))

    st.write('You selected:', option)
    
    values=instance_values(PAY_1,PAY_2,BILL_AMT2,BILL_AMT3,PAY_AMT1,PAY_AMT2,AGE,Closeness_2,Closeness_3)
    
    #print(type(values))
    
    
    result,probability,text="","",""
    if st.button("Predict"):
        with st.spinner('Predicting The Results...'):
            
            result, probability,fig1,fig2 = predict_default(PAY_1,PAY_2,BILL_AMT2,BILL_AMT3,PAY_AMT1,PAY_AMT2,AGE,Closeness_2,Closeness_3,option)
            if result==1:
                name=NAME if len(NAME)>=1 else "Person"
                text="{} is more likely to default next month payment.".format(name)
                st.success('{}.  \nProbability of default is {}  '.format(text,probability))
                st.plotly_chart(fig1)
                st.plotly_chart(fig2)
                
                #Using Google Text To Speech API
                ta_tts = gTTS('{}. Probability of default is {} percent, which is {} percent more than the threshold value of 50% '.format(text,
                                                                                                                                            probability*100,(probability*100)-50))
                ta_tts.save("trans.mp3")
                audio_file = open("trans.mp3","rb")
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/ogg")
# =============================================================================
#                 audio_file = open('default.wav', 'rb')
#                 audio_bytes = audio_file.read()
#                 st.audio(audio_bytes, format='audio/wav')
# =============================================================================
            else:
                name=NAME if len(NAME)>=1 else "Person"
                text=" {} is likely to Make the Payment on Time".format(name)
                st.success('{}.  \n Probability of default is {}  '.format(text,probability))
                st.plotly_chart(fig1)
                st.plotly_chart(fig2)
                ta_tts = gTTS('{}. probability of default is {} percent which is {} percent less than the threshold value of 50%'.format(text,probability*100,50-(probability*100)))
                ta_tts.save("trans.mp3")
                audio_file = open("trans.mp3","rb")
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/ogg")
    if st.button("Explain LIME Results"):
        st.text("Note:'CLOSENESS' represents how far was the particular bill amount from the Limit Balance of the person")
        
        #Lime Explainer
        classifier=model_selection(option)
        limeexplainer=interpret_model(sampled_df,feature_set,classifier)
        values=instance_values(PAY_1,PAY_2,BILL_AMT2,BILL_AMT3,PAY_AMT1,PAY_AMT2,AGE,Closeness_2,Closeness_3)
        explainable_exp = limeexplainer.explain_instance(np.array(values), classifier.predict_proba, num_features=len(feature_set))
        explain = explainable_exp.as_pyplot_figure()
        
        st.pyplot(explain)
        # Display explainer HTML object
        components.html(explainable_exp.as_html(), height=800)
    
    if st.button("Explain SHAP Results"):
        with st.spinner('Generating Explanations...'):
            classifier=model_selection(option)
            feat_plot, summary= shap_explainer(sampled_df,feature_set,classifier)
            
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot(feat_plot)
            print("\n")
            st.pyplot(summary)
            #for plots in dependance_plots:
                #st.pyplot(plots)
         

    if st.button("Precision Recall Curve For the Classifier You selected"):
        
       
        classifier=model_selection(option)
        fig=plot_prec_recall_vs_tresh(sampled_df,feature_set,classifier,option)
        st.pyplot(fig)

    
        
        st.text("")

if __name__=='__main__':
    main()
    
    
# =============================================================================
# file_to_be_uploaded = st.file_uploader("Choose an audio...", type="wav")
# =============================================================================
