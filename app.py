import streamlit as st
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from datetime import datetime
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import accuracy_score,confusion_matrix

# logger
def log(message):
    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

#session state
if "clean_saved" not in st.session_state:
    st.session_state.clean_saved=False

# folder setup
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
RAW_DIR=os.path.join(BASE_DIR,"data","raw")
CLEAN_DIR=os.path.join(BASE_DIR,"data","cleaned")
os.makedirs(RAW_DIR,exist_ok=True)
os.makedirs(CLEAN_DIR,exist_ok=True)

log("Application started")
log(f"RAW_DIR={RAW_DIR}")
log(f"CLEAN_DIR={CLEAN_DIR}")

#page configuration
st.set_page_config("End-to-End SVM Platform",layout="wide")
st.title("End-to-End SVM Platform")

#sidebar : Model settings
st.sidebar.header("SVM Settings")

kernal=st.sidebar.selectbox("kernal",["linear","rbf","ploy","sigmoid"])
C=st.sidebar.slider("C (Regularization)",0.01,10.0,1.0)
gamma=st.sidebar.selectbox("gamma",["scale","auto"])

log(f"SVM Settings ---> kernal={kernal},C={C},gamma={gamma}")

#step 1: Data Ingestion
st.header("step 1: Data Ingestion")
log("step 1 started: Data Ingestion")

option=st.radio("Choose data source",["Download Dataset","Upload CSV"])
df =None
raw_path=None
if option=="Download Dataset":
    if st.button("Download Iris Dataset"):
        log("Downloading Iris dataset")
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
        response = requests.get(url)
        
        raw_path = os.path.join(RAW_DIR,"iris.csv")
        with open(raw_path,"wb") as f:
            f.write(response.content)
        df=pd.read_csv(raw_path)
        st.success("File uploaded successfully")
        log(f"Uploaded dataset saved at {raw_path}")

#step 2:EDA
if df is not None:
    st.header("Step 2: Explortary Data Analysis")
    log("step 2 started EDA")
    st.dataframe(df.head())
    st.write("Shape:",df.shape)
    st.write("Missing Values:",df.isnull().sum())
    fig,ax=plt.subplots()
    sns.heatmap(df.corr(numeric_only=True),annot=True,cmap="coolwarm",ax=ax)
    st.pyplot(fig)
    log("EDA Completed")

# step 3:Data cleaning

if df is not None:
    st.header("step 3: data Cleaning")

    strategy=st.selectbox(
        "Missing Value Strategy",
        ["Mean","Median","Drop Rows"]
    )

    df_clean = df.copy()
    if strategy == "Drop Rows":
        df_clean=df_clean.dropna()
    else:
        for col in df_clean.select_dtypes(include=np.number):
            if strategy=="Mean":
                df_clean[col]=df_clean[col].fillna(df_clean[col].mean())
            else:
                df_clean[col]=df_clean[col].fillna(df_clean[col].meadian())
    st.session_state.df_clean=df_clean
    st.success("Data Cleaning Completed")
else:
    st.info("please complete the step 1(data Ingestion) First..")

#step 4:
if st.button("Save Cleaned Dataset"):
      
      if st.session_state.df_clean is None:
            st.error("No Cleaned data found..")
      else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            clean_filename = f"cleaned_dataset_{timestamp}.csv"
            clean_path = os.path.join(CLEAN_DIR,clean_filename)

            st.session_state.df_clean.to_csv(clean_path,index = False)

            st.success("Cleaned data saved")
            st.info(f"Saved at : {clean_path}")

            log(f"Cleaned data saved at {clean_path}")

#step 5:  load cleaned data
 
st.header("Step 5: Load Cleaned Dataset")
clean_files=os.listdir(CLEAN_DIR)

if not clean_files:
    st.warning("No cleaned datasets found")
    log("No cleaned datadsets found.")
else:
    selected=st.selectbox("select cleaned dataset",clean_files)
    df_model=pd.read_csv(os.path.join(CLEAN_DIR,selected))
    st.success("Loaded dataset:{selected}")
    log(f"Loaded cleaned dataset: {selected}")
    st.dataframe(df_model.head())

#step 6: Train model
st.header("Step 6: Train SVM")
log("Step 6 started SVM training")
target=st.selectbox("Select target column",df_model.columns)
y=df_model[target]
x=df_model.drop(columns=[target])