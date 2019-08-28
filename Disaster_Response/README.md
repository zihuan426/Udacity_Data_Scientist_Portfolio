# Disaster Response Project

### Introduction
When it comes to disaster, people experienced it would send out many messages in a minute. It's impossible for human to read and classify them into different categories on time. So, in order to manage to much more efficient disaster response, I built a pipeline using ETL and machine learning technique to classify messages and also designed a web app.

### Installation
To replicate the findings and execute the code in this repository you will need basically the following Python packages:
- Numpy
- pandas
- scikit-learn
- flask
- sqlalchemy

### File Description
1. Data folder contains datasets we need and the ETL pipeline, process_data.py
2. train_classifier.py in models folder is machine learning pipeline code
3. App folder contains the whole matericals needed to run the web application

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### Resources
Udacity all rights reserved.
