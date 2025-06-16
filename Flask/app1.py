from lib2to3.pgen2.pgen import DFAState
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
import seaborn as sns
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import flask
from flask import Flask, redirect, render_template, request, url_for
import pickle

app = Flask(__name__)

# Define tfidf_matrix globally
tfidf_matrix = None

# Load the updated dataset
zomato_df = pd.read_csv(r'restaurant1.csv')

def get_recommendations(restaurant_name):
    # Check if the restaurant exists
    match = zomato_df[zomato_df['name'].str.lower() == restaurant_name.strip().lower()]
    
    if match.empty:
        return f"Error: The restaurant '{restaurant_name}' was not found in the database. Please check the spelling or try another."

    # Safe to access now
    input_restaurant = match.iloc[0]

    # Get the first cuisine keyword
    if pd.isnull(input_restaurant['cuisines']) or input_restaurant['cuisines'].strip() == "":
        return "Error: No cuisine information available for the selected restaurant."

    first_cuisine_keyword = input_restaurant['cuisines'].split()[0]

    # Filter restaurants by the cuisine
    similar_restaurants = zomato_df[
        zomato_df['cuisines'].apply(lambda x: isinstance(x, str) and x.split()[0] == first_cuisine_keyword)
    ]

    # Sort and clean
    top_restaurants = similar_restaurants.sort_values(by='Mean Rating', ascending=False)
    top_restaurants = top_restaurants.drop_duplicates(subset=['name', 'cuisines', 'cost'])

    return top_restaurants.head(10)



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['GET'])
def recommend():
    return render_template('recommend.html')

@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        restaurant_name = request.form.get('restaurant_name')
        if not restaurant_name:
            return "Error: No restaurant name provided. Please go back and enter a restaurant name."
        top_restaurants = get_recommendations(restaurant_name)
        if isinstance(top_restaurants, str):
            # If top_restaurants is a string, it's an error message
            return top_restaurants
        top_restaurants_list = top_restaurants.to_dict('records')
        return render_template('result.html', recommended_restaurants=top_restaurants_list)
    else:
        return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)




