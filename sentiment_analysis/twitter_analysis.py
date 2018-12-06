import tweepy
from textblob import TextBlob
from os import environ
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt


def get_api_conection():
    auth = tweepy.OAuthHandler(environ.get('twitter_api_key'), environ.get('twitter_api_secret_key'))
    auth.set_access_token(environ.get('twitter_consumer_token'), environ.get('twitter_consumer_secret'))
    api = tweepy.API(auth)
    return api

def fetch_tweets(api, search,  lang='en'):
    # search
    tweets = api.search('inmigrants', lang='en')
    polarities = []
    for tweet in tweets:
        text = tweet.text
        analysis = TextBlob(tweet.text)
        polarity = analysis.sentiment
        polarities.append(polarity)

    return polarities


def load_map(file_name):
    return gpd.read_file(file_name)

if __name__ == "__main__":
    # stablish api connection
    api = get_api_conection()

    countries = load_map('countries.geo.json')[0:5]
    country_sentiment = {}
    for index, c in countries.iterrows():
        country_name = c['name']
        places = api.geo_search(query= country_name, granularity="country")
        place_id = places[0].id

        search_query = "{0} place:{1}".format('inmigrants', place_id)
        polarities = fetch_tweets(api, search_query, 'en')
        country_sentiment[country_name] = np.average(polarities)

    # create new dataframw column in term of country sentiment
    countries['sentiment'] = countries.apply(lambda row: country_sentiment[row['name']], axsi=1)

    # draw map
    countries.plot(column='sentiment', cmap='Blues', linewidth=0.8, ax=ax, edgecolor='0.8')
    plt.show()