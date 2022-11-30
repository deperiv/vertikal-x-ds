# -*- coding: utf-8 -*-
import tweepy
import requests
import email
import imaplib
import re
import pandas   as pd
import numpy    as np
from fastapi      import FastAPI
from instagrapi   import Client
from instagrapi.mixins.challenge import ChallengeChoice
from datetime                    import datetime as dt

app = FastAPI(title="Data extraction",
              description="""Extract data from Instagram and Twitter""",
              version="0.1.0",)

@app.get("/")
def home():
    return {"API":"Data extraction for Instagram and Twitter"}


@app.get('/api/status')
def status():
    """
    GET method for API status verification.
    """
    
    message = {
        "status": 200,
        "message": [
            "This API is up and running!"
        ]
    }
    return message



# INSTAGRAM

CHALLENGE_EMAIL = "samanthaoakley202212@gmail.com"
CHALLENGE_PASSWORD = "tsgvcspxtnbojamx" 

def get_code_from_email(username):
    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(CHALLENGE_EMAIL, CHALLENGE_PASSWORD)
    mail.select("inbox")
    result, data = mail.search(None, "(UNSEEN)")
    assert result == "OK", "Error1 during get_code_from_email: %s" % result
    ids = data.pop().split()
    for num in reversed(ids):
        mail.store(num, "+FLAGS", "\\Seen")  # mark as read
        result, data = mail.fetch(num, "(RFC822)")
        assert result == "OK", "Error2 during get_code_from_email: %s" % result
        msg = email.message_from_string(data[0][1].decode())
        payloads = msg.get_payload()
        if not isinstance(payloads, list):
            payloads = [msg]
        code = None
        for payload in payloads:
            body = payload.get_payload(decode=True).decode()
            if "<div" not in body:
                continue
            match = re.search(">([^>]*?({u})[^<]*?)<".format(u=username), body)
            if not match:
                continue
            print("Match from email:", match.group(1))
            match = re.search(r">(\d{6})<", body)
            if not match:
                print('Skip this email, "code" not found')
                continue
            code = match.group(1)
            if code:
                return code
    return False

def challenge_code_handler(username, choice):
    if choice == ChallengeChoice.EMAIL:
        print("Asking for email code verification")
        return get_code_from_email(username)
    return False


IG_USERNAME = "samanthaoakley202212"
IG_PASSWORD = "1OE0t5@Po9*z"

cl = Client()
cl.challenge_code_handler = challenge_code_handler
cl.login(IG_USERNAME, IG_PASSWORD)

def extract_text(comment):
    comments_text_ls = []
    if len(comment[0]) != 0:
        comments_text_ls = [comment.text for comment in comment[0]]
    return comments_text_ls

# --- TWITTER ---

CONSUMER_KEY = 'P8uR3oU2eTF1hZw2SX1lma8Zw'
CONSUMER_SECRET = 'Kg85hTJdliuTdFroydQvoRsg7cpr5WZm2MdEQtCz8EjcjG04dp'
ACCESS_TOKEN = '386339280-5sy1Smvnkw91cBObIwu3ju6aMQJU4B8X8HZNsDSo'
ACCESS_TOKEN_SECRET = 'WuZNLBRfzZgT6DIDnZOPqaVlt8uMtrvOUJ9aSbUwkmocD'

auth = tweepy.OAuth1UserHandler(
    CONSUMER_KEY, 
    CONSUMER_SECRET, 
    ACCESS_TOKEN, 
    ACCESS_TOKEN_SECRET
)

api = tweepy.API(auth, wait_on_rate_limit=True)

twitter_username = ""

def get_replies(api, username, tweet_id, max_replies=10, max_attempts=20):
    replies = tweepy.Cursor(api.search_tweets, q='to:{}'.format(username),
                                    since_id=tweet_id, tweet_mode='extended').items()

    replies_ls = []

    counter_fetched_rep = 0
    counter_attempts = 0
    while counter_fetched_rep < max_replies and counter_attempts < max_attempts:
        try:
            reply = replies.next()
            if not hasattr(reply, 'in_reply_to_status_id_str'):
                continue
            if reply.in_reply_to_status_id == tweet_id:
                replies_ls.append(reply.full_text)
                counter_fetched_rep = counter_fetched_rep + 1 
            counter_attempts = counter_attempts + 1 

        except StopIteration:
            break

        except Exception as e:
            print("Failed while fetching replies {}".format(e))
            break
    return replies_ls

def get_sentiment(comments):
    url = "http://localhost:6001/api/get_sentiment"
    body = {"comments": comments}

    r = requests.post(url, json=body)
    response = r.json()
    return response

def get_comments_ls(data_dict, mode="ig"):
    all_comments = []
    if mode=="ig":
        for post_id in data_dict['posts_info'].keys():
            all_comments.extend(data_dict['posts_info'][post_id]["comments_text"])
    else:
        for tweet_replies in data_dict["tweets_replies"]:
            for comment in tweet_replies:
                all_comments.append(comment)
    return all_comments


@app.post('/api/twitter')
def get_twitter_data(query: dict):

    username = query["username"]
    max_tweets = 5
    max_replies = 10
    if "max_tweets" in list(query.keys()):
        max_tweets = query["max_tweets"]
    if "max_replies" in list(query.keys()):
        max_replies = query["max_replies"]
    
    n_followers = api.search_users(username)[0].followers_count
    extracted_tweets = []

    for status in tweepy.Cursor(api.search_tweets, 
                                f"from:{username}",
                                count=max_tweets).items(max_tweets):
        extracted_tweets.append(status)
    retweets = [tweet for tweet in extracted_tweets if "RT @" in tweet.text]
    tweets = set(extracted_tweets) - set(retweets)
    n_retweets = len(retweets)
    n_tweets = len(tweets)
    n_id_ls = []
    n_retweets_ls = []
    n_favorites_ls = []
    tweet_text_ls = []
    for tweet in tweets:
        n_id_ls.append(tweet.id)
        n_retweets_ls.append(tweet.retweet_count)
        n_favorites_ls.append(tweet.favorite_count)
        tweet_text_ls.append(tweet.text)
    replies_ls = [get_replies(api, username, tweet_id, max_replies=max_replies) for tweet_id in n_id_ls]

    user_data = {
        "username": username,
        "n_followers": n_followers,
        "n_retweets": n_retweets,
        "n_tweets": n_tweets,
        "n_retweets_to_user": sum(n_retweets_ls),
        "n_favorites_to_user": sum(n_favorites_ls),
        "n_replies_to_user": sum([len(replies_post) for replies_post in replies_ls]),
        "tweets_text": tweet_text_ls,
        "tweets_replies": replies_ls,
        "created_at": dt.now()
    }

    posts_comments_tw = get_comments_ls(user_data, mode="tw")
    sentiment_twitter = get_sentiment(posts_comments_tw)["sentiment"]
    user_data.pop('tweets_text', None)
    user_data.pop('tweets_replies', None)
    user_data.pop('posts_info', None)
    user_data["sentiment_twitter"] = sentiment_twitter

    return user_data

@app.post('/api/instagram')
def get_ig_data(query: dict): 
    
    username = query["username"]
    max_posts = 10
    max_comments = 10
    retrieve_all=False

    if "max_posts" in list(query.keys()):
        max_posts = query["max_posts"]
    if "max_comments" in list(query.keys()):
        max_comments = query["max_comments"]
    if "retrieve_all" in list(query.keys()):
        retrieve_all = query["retrieve_all"]

    user_id = cl.user_id_from_username(username)
    if retrieve_all:
        posts = cl.user_medias(user_id)
    else: 
        posts = cl.user_medias(user_id, max_posts)
    user_information = cl.user_info(user_id)

    n_followers = user_information.follower_count
    n_following = user_information.following_count
    n_posts = user_information.media_count

    # time.sleep(60)
    posts_info = {}
    
    for i, post in enumerate(posts):
        if i >= max_posts:
            posts_info[post.id] = {
                                "n_comments": post.comment_count, 
                                "n_likes": post.like_count, 
                                "caption": post.caption_text, 
                                "comments_text": []
                                }    
        else: 
            posts_info[post.id] = {
                                    "n_comments": post.comment_count, 
                                    "n_likes": post.like_count, 
                                    "caption": post.caption_text, 
                                    "comments_text": extract_text(cl.media_comments_chunk(post.id, max_amount=max_comments))
                                    }                              
    # time.sleep(60)
    user_data = {
        "username": username,
        "user_id_instagram": user_id,
        "n_followers": n_followers,
        "n_following": n_following,
        "n_posts_total": n_posts,
        "n_posts_retrieved": len(posts_info),
        "n_likes_total": sum([posts_info[key]["n_likes"] for key in posts_info.keys()]),
        "n_likes_retrieved": sum([posts_info[key]["n_likes"] for key in list(posts_info.keys())[:max_posts]]),
        "n_comments_total": sum([posts_info[key]["n_comments"] for key in posts_info.keys()]),
        "n_comments_retrieved": sum([len(posts_info[key]["comments_text"]) for key in posts_info.keys()]),
        "created_at": dt.now(),
        "posts_info": posts_info
    }

    posts_comments_ig = get_comments_ls(user_data, mode="ig")
    sentiment_instagram = get_sentiment(posts_comments_ig)["sentiment"]
    user_data.pop('posts_info', None)
    user_data["sentiment_instagram"] = sentiment_instagram

    return user_data