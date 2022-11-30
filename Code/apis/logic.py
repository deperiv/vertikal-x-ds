import requests
import mariadb
import sys
import pandas       as pd
import numpy        as np
from datetime           import datetime     as dt
from scipy              import stats
from pandas.api.types   import is_numeric_dtype

# --- Connect to database ---
try:
    conn = mariadb.connect(
        user="admin_vtx",
        password="Prueba12#",
        host="develop-free-vtx.csgmphwpjwca.us-east-1.rds.amazonaws.com",
        port=3306,
        database="catalogs"

    )
except mariadb.Error as e:
    print(f"Error connecting to MariaDB Platform: {e}")
    sys.exit(1)

# Get Cursor
cur = conn.cursor()

# --- Retrieve users and social media usernames ---
cur.execute("SELECT * FROM users;")

ids = []
names = []
instagram_usernames = []
twitter_usernames = []

for id, name, instagram_username, twitter_username, last_update in cur: 
    ids.append(id)
    names.append(name)
    instagram_usernames.append(instagram_username)
    twitter_usernames.append(twitter_username)
    print(f"name: {name}, instagram_username: {instagram_username}, twitter_username: {twitter_username}, last_update: {last_update}")

# --- Extract the data for all the users ---

url_ig = "http://localhost:5000/api/instagram"
url_tw = "http://localhost:5000/api/twitter"

users_data = []

status = 1
try:
    for ix, id in enumerate(ids):
        user_data_dict = {}
        user_data_dict["id"] = id
        user_data_dict["name"] = names[ix]
        user_data_dict["instagram_username"] = instagram_usernames[ix]
        user_data_dict["twitter_username"] = twitter_usernames[ix]
        print(f"Working on {names[ix]}")

        if user_data_dict["instagram_username"] != None:
            
            print("Getting IG data")
            body = {"username": user_data_dict["instagram_username"]}
            r = requests.post(url_ig, json=body)
            response = r.json()
            user_data_dict["instagram_data"] = response

        if user_data_dict["twitter_username"] != None:
            print("Getting TW data")
            body = {"username": user_data_dict["twitter_username"]}
            r = requests.post(url_tw, json=body)
            response = r.json()
            user_data_dict["twitter_data"] = response
        
        users_data.append(user_data_dict)
        print(f"Done with {names[ix]}")
        print("----------------------------------")
except:
    status = 0
    print("---Connection error: Try again in a couple of minutes--")

if status:

    # --- Process the data, compute engagement, percentiles and spi ---
    def compute_spi_ig(x):
        spi_ig_weights_dict = {
            "n_followers": 0.1,
            "n_posts_total": 0.1, 
            "n_likes_retrieved": 0.2, 
            "n_comments_retrieved": 0.1, 
            "sentiment_instagram": 0.3, 
            "relative_engagement": 0.2
        }
        spi = 0
        for k in spi_ig_weights_dict.keys():
            spi += spi_ig_weights_dict[k]*x[k]
        return spi

    def process_ig_data(data):
        users_data_instagram = []
        for user_data in data:
            if ("instagram_data" in user_data.keys()):
                dict_data = user_data["instagram_data"].copy()
                dict_data["user_id"] = user_data["id"]
                users_data_instagram.append(dict_data)
            else:
                empty_dict = {'user_id': users_data["id"],
                            'username': '',
                            "user_id_instagram": '',
                            'n_followers': 0,
                            'n_following': 0,
                            'n_posts_total': 0,
                            'n_posts_retrieved': 0,
                            'n_likes_total': 0,
                            'n_likes_retrieved': 0,
                            'n_comments_total': 0,
                            'n_comments_retrieved': 0,
                            'created_at': '',
                            'sentiment_instagram': 0}
                users_data_instagram.append(empty_dict)
            
        df_ig = pd.DataFrame(users_data_instagram)
        df_ig["relative_engagement"] = np.clip((df_ig["n_likes_retrieved"] + df_ig["n_comments_retrieved"])/(df_ig["n_followers"])*100, 0, 100)
        selected_cols = ["n_followers", "n_posts_total", "n_likes_retrieved", "n_comments_retrieved", "sentiment_instagram", "relative_engagement"]
        df_ig_perc = df_ig.copy()
        for col in selected_cols:
            df_ig_perc[col] = df_ig_perc[col].apply(lambda x: stats.percentileofscore(df_ig_perc[col], x))
        df_ig_perc["spi"] = df_ig_perc[selected_cols].apply(compute_spi_ig, axis=1)
        spi_ig_columns = ["user_id", "username", "n_followers", "n_posts_total", "n_likes_retrieved", "n_comments_retrieved", "sentiment_instagram", "relative_engagement"]
        return pd.concat([df_ig[spi_ig_columns], df_ig_perc["spi"], df_ig["created_at"]], axis=1)

    def compute_spi_tw(x):
        spi_ig_weights_dict = {
            "n_followers": 0.1,
            "n_retweets": 0.1, 
            "n_tweets": 0.1, 
            "n_retweets_to_user": 0.1, 
            "n_favorites_to_user": 0.1, 
            "n_replies_to_user": 0.1,
            "sentiment_twitter": 0.3,
            "relative_engagement": 0.1
        }
        spi = 0
        for k in spi_ig_weights_dict.keys():
            spi += spi_ig_weights_dict[k]*x[k]
        return spi

    def process_tw_data(data):
        users_data_twitter = []
        for user_data in data:
            if ("twitter_data" in user_data.keys()):
                dict_data = user_data["twitter_data"].copy()
                dict_data["user_id"] = user_data["id"]
                users_data_twitter.append(dict_data)
            else:
                empty_dict = {'user_id': user_data["id"],
                            'username': '',
                            "n_followers": 0,
                            'n_retweets': 0,
                            'n_tweets': 0,
                            'n_retweets_to_user': 0,
                            'n_favorites_to_user': 0,
                            'n_replies_to_user': 0,
                            'created_at': dt.now(),
                            'sentiment_twitter': 0}
                users_data_twitter.append(empty_dict)
        
        df_tw = pd.DataFrame(users_data_twitter)
        df_tw["relative_engagement"] = (df_tw.apply(lambda x: (x["n_favorites_to_user"] + x["n_retweets_to_user"] + x["n_replies_to_user"])/(x["n_followers"])*100 if x["n_tweets"] > 0 else 0, axis=1))
        selected_cols = ["n_followers", "n_retweets", "n_tweets", "n_retweets_to_user", "n_favorites_to_user", "n_replies_to_user", "sentiment_twitter", "relative_engagement"]
        df_tw_perc = df_tw.copy()
        for col in selected_cols:
            df_tw_perc[col] = df_tw_perc[col].apply(lambda x: stats.percentileofscore(df_tw_perc[col], x))
        df_tw_perc["spi"] = df_tw_perc[selected_cols].apply(compute_spi_tw, axis=1)
        spi_tw_columns = ["user_id","username","n_followers","n_retweets","n_tweets","n_retweets_to_user","n_favorites_to_user","n_replies_to_user","sentiment_twitter","relative_engagement"]
        return pd.concat([df_tw[spi_tw_columns], df_tw_perc["spi"], df_tw["created_at"]], axis=1)

    processed_ig_data = process_ig_data(users_data)
    processed_tw_data = process_tw_data(users_data)

    # --- Compute spi bonus ---

    def input_ig_data(cur, record):
        record_dict = record.to_dict('records')[0]
        cur.execute("INSERT INTO instagram_stats(user_id, instagram_username, n_followers, n_posts_total, n_likes_retrieved, n_comments_retrieved, sentiment, engagement, spi, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, now());",
        (record_dict["user_id"], record_dict["username"], record_dict["n_followers"], record_dict["n_posts_total"], record_dict["n_likes_retrieved"], record_dict["n_comments_retrieved"], record_dict["sentiment_instagram"], record_dict["relative_engagement"], record_dict["spi"]))

    def retrieve_latest_record_ig(user_id):
        cur.execute(f"SELECT * FROM instagram_stats WHERE user_id = {user_id} ORDER BY created_at DESC LIMIT 1;") 
        for record_id, user_id, instagram_username, n_followers, n_posts_total, n_likes_retrieved, n_comments_retrieved, sentiment, engagement, spi, created_at in cur:
            record_dict = {
                "user_id": user_id,
                "username": instagram_username,
                "n_followers": n_followers,
                "n_posts_total": n_posts_total,
                "n_likes_retrieved": n_likes_retrieved,
                "n_comments_retrieved": n_comments_retrieved,
                "sentiment_instagram": sentiment,
                "relative_engagement": engagement,
                "spi": spi,
                "created_at": created_at
            }
        return pd.DataFrame(record_dict, index=[0])
        

    def compute_definitive_ig_spi(user_id, new_record, bonus_weight=0.15):
        cur.execute(f"SELECT COUNT(*) FROM instagram_stats WHERE user_id = {user_id};") 
        for result in cur: 
            count_records = result[0]
        if count_records == 0:
            return new_record
        else: 
            
            dict_change = {}
            num_cols = [col for col in new_record.columns if is_numeric_dtype(new_record[col])]
            dict_change["username"] = new_record["username"].unique()[0]

            latest_record_db = retrieve_latest_record_ig(user_id)
            
            for col in num_cols:
                dict_change[col] = new_record.loc[0][col] - latest_record_db.loc[0][col]
                
            df_change = pd.DataFrame(dict_change, index=[0])
            
            dict_relative_change = {}
            dict_change = df_change.to_dict('records')[0]
            dict_base = latest_record_db.loc[0].to_dict()

            for key in dict_change.keys():
                if key == "username":
                    dict_relative_change[key] = dict_change[key]
                elif key == "created_at" or key == "spi":
                    pass
                else:
                    if dict_base[key] == 0 and dict_change[key] != 0:
                        dict_relative_change[key] = 1
                    elif dict_base[key] == 0 and dict_change[key] == 0:
                        dict_relative_change[key] = 0
                    else:
                        dict_relative_change[key] = dict_change[key] / dict_base[key]
                        
            dict_ig_spi_bonus_weight = {
                'n_followers': 0.1,
                'n_posts_total': 0.1,
                'n_likes_retrieved': 0.1,
                'n_comments_retrieved': 0.1,
                'sentiment_instagram': 0.4,
                'relative_engagement': 0.2
            }

            bonus_spi = 0
            for k in dict_relative_change.keys():
                if k != "username" and k != "user_id":
                    bonus_spi += dict_ig_spi_bonus_weight[k]*dict_relative_change[k]

            new_record_upd = new_record.copy()
            new_record_upd.loc[0, "spi"] = new_record_upd.loc[0, "spi"] + bonus_weight*bonus_spi*100

            return new_record_upd[:1]

    def input_tw_data(cur, record):
        record_dict = record.to_dict('records')[0]
        cur.execute("INSERT INTO twitter_stats(user_id, twitter_username, n_followers, n_retweets , n_tweets , n_retweets_to_user , n_favorites_to_user, n_replies_to_user, sentiment, engagement, spi, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, now());",
        (record_dict["user_id"], record_dict["username"], record_dict["n_followers"], record_dict["n_retweets"], record_dict["n_tweets"], record_dict["n_retweets_to_user"], record_dict["n_favorites_to_user"], record_dict["n_replies_to_user"], record_dict["sentiment_twitter"], record_dict["relative_engagement"], record_dict["spi"]))

    def retrieve_latest_record_tw(user_id):
        cur.execute(f"SELECT * FROM twitter_stats WHERE user_id = {user_id} ORDER BY created_at DESC LIMIT 1;") 
        for record_id, user_id, twitter_username, n_followers, n_retweets, n_tweets, n_retweets_to_user, n_favorites_to_user, n_replies_to_user, sentiment, engagement, spi, created_at in cur:
            record_dict = {
                "user_id": user_id,
                "username": twitter_username,
                "n_followers": n_followers,
                "n_retweets": n_retweets,
                "n_tweets": n_tweets,
                "n_retweets_to_user": n_retweets_to_user,
                "n_favorites_to_user": n_favorites_to_user,
                "n_replies_to_user": n_replies_to_user,
                "sentiment_twitter": sentiment,
                "relative_engagement": engagement,
                "spi": spi,
                "created_at": created_at
            }
        return pd.DataFrame(record_dict, index=[0])
        

    def compute_definitive_tw_spi(user_id, new_record, bonus_weight=0.15):
        cur.execute(f"SELECT COUNT(*) FROM twitter_stats WHERE user_id = {user_id};") 
        for result in cur: 
            count_records = result[0]
        if count_records == 0:
            return new_record
        else: 
            
            dict_change = {}
            num_cols = [col for col in new_record.columns if is_numeric_dtype(new_record[col])]
            dict_change["username"] = new_record["username"].unique()[0]

            latest_record_db = retrieve_latest_record_tw(user_id)
            for col in num_cols:
                dict_change[col] = new_record.loc[0][col] - latest_record_db.loc[0][col]
                
            df_change = pd.DataFrame(dict_change, index=[0])
            
            dict_relative_change = {}
            dict_change = df_change.to_dict('records')[0]
            dict_base = latest_record_db.loc[0].to_dict()

            for key in dict_change.keys():
                if key == "username":
                    dict_relative_change[key] = dict_change[key]
                elif key == "created_at" or key == "spi":
                    pass
                else:
                    if dict_base[key] == 0 and dict_change[key] != 0:
                        dict_relative_change[key] = 1
                    elif dict_base[key] == 0 and dict_change[key] == 0:
                        dict_relative_change[key] = 0
                    else:
                        dict_relative_change[key] = dict_change[key] / dict_base[key]
                        
            dict_tw_spi_bonus_weight = {
            'n_followers': 0.1,
            'n_retweets': 0.1,
            'n_tweets': 0.1,
            'n_retweets_to_user': 0.1,
            'n_favorites_to_user': 0.1,
            'n_replies_to_user': 0.1,
            'sentiment_twitter': 0.3,
            'relative_engagement': 0.1
            }

            bonus_spi = 0
            for k in dict_relative_change.keys():
                if k != "username" and k != "user_id":
                    bonus_spi += dict_tw_spi_bonus_weight[k]*dict_relative_change[k]

            new_record_upd = new_record.copy()
            new_record_upd.loc[0, "spi"] = new_record_upd.loc[0, "spi"] + bonus_weight*bonus_spi*100

            return new_record_upd[:1]

    # --- Compute general SPI and input the data into the database ---

    df_ig_fully_proc = pd.DataFrame()
    for user_id in processed_ig_data["user_id"].unique():
        new_record = processed_ig_data[processed_ig_data["user_id"] == user_id].reset_index().drop("index", axis=1)
        record_updated = compute_definitive_ig_spi(user_id, new_record)
        df_ig_fully_proc = pd.concat([df_ig_fully_proc, record_updated], axis=0, ignore_index=True)
        input_ig_data(cur, record_updated)

    df_tw_fully_proc = pd.DataFrame()
    for user_id in processed_tw_data["user_id"].unique():
        new_record = processed_tw_data[processed_tw_data["user_id"] == user_id].reset_index().drop("index", axis=1)
        record_updated = compute_definitive_tw_spi(user_id, new_record)
        df_tw_fully_proc = pd.concat([df_tw_fully_proc, record_updated], axis=0, ignore_index=True)
        input_tw_data(cur, record_updated)

    def input_spi_data(cur, record):
        record_dict = record.to_dict('records')[0]
        cur.execute("INSERT INTO spis(user_id, spi, spi_instagram, spi_twitter, created_at) VALUES (?, ?, ?, ?, now());",
        (record_dict["user_id"], record_dict["spi"], record_dict["spi_instagram"], record_dict["spi_twitter"]))

    spi_df = pd.merge(df_ig_fully_proc[["user_id", "spi"]], df_tw_fully_proc[["user_id", "spi"]], on="user_id", suffixes=["_instagram", "_twitter"])
    weight_ig = 0.7
    weight_tw = 0.3
    spi_df["spi"] = spi_df["spi_instagram"] * weight_ig +  spi_df["spi_twitter"] * weight_tw

    for user_id in spi_df["user_id"].unique():
        new_record = spi_df[spi_df["user_id"] == user_id].reset_index().drop("index", axis=1)
        input_spi_data(cur, new_record) 

    conn.commit()
    conn.close()
