from TikTokApi import TikTokApi
import random
# verify_la8bz5oa_edtf4jiT_aeSt_4LSp_AtTR_yzQHNdzC4WeV
# Watch https://www.youtube.com/watch?v=-uCt1x8kINQ for a brief setup tutorial

custom_verify_fp = "verify_laqzuvay_McOL0ZE5_zufY_4V2o_8lLl_10GycSsgLhLy"
ttwid = "1%7C_Ya2QBsAhdoVSDoyVqWVeyd3DYbFpHRMUgqztYFGcLE%7C1669047639%7C765da0065c872839dbde39c1a6ff5f78e9d0f9e6450e0399ebcdac9143937717"


with TikTokApi(custom_verify_fp = custom_verify_fp, ttwid=ttwid, use_test_endpoints=True) as api: # .get_instance no longer exists
    tiktok_video_id = 7107272719166901550
    video = api.video(id=tiktok_video_id)

    for comment in video.comments():
        print(comment.text)

# api = TikTokApi(custom_verify_fp = custom_verify_fp, ttwid=ttwid, use_test_endpoints=True)
# user = api.user(username="therock")

# i = 0
# for video in user.videos():
#     if i > 10:
#         break
#     print(video.id)
#     i = i + 1

# for liked_video in api.user(username="public_likes").videos():
#     print(liked_video.id)

# This one works by replacing ttwid=spawn.cookies["ttwid"] by ttwid="7C0af72ef4deab1f3b56334cf21dc68aea5edf6154165136e0769d7417dd61db30" in video.py
# api = TikTokApi(custom_verify_fp = custom_verify_fp, ttwid=ttwid, use_test_endpoints=True)
# tiktok_video_id = 7107272719166901550
# video = api.video(id=tiktok_video_id)
# print(video.create_time)
# for comment in video.comments():
#     print(comment.text)


# proxies = open('proxies.txt', 'r').read().splitlines()
# proxy = random.choice(proxies)
# proxies = {'http': f'http://{proxy}', 'https': f'https://{proxy}'}