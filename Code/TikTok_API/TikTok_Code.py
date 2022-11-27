import asyncio
from tiktokpy import TikTokPy
from TikTokApi import TikTokApi

# Users list
users = ["johnmayer", "therock", "charlidamelio", "shakira"]

# Async function to retrieve ids of the videos of the users
async def tiktokpy_query():
    async with TikTokPy() as bot:
        video_ids_dict = {}
        for user in users:
            user_feed_items = await bot.user_feed(username=user, amount=3)
            count = 0
            while len(user_feed_items) == 0 and count<3:
                user_feed_items = await bot.user_feed(username=user, amount=3)
                count = count + 1
            video_ids_dict[user] = [item.video.id for item in user_feed_items]
    return video_ids_dict

# Helper syncronizer function
def synchronize_async_helper(to_await):
    async_response = []

    async def run_and_capture_result():
        r = await to_await
        async_response.append(r)

    loop = asyncio.get_event_loop()
    coroutine = run_and_capture_result()
    loop.run_until_complete(coroutine)
    return async_response[0]


video_ids_dict = synchronize_async_helper(tiktokpy_query())

# proxy = "http://149.248.55.128:21218"

# video_ids_dict = {'johnmayer': ['7135887254077230382', '7084392445101772074', '7053962096660253999'], 'therock': ['7156636740671868202', '7155830902386445614', '7155355051563978030'], 'charlidamelio': ['7164109765776313643', 
# '7163444310539619630', '7163054194604133674'], 'shakira': ['7163706259676417322', '7161935832578854190', '7161118677046299950']}

# custom_verify_fp = "verify_laqz3ihb_907YcnG3_xaGA_4ccG_9WqC_Wgrm59Q17Jeg"
# ttwid = "7C9975dd381a481a0df02bc6e3c1309b99251bc157e444a51f78255798250d1c01"

# # From each video get the comments
# api = TikTokApi(custom_verify_fp = custom_verify_fp, ttwid=ttwid, use_test_endpoints=True)
# tiktok_video_id = video_ids_dict["johnmayer"][0]
# video = api.video(id=tiktok_video_id)
# print(video.create_time)
# for comment in video.comments():
#     print(comment.text)


