import asyncio
from tiktokpy import TikTokPy
from TikTokApi import TikTokApi

# users = [" ", "justinbieber", "therock", "charlidamelio", "shakira"]

users = [" ", "therock"]

async def tiktokpy_query():
    async with TikTokPy() as bot:
        video_ids_dict = {}
        for user in users:
            user_feed_items = await bot.user_feed(username=user, amount=3)
            video_ids_dict[user] = [item.video.id for item in user_feed_items]
    return video_ids_dict


def synchronize_async_helper(to_await):
    async_response = []

    async def run_and_capture_result():
        r = await to_await
        async_response.append(r)

    loop = asyncio.get_event_loop()
    coroutine = run_and_capture_result()
    loop.run_until_complete(coroutine)
    return async_response[0]

result = synchronize_async_helper(tiktokpy_query())
print(result)