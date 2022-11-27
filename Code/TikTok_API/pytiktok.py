from TikTokAPI import TikTokAPI

cookie = {
  "s_v_web_id": "verify_la8bz5oa_edtf4jiT_aeSt_4LSp_AtTR_yzQHNdzC4WeV",
  "tt_webid": "7C247af707fa3e98ea874f6eddac7a08a620f245b04e0b1ad2f88cee9cfedd0a00"
}

api = TikTokAPI(cookie=cookie)
user_videos = api.getVideosByUserName("therock")

print(user_videos)