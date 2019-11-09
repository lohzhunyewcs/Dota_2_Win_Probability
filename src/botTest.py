import praw
import os
import requests

app_id = 'dQqk27b1_7x-AA'
secret = 'ngPZ2lzK30Jks_0Wz4z-aV8y1q8'
user_agent = f'reddit:dota2.predictor:1.0.0 (by /u/ExpectedOutcome0)'

# Placeholder, TODO: config environment
username = 'ExpectedOutcome0'# os.getenv('reddit_username')
password = 'KhaiFungNumber1'# os.getenv('reddit_password')

reddit = praw.Reddit(client_id=app_id,
                     client_secret=secret,
                     user_agent=user_agent,
                     username=username,
                     password=password)

subreddit = reddit.subreddit('test')

heroTable = '\nRadiant Hero 1| Radiant Hero 2| Radiant Hero 3| Radiant Hero 4| Radiant Hero 5| Dire Hero 1|Dire Hero 2|Dire Hero 3|Dire Hero 4| Dire Hero 5'\
'\n---|---|----|----|----|----|----|----|----|----'\
'\n{rHero[0]}|{rHero[1]}|{rHero[2]}|{rHero[3]}|{rHero[4]}|{dHero[0]}|{dHero[1]}|{dHero[2]}|{dHero[3]}|{dHero[4]}'

post = subreddit.submit(title='test322', selftext=heroTable, url=None, spoiler=True)

newBody = post.selftext + '\nEdit: It works!'
post.edit(newBody)

