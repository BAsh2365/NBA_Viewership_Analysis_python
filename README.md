# NBA_Viewership_Analysis_python
Analyzing viewership Data in the NBA to see what factors influence fan engagement.

# Analysis
When discussing the decline of NBA viewership over the past 2-3 decades, many note that this issue is a result of many factors, such as the over-monetization of the league (sponsorships, TV deals, streaming, etc), the decrease in NBA rivalries, and "tenacity" the league used to have, the introduction of social media highlight reels for the best plays, or even generational shifts in viewership from the Jordan era vs the Lebron era vs the new era (Shai, Victor Wenbanyama, Luka, etc).

When analyzing the data on NBA finals matchups from 1987-2024 (data taken from a Wikipedia dataset), we can infer and observe the data shown to predict why a viewer will tune into a finals matchup.

What we see is that the average viewership variance (confounding factors that drive up viewership on a game-by-game basis, potentially based on other outside variables such as MVP predictions, team storylines, etc) is one of the positive, driving factors for someone to tune into the NBA finals, followed by the game number. 

Interestingly, the length of the series has a negative effect on viewership, which doesn't make sense. Why is this the case?

Well, the answer points us back to our average viewership variance, which is a nuanced variable. This nuance includes the context of the NBA finals, the storylines (as mentioned above), the team structure, records, seeding, etc. So while it may look like series length has a less-than-ideal effect on the prediction, you have to remember that the variable itself is context-averse. 

We see that NBA viewership peaked around the year 1998-1999 (graph from Wikipedia) (around the time of Jordan's final stretch with the Bulls, leading to 6 rings after a series win over the Utah Jazz) and has decreased ever since that time period, despite the influx of talent from 2000-2024. Some attribute the majority of NBA ratings to Michael Jordan and his popularity in the 90s, pre-dating social media.

Truth be told, the decline in NBA viewership is due to many of the issues we've mentioned above; it could be all of them. It is the amalgamation of these factors that creates the question: What can the NBA do to increase viewership? 

It is a very complex question, but we think it's a simple formula:

- Enhance TV/streaming viewrship experiences (more camera angles, Social Media exclusive NBA edits, extended interview cuts)

- Up the stakes during the season/offseason. This has been done with the introduction of the NBA cup; however, if the NBA cup took place over the summer months, which showcased players who did not get enough rotation time, it could be more interesting to watch (with prizes such as increased draft lottery odds, minor schedule changes, etc). 

- Finally, bringing back old school NBA rivalries and potentially easing up on penalties and reffing (slightly controversial)!

Hope you all enjoyed this analysis!



# ROC curve and Feature Importance Plots

<img width="800" height="600" alt="Figure_1" src="https://github.com/user-attachments/assets/fed369b1-d304-4f06-a984-4a6a2ede59e6" />


<img width="1000" height="600" alt="Figure_2" src="https://github.com/user-attachments/assets/31f32dd6-ac9a-4a61-bf0f-71a8495a8e13" />


# Resources

https://neptune.ai/blog/f1-score-accuracy-roc-auc-pr-auc

https://www.datacamp.com/tutorial/understanding-logistic-regression-python

https://medium.com/learning-data/which-machine-learning-algorithm-should-i-use-for-my-analysis-962aeff11102

https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/ (potential usage with Oversampling if needed for larger-scale data)

Use of Claude 4.5 sonnet for data collection from https://en.wikipedia.org/wiki/NBA_Finals_television_ratings and model debugging

