---
layout: post
title:  "Conversion Rate Optimisation"
date:   2017-07-25 18:11:16
categories: portfolio
featured: true
featured_image: "../../assets/img/project.png"
tags: Coursera
---

This is a data science project taken from the book A collection of Data Science Take Home challenges by Guilio Palombo. It os a pretty interesting book with a variety of analytical challenges that you may face working as a data scientist.

This particular challenge involves trying to predict
conversion rates on a website containing the following variables.

1. country : user country based on the IP address

2. age : user age. Self-reported at sign-in step

3. new_user : whether the user created the account during this session or had already an account and simply came back to the site

4. source : marketing channel source
Ads: came to the site by clicking on an advertisement
Seo: came to the site by clicking on search results
Direct: came to the site by directly typing the URL on the browser

5. total_pages_visited: number of total pages visited during the session. This is a proxy for time spent on site and engagement during the session.

6. converted: this is our label. 1 means they converted within the session, 0 means they left without buying anything. The company goal is to increase conversion rate: # conversions / total sessions.

Our goal is to first construct a model and use it to predict conversion rates and then provide recommendations based on our findings to try an increase conversion rates.

The project is done in python and the link to the notebook is [here](http://nbviewer.jupyter.org/github/DFoly/Data-Science-Projects/blob/master/Conversion%20Rate/Conversion%20Rate.ipynb)
