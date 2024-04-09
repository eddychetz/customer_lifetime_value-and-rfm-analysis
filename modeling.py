#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import joblib 

import plotnine as pn
from plotnine import *

import plotly.express as px

from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV

pn.options.dpi = 300

# Supressing warnings
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# ## 1.0 DATA PREPARATION

raw_df = pd.read_csv('./data/CDNOW_master.txt', sep='\s+', names=['customer_id', 'date', 'quantity', 'price'])
raw_df.info()

df = raw_df.assign(
    date = lambda x: x['date'].astype(str)
).assign(
    date = lambda x: pd.to_datetime(x['date'])
).dropna()
df.info()

# ## 2.0 COHORT ANALYSIS
# 
# - Only the customers that have joined at the specific business day

# Get Range of Initial Purchases
first_purchase_tbl = (
    df
    .sort_values(['customer_id', 'date'])
    .groupby('customer_id')
    .first()
)
first_purchase_tbl.head()

first_purchase_tbl.date.min()

first_purchase_tbl.date.max()

# Visualize: All purchases within cohort
(df
 .set_index('date')[['price']]
 .resample(rule='MS')
 .sum()
).plot();

# Visualize: Individual Customer Purchases
ids = df.customer_id.unique()
ids_selected = ids[0:12]

cust_id_subset_df=(
    df[df['customer_id'].isin(ids_selected)]
    .groupby(['customer_id', 'date'])
    .sum()
    .reset_index()
)

cust_id_subset_df.head()

# Plot using plotnine with facet_wrap of 3 columns
p = (
    ggplot(cust_id_subset_df, aes(x='date', y='price', fill='customer_id')) + 
    geom_line() +
    geom_point(show_legend=False) + 
    facet_wrap('~customer_id', scales='free_y', ncol=4) +  # Facet wrap with 3 columns
    labs(x='Date', y='Price') +  # Set axis labels
    theme(axis_text_x=element_text(hjust=1), figure_size=(12, 8))  # Rotate x-axis labels for better readability
)

# Add annotations
p =  p + scale_x_date(date_breaks = '1 year', date_labels = '%Y')
# Display the plot
print(p)


# ## 3.0 MACHINE LEARNING
# 
# #### Research Questions:
# 
# 1. What will the customers spend in the next 90-Days? (Regression)
# 2. What is the probability of a customer to make a purchase in next 90-Days? (Classification)
# 
# ### 3.1 TIME SPLITTING STAGE

n_days = 90
max_date = df['date'].max()

cutoff = max_date - pd.to_timedelta(n_days, unit = 'd')

temp_in_df = df[df.date <= cutoff]

temp_out_df = df[df.date > cutoff]

# ### 3.2 FEATURE ENGINEERING (RFM)

# __Keynotes:__
# - Most challenging part.
# - 2-Stage Process.
# - Need to frame the problem.
# - Need to think about what features to include.

# __Make Targets from out data__

targets_df = pd.DataFrame(
    temp_out_df
    .drop('quantity', axis=1)
    .groupby('customer_id')['price']
    .sum()
    .rename('spend_90_total') 
).assign(spend_90_flag = 1)

targets_df.head()

# __Make Recency (Date) Features from in data__

max_date = temp_in_df['date'].max()

recency_features_df = (
    pd.DataFrame(
        temp_in_df[['customer_id', 'date']]
        .groupby('customer_id')
        .apply(lambda x: (x['date'].max() - max_date) / pd.to_timedelta(1, 'day'), include_groups=False)
    ).set_axis(["recency"], axis=1)
)

recency_features_df.head()

# __Make Frequency (Count) Features from in data__

frequency_features_df = (
    temp_in_df[['customer_id', 'date']]
    .groupby('customer_id')
    .count()
    .set_axis(['frequency'], axis=1)
)

frequency_features_df.head()

# __Make Price (Monetary) Features from in data__

price_features_df = (
    temp_in_df
    .groupby('customer_id')
    .aggregate(
        {
            'price': ["sum", "mean"]
        }
    )
    .round(2)
    .set_axis(['price_sum', 'price_mean'], axis = 1)
)

price_features_df.head()

# ### 3.3 COMBINE FEATURES

features_df = (
    pd.concat([recency_features_df, frequency_features_df, price_features_df], axis = 1)
    .merge(
        targets_df, 
        left_index  = True, 
        right_index = True, 
        how         = "left"
    )).fillna(0)


# ## 4.0 MACHINE LEARNING

# ### 4.1 NEXT 90-DAY SPEND PREDICTION 

X = features_df[['recency', 'frequency', 'price_sum', 'price_mean']]

y_spend = features_df['spend_90_total']

# Build model
xgb_reg_spec = XGBRegressor(
    objective="reg:squarederror",   
    random_state=42
)

# Cross-Validation
xgb_reg_model = GridSearchCV(
    estimator=xgb_reg_spec, 
    param_grid=dict(
        learning_rate = [0.01, 0.1, 0.3, 0.5]
    ),
    scoring = 'neg_mean_absolute_error',
    refit   = True,
    cv      = 5
)

# Fit the model
xgb_reg_model.fit(X, y_spend)

best_score = xgb_reg_model.best_score_
print(f'Our model is off by ${best_score:.2f} on average per customer.')

xgb_reg_model.best_params_

xgb_reg_model.best_estimator_

predictions_reg = xgb_reg_model.predict(X)
predictions_reg

# ### 4.2 NEXT 90-DAY SPEND PROBABILITY

y_prob = features_df['spend_90_flag']

xgb_clf_spec = XGBClassifier(
    objective    = "binary:logistic",   
    random_state = 42
)

xgb_clf_model = GridSearchCV(
    estimator=xgb_clf_spec, 
    param_grid=dict(
        learning_rate = [0.01, 0.1, 0.3, 0.5]
    ),
    scoring = 'roc_auc',
    refit   = True,
    cv      = 5
)
xgb_clf_model.fit(X, y_prob)

xgb_clf_model.best_score_

xgb_clf_model.best_params_

xgb_clf_model.best_estimator_

predictions_clf = xgb_clf_model.predict_proba(X)

predictions_clf


# ### 4.3 FEATURE IMPORTANCE (GLOBAL)

# __Importance | Spend Amount Model__

imp_spend_amount_dict = (
    xgb_reg_model
    .best_estimator_
    .get_booster()
    .get_score(importance_type = 'gain') 
)
imp_spend_amount_dict

# Create a DataFrame from imp_spend_amount_dict
imp_spend_amount_df = pd.DataFrame(
    data={
        'feature': list(imp_spend_amount_dict.keys()),
        'value': list(imp_spend_amount_dict.values())
    }
).sort_values('value', ascending=False)
imp_spend_amount_df

fig = px.bar(
    (imp_spend_amount_df.round(2)).sort_values('value', ascending=True),
    x='value',
    y='feature',
    orientation='h'
)

# Show the plot
fig.show()


# ___Problem related to big spenders:___
# - Depending on the goal of the business, this has a profound effect if you want more people to spend more money, focus on the customers that have previously spent a lot of money.

# __Importance | Spend Probability Model__

imp_spend_prob_dict = (
    xgb_clf_model
    .best_estimator_ 
    .get_booster()
    .get_score(importance_type = 'gain') 
)

imp_spend_prob_df = pd.DataFrame(
    data  = {
        'feature':list(imp_spend_prob_dict.keys()),
        'value':list(imp_spend_prob_dict.values())
    }
).sort_values('value', ascending=False)

imp_spend_prob_df

fig = px.bar(
    (imp_spend_prob_df.round(2)).sort_values('value', ascending=True),
    x='value',
    y='feature',
    orientation='h'
)

# Show the plot
fig.show()


# ___Problem of having not to fall off the Wagon:___
#  - If you want to retain customers, focus on increasing the recency and frequency of transactions.

# ## 5.0 SAVE WORK 

# Save Predictions
predictions_df = pd.concat(
    [
        pd.DataFrame(predictions_reg).set_axis(['pred_spend'], axis=1),
        pd.DataFrame(predictions_clf)[[1]].set_axis(['pred_prob'], axis=1),
        features_df.reset_index()
    ], 
    axis=1
)

predictions_df.to_pickle("./artifacts/predictions_df.pkl")
pd.read_pickle('./artifacts/predictions_df.pkl')

# Save Importance
imp_spend_amount_df.to_pickle("artifacts/imp_spend_amount_df.pkl")
imp_spend_prob_df.to_pickle("artifacts/imp_spend_prob_df.pkl")

pd.read_pickle("artifacts/imp_spend_amount_df.pkl")

# Save Models
joblib.dump(xgb_reg_model, 'artifacts/xgb_reg_model.pkl')
joblib.dump(xgb_clf_model, 'artifacts/xgb_clf_model.pkl')

model = joblib.load('artifacts/xgb_reg_model.pkl')
model.predict(X)


# ### 6.0 HOW CAN WE USE THIS INFORMATION

# ### 6.1 Which customers have the highest spend probability in next 90-days? 
#    - Target for new products similar to what they have purchased in the past.

predictions_df.sort_values('pred_prob', ascending=False)


# ### 6.2 Which customers have recently purchased but are unlikely to buy? 
#    - Incentivize actions to increase probability - relates to recency.
#    - Provide discounts, encourage referring a friend, nurture by letting them know what's coming - revitalize customers.


recency = predictions_df['recency'] > -90
proba = predictions_df['pred_prob'] < 0.3
predictions_df[recency & proba].sort_values('pred_prob', ascending=False)


# ### 6.3 Missed opportunities: Big spenders that could be unlocked
#    - Send bundle offers encouraging volume purchases - focus on customers predicted spending but did not spend any money

# In[56]:


zero_spenders =  predictions_df['spend_90_total'] == 0.0
predictions_df[zero_spenders].sort_values('pred_spend', ascending=False) 


# __Foundational skills set:__
#    - Data Wrangling
#    - Modeling
#    - Visualization

# ## 7.0 WEB APPLICATION


predictions_df = pd.read_pickle('artifacts/predictions_df.pkl')

df = predictions_df.assign(
    spend_actual_vs_pred = lambda x: x['spend_90_total'] - x['pred_spend'] 
)

fig = px.scatter(
    data_frame=df,
    x = 'frequency',
    y = 'pred_prob',
    color = 'spend_actual_vs_pred', 
    color_continuous_midpoint=0, 
    opacity=0.5, 
    color_continuous_scale='IceFire'
)

# Add annotations
fig.update_layout(
    {'plot_bgcolor': 'white'}
)
fig.show()


