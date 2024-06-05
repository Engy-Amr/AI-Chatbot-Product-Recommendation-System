#!/usr/bin/env python
# coding: utf-8

# In[126]:


#pip install --user numpy


# In[127]:


#pip install --user pandas


# In[128]:


#pip install --user matplotlib


# In[129]:


#pip install --user seaborn


# In[130]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv
import matplotlib.pyplot as plt
import seaborn as sns


# In[131]:


df = pd.read_csv('/Users/engyamr/Downloads/Python/sales_data.csv')
df.head()


# In[132]:


df.info()


# In[133]:


df.isnull().sum()


# In[134]:


df.rename(columns={'catégorie': 'Category'}, inplace=True)

data_mapping = {
    'Vêtements': 'Clothes',
    'Électronique': 'Electronics'
}
df['Category'] = df['Category'].map(data_mapping).fillna(df['Category'])


# In[135]:


unique_products = df['Product'].unique()

# Print the unique values
for product in unique_products:
    print(product)


# In[136]:


unique_categories = df['Category'].unique()

# Print the unique values
for category in unique_categories:
    print(category)


# In[137]:


#we make a def funtion to minimize our unique values to lower
def change(x):
    if x in ['USB-C Charging Cable','Lightning Charging Cable']:
        return 'Charging Cables'
    elif x in ['AAA Batteries (4-pack)','AA Batteries (4-pack)']:
        return 'Batteries'
    elif x in ['Wired Headphones','Apple Airpods Headphones','Bose SoundSport Headphones']:
        return 'Headphones'
    elif x in ['27in FHD Monitor','27in 4K Gaming Monitor','34in Ultrawide Monitor','Flatscreen TV','20in Monitor']:
        return 'Smart Tv'
    elif x in ['iPhone','Google Phone','Vareebadd Phone']:
        return 'Smart Phones'
    elif x in ['Macbook Pro Laptop','ThinkPad Laptop']:
        return 'Laptops'
    elif x in ['LG Washing Machine','LG Dryer']:
        return 'Cleaning Machines'
    else:
        return 'Others'


# In[138]:


df['Category'] = df['Product'].apply(change)


# # Data Preprocessing

# Date-Time Conversion

# In[139]:


df['Order Date'] = pd.to_datetime(df['Order Date'])

# Verify the changes
print(df.dtypes)


# In[140]:


# Select only numeric columns
# numeric_df = df.select_dtypes(include='number')

# Compute the correlation matrix
# corr_matrix = numeric_df.corr()

# Set up the matplotlib figure
# plt.figure(figsize=(10, 8))

# Define a custom color palette (replace 'coolwarm' with the desired colormap)
# custom_cmap = sns.color_palette('coolwarm', as_cmap=True)

# Create a heatmap with the custom colormap
# sns.heatmap(corr_matrix, cmap=custom_cmap, annot=True, fmt=".2f", linewidths=.5)

# Display the plot
# plt.show()


# Outlier Detection and Handling in Data Preprocessing

# In[141]:


# Calculate IQR for each numerical column
Q1 = df.quantile(0.25, numeric_only=True)
Q3 = df.quantile(0.75, numeric_only=True)
IQR = Q3 - Q1

# Define the outlier threshold
threshold = 1.5

# Identify outliers
outliers = ((df.select_dtypes(include='number') < (Q1 - threshold * IQR)) |
            (df.select_dtypes(include='number') > (Q3 + threshold * IQR))).any(axis=1)

# Display or handle outliers as needed
df_outliers = df[outliers]


# In[142]:


df.info()


# Monthly Aggregation

# In[143]:


df['Order Date'] = pd.to_datetime(df['Order Date'])

# Extract month and year
df["Year"]=df["Order Date"].dt.year
df["Month"]=df["Order Date"].dt.month

# Verify the changes
print(df[['Order Date', 'Month', 'Year']].head())

# Aggregate sales data on a monthly basis
#monthly_sales = dataset.groupby(['Year', 'Month'])['turnover'].sum().reset_index()
monthly_aggregated = df.groupby(['Year', 'Month']).agg({
    'Quantity Ordered': 'sum',
    'turnover': 'sum',
    'margin': 'mean'
}).reset_index()

print(monthly_aggregated)


# Feature Engineering

# In[144]:


df['Total Sales'] = df['Quantity Ordered'] * df['Price Each']

# Verify the changes
print(df[['Order Date', 'Month', 'Year', 'Total Sales']].head())


# # Data Analysis

# Trend Visualization

# In[145]:


import matplotlib.dates as mdates

# Assuming 'Order Date' column is in datetime format
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Extract month and year from the 'Order Date' column
df['Month'] = df['Order Date'].dt.to_period('M')

# Group by month and calculate the sum of 'Total Sales'
monthly_sales = df.groupby('Month')['Total Sales'].sum().reset_index()

# Convert the 'Month' column to datetime for better formatting
monthly_sales['Month'] = monthly_sales['Month'].dt.to_timestamp()

# Specify the months you want to include
selected_months = ['2019-01', '2019-03', '2019-05', '2019-07', '2019-09', '2019-11', '2020-01']

# Filter the DataFrame based on selected months
selected_sales = monthly_sales[monthly_sales['Month'].dt.to_period('M').astype(str).isin(selected_months)]

# Plotting the line graph
# plt.figure(figsize=(10, 6))
# plt.plot(selected_sales['Month'], selected_sales['Total Sales'], marker='o', linestyle='-')
# plt.title('Monthly Total Sales')
# plt.xlabel('Month')
# plt.ylabel('Total Sales')

# Set x-axis date format
# date_format = mdates.DateFormatter('%b %Y')
# plt.gca().xaxis.set_major_formatter(date_format)

# Set y-axis ticks and labels
# plt.yticks([0, 1000000, 2000000, 3000000, 4000000], ['0', '1M', '2M', '3M', '4M'])

# plt.tight_layout()
# plt.grid(True)
# plt.show()


# Time Series Data Composition

# Monthly Sales Growth Rate

# In[146]:


df.set_index('Order Date', inplace=True)

# Specify the column(s) to sum explicitly
monthly_sales = df['Total Sales'].resample('M').sum().to_frame()

# Calculate the monthly sales growth rate
monthly_sales['Sales Growth Rate'] = monthly_sales['Total Sales'].pct_change() * 100

# Plot the Monthly Sales Growth Rate line graph
# plt.figure(figsize=(10, 6))
plt.plot(monthly_sales.index, monthly_sales['Sales Growth Rate'], marker='o', linestyle='-', color='b')
# plt.title('Monthly Sales Growth Rate')
# plt.xlabel('Date')
# plt.ylabel('Growth Rate (%)')
# plt.grid(True)
# plt.show()


# Creating User ID Column

# In[147]:


#Creating a new Column for User ID
# Generate User IDs based on Purchase Address
# Create a dictionary to map Purchase Address to User ID starting from 1
user_id_map = {address: i + 1 for i, address in enumerate(df['Purchase Address'].unique())}

# Map Purchase Address to User ID
df['User ID'] = df['Purchase Address'].map(user_id_map)
df.head()


# In[148]:


# Check if every repeated "Purchase Address" has the same "User ID"
is_same_user_id = df.groupby('Purchase Address')['User ID'].nunique().eq(1).all()

print("Every repeated Purchase Address has the same User ID:", is_same_user_id)


# In[149]:


# Check the number of unique values in the "Purchase Address" column
unique_purchase_addresses = df['Purchase Address'].nunique()
print("Number of unique values in Purchase Address column:", unique_purchase_addresses)

# Check the number of unique values in the "User ID" column
unique_user_ids = df['User ID'].nunique()
print("Number of unique values in User ID column:", unique_user_ids)


# In[150]:


#Creating a new column for the Product ID by Product

# Dictionary to store product IDs
product_ids = {}
product_id_counter = 1

# Function to generate product IDs
def generate_product_id(product):
    global product_id_counter
    if product not in product_ids:
        product_ids[product] = product_id_counter
        product_id_counter += 1
    return product_ids[product]

# Applying the function to create Product ID column
df['Product ID'] = df['Product'].apply(generate_product_id)

# Display the updated dataset
df.head()


# In[151]:


# Checking if any Product ID is repeated for the same Product
df['Is Repeated'] = df.duplicated(subset=['Product ID'])


# In[152]:


df.head()


# In[153]:


print(df.columns)


# # New Dataset combining all Products ordered by the same user

# # Model Development

# Creating a new dataset, where I add all products ordered for all users in one column called Products_Ordered

# In[154]:


# Group by 'User ID' and concatenate all products into a single string
grouped_df = df.groupby('User ID')[['Product ID', 'Quantity Ordered']].apply(lambda x: ' | '.join([str(p) for p, q in zip(x['Product ID'], x['Quantity Ordered']) for _ in range(q)])).reset_index(name='Product_Ordered')

# Display the new dataset
grouped_df.head()


# In[155]:


# Group by 'User ID' and concatenate all products into a single string
#grouped_df = df.groupby('User ID').apply(lambda x: ' | '.join([str(p) for p, q in zip(x['Product ID'], x['Quantity Ordered']) for _ in range(q)])).reset_index(name='Product_Ordered')

# Display the new dataset
#grouped_df.head()


# In[156]:


user_rows = grouped_df[grouped_df['User ID'] == 3]

user_rows.head()


# In[157]:


user_rows = df[df['User ID'] == 3]

user_rows.head()


# In[158]:


product_mapping = df[['Product ID', 'Product']].drop_duplicates()

# Display the DataFrame without showing the index
print(product_mapping.to_string(index=False))


# User-Product Interacion

# In[159]:


grouped_df['Product_Ordered'] = grouped_df['Product_Ordered'].apply(lambda x: [int(i) for i in x.split('|')])
grouped_df.head(2).set_index('User ID')['Product_Ordered'].apply(pd.Series).reset_index()


# In[160]:


user_interacted_product = pd.melt(grouped_df.set_index('User ID')['Product_Ordered'].apply(pd.Series).reset_index(),
                                     id_vars=['User ID'],
                                     value_name='Product_Ordered'
                                    ).dropna().drop(['variable'], axis=1).rename(columns={'Product_Ordered': 'productId'}).reset_index(drop=True)
user_interacted_product['productId'] = user_interacted_product['productId'].astype(np.int64)
user_interacted_product.head()


# In[161]:


#pip install scikit-learn


# In[162]:


from sklearn.model_selection import train_test_split

data_users = user_interacted_product['User ID']
data_items = user_interacted_product['productId']
# split the data test and train
train_users, test_users, train_items, test_items = train_test_split(data_users, data_items,
                                                                    test_size=0.2, random_state=42, shuffle=True)
train_data = pd.DataFrame((zip(train_users, train_items)),columns=['User ID', 'productId'])

train_data.head()


# In[163]:


train_data['interactions'] = 1
train_data.head()


# In[ ]:


#pip install tqdm


# In[ ]:


#pip install ipywidgets==7.6.5


# In[164]:


# Get a list of all sku_ids
all_product_ids = train_data['productId'].unique()
all_customer_ids = train_data['User ID'].unique()
# Placeholders that will hold the training data
customerId, productId, interactions = [], [], []
# This is the set of items that each user has interaction with
customer_product_set = set(zip(train_data['User ID'], train_data['productId']))
# 4:1 ratio of negative to positive samples
num_negatives = 4

# Iterate through the customer_product_set without tqdm
for (u, i) in customer_product_set:
    customerId.append(u)
    productId.append(i)
    interactions.append(1) # items that the user has interacted with are positive
    for _ in range(num_negatives):
        # randomly select an item
        negative_item = np.random.choice(all_product_ids)
        # check that the user has not interacted with this item
        while (u, negative_item) in customer_product_set:
            negative_item = np.random.choice(all_product_ids)
        customerId.append(u)
        productId.append(negative_item)
        interactions.append(0) # items not interacted with are negative

interaction_matrix = pd.DataFrame(list(zip(customerId, productId, interactions)),columns=['User ID', 'productId', 'interactions'])
interaction_matrix.head()


# Embedding Layers, Concatenation Layer, Output Layer

# In[342]:


data_x = np.array(interaction_matrix[['User ID', 'productId']].values)
data_y = np.array(interaction_matrix[['interactions']].values)
# split validation data
train_data_x, val_data_x, train_data_y, val_data_y = train_test_split(data_x, data_y, test_size=0.1, random_state=42, shuffle=True)
print("Train Data Shape {}".format(train_data_x.shape))
print("Validation Data Shape {}".format(val_data_x.shape))


# In[343]:


# train data
train_data_users = train_data_x[:,0]
train_data_items = train_data_x[:,1]
# validation data
val_data_users = val_data_x[:,0]
val_data_items = val_data_x[:,1]


# In[344]:


number_of_users = train_data['User ID'].max()
number_of_items = train_data['productId'].max()
latent_dim_mf = 4
latent_dim_mlp = 32
reg_mf = 0
reg_mlp = 0.1
dense_layers = [128, 64, 32]
reg_layers = [0.1, 0.1, 0.1]
activation_dense = "relu"


# In[168]:


#pip install tensorflow


# In[345]:


import random
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate, Dense, Embedding, Flatten, Input, Multiply, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

# Define dropout rate
dropout_rate = 0.05  # Example: 20%

# input layer
user = Input(shape=(), dtype="int64", name="user_id")
item = Input(shape=(), dtype="int64", name="item_id")

# Embedding layers with L2 regularization and dropout
user_embedding = Embedding(
    input_dim=number_of_users + 1,
    output_dim=latent_dim_mf,
    embeddings_initializer="RandomNormal",
    embeddings_regularizer=l2(reg_mf),
    name="user_embedding",
)
item_embedding = Embedding(
    input_dim=number_of_items + 1,
    output_dim=latent_dim_mf,
    embeddings_initializer="RandomNormal",
    embeddings_regularizer=l2(reg_mf),
    name="item_embedding",
)

# Add dropout after embedding layers
user_dropout = Dropout(dropout_rate)(user_embedding(user))
item_dropout = Dropout(dropout_rate)(item_embedding(item))

# MF vector
user_latent = Flatten()(user_dropout)
item_latent = Flatten()(item_dropout)
mf_cat_latent = Multiply()([user_latent, item_latent])

# MLP vector
mlp_user_latent = Flatten()(user_dropout)
mlp_item_latent = Flatten()(item_dropout)
mlp_cat_latent = Concatenate()([mlp_user_latent, mlp_item_latent])

# Add dropout after concatenation for MLP
mlp_dropout = Dropout(dropout_rate)(mlp_cat_latent)

# Concatenation
mlp_vector = mlp_dropout

# build dense layers for model
for i in range(len(dense_layers)):
    layer = Dense(
            dense_layers[i],
            activity_regularizer=l2(reg_layers[i]),
            activation=activation_dense,
            name="layer%d" % i)
    mlp_vector = layer(mlp_vector)

NeuMf_layer = Concatenate()([mf_cat_latent, mlp_vector])

result = Dense(1, activation="relu", kernel_initializer="lecun_uniform", name="interaction")
output = result(NeuMf_layer)
model = Model(inputs=[user, item], outputs=output)

model.summary()


# In[346]:


model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mse",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ],
)


# In[ ]:


# history = model.fit(x=[train_data_users, train_data_items], y=train_data_y,
#                       batch_size=64, epochs=50,
#                       validation_data=([val_data_users, val_data_items], val_data_y))


# In[ ]:





# In[349]:


import random
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate, Dense, Embedding, Flatten, Input, Multiply, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

# Define dropout rate
dropout_rate = 0.05  # Example: 20%

# input layer
user = Input(shape=(), dtype="int64", name="user_id")
item = Input(shape=(), dtype="int64", name="item_id")

# Embedding layers with L2 regularization and dropout
user_embedding = Embedding(
    input_dim=number_of_users + 1,
    output_dim=latent_dim_mf,
    embeddings_initializer="RandomNormal",
    embeddings_regularizer=l2(reg_mf),
    name="user_embedding",
)
item_embedding = Embedding(
    input_dim=number_of_items + 1,
    output_dim=latent_dim_mf,
    embeddings_initializer="RandomNormal",
    embeddings_regularizer=l2(reg_mf),
    name="item_embedding",
)

# Add dropout after embedding layers
user_dropout = Dropout(dropout_rate)(user_embedding(user))
item_dropout = Dropout(dropout_rate)(item_embedding(item))

# MF vector
user_latent = Flatten()(user_dropout)
item_latent = Flatten()(item_dropout)
mf_cat_latent = Multiply()([user_latent, item_latent])

# MLP vector
mlp_user_latent = Flatten()(user_dropout)
mlp_item_latent = Flatten()(item_dropout)
mlp_cat_latent = Concatenate()([mlp_user_latent, mlp_item_latent])

# Add dropout after concatenation for MLP
mlp_dropout = Dropout(dropout_rate)(mlp_cat_latent)

# Concatenation
mlp_vector = mlp_dropout

# Build dense layers for DNN with ReLU activation
dense_layers = [128, 64, 32]  # Example dense layers configuration
for i, units in enumerate(dense_layers):
    mlp_vector = Dense(units, activation='relu', name=f'dense_{i}')(mlp_vector)

# Concatenate MF vector and DNN output
NeuMf_layer = Concatenate()([mf_cat_latent, mlp_vector])

# Output layer with softmax activation for multi-class classification
output = Dense(number_of_items, activation='softmax', name='output')(NeuMf_layer)

model.summary()


# In[350]:


model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mse",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ],
)


# In[351]:


history = model.fit(x=[train_data_users, train_data_items], y=train_data_y,
                      batch_size=64, epochs=50,
                      validation_data=([val_data_users, val_data_items], val_data_y))


# # Testing Model recommendations

# In[ ]:


df.head()


# In[ ]:


# Assuming your dataset has a column named 'Product ID' for product IDs
all_product_ids = df['Product ID'].unique()

# Function to generate negative samples
def generate_negative_samples(customer_product_set, all_product_ids, num_negatives=4):
    customerId, productId, interactions = [], [], []
    
    for (u, i) in customer_product_set:
        customerId.append(u)
        productId.append(i)
        interactions.append(1)  # Items that the user has interacted with are positive
        for _ in range(num_negatives):
            # Randomly select an item
            negative_item = np.random.choice(all_product_ids)
            # Check that the user has not interacted with this item
            while (u, negative_item) in customer_product_set:
                negative_item = np.random.choice(all_product_ids)
            customerId.append(u)
            productId.append(negative_item)
            interactions.append(0)  # Items not interacted with are negative

    return customerId, productId, interactions

# Assuming your dataset has 'User ID' and 'Product ID' columns
data_users = df['User ID']
data_items = df['Product ID']

# Split the data into training and test sets
train_users, test_users, train_items, test_items = train_test_split(data_users, data_items,
                                                                    test_size=0.2, random_state=42)

# Create a set of items that each user has interacted with
customer_product_set = set(zip(train_users, train_items))

# Generate negative samples
customerId, productId, interactions = generate_negative_samples(customer_product_set, all_product_ids)

# Create interaction matrix
interaction_matrix = pd.DataFrame(list(zip(customerId, productId, interactions)),
                                  columns=['User ID', 'Product ID', 'Interactions'])

# Display interaction matrix
print(interaction_matrix.head())


# In[ ]:


def generate_negative_samples(customer_product_set, all_product_ids, num_negatives=4):
    customerId, productId, interactions = [], [], []
    
    for (u, i) in customer_product_set:
        customerId.append(u)
        productId.append(i)
        interactions.append(1)  # Items that the user has interacted with are positive
        
        negative_samples = set()  # Store negative samples to avoid duplicates
        
        for _ in range(num_negatives):
            # Randomly select an item
            negative_item = np.random.choice(all_product_ids)
            # Check that the user has not interacted with this item
            while (u, negative_item) in customer_product_set or negative_item in negative_samples:
                negative_item = np.random.choice(all_product_ids)
            negative_samples.add(negative_item)  # Add negative sample to set to avoid duplicates
            customerId.append(u)
            productId.append(negative_item)
            interactions.append(0)  # Items not interacted with are negative

    return customerId, productId, interactions


# In[ ]:


# Assuming your dataset has 'User ID' and 'Product ID' columns
data_users = df['User ID']
data_items = df['Product ID']

# Extract all product IDs from your dataset
all_product_ids = df['Product ID'].unique()

# Split the data into training and test sets
train_users, test_users, train_items, test_items = train_test_split(data_users, data_items,
                                                                    test_size=0.2, random_state=42)

# Create a set of items that each user has interacted with
customer_product_set = set(zip(train_users, train_items))

# Generate negative samples without duplicates
customerId, productId, interactions = generate_negative_samples(customer_product_set, all_product_ids)

# Create interaction matrix
interaction_matrix = pd.DataFrame(list(zip(customerId, productId, interactions)),
                                  columns=['User ID', 'Product ID', 'Interactions'])

# Display interaction matrix
print(interaction_matrix.head())


# In[ ]:


# test_data contains user-item pairs for testing
test_data = pd.DataFrame({'User ID': test_users, 'Product ID': test_items})

# Assuming your recommendation model is called 'model'
# Predict the interaction probabilities for test data
predicted_interactions = model.predict([test_data['User ID'], test_data['Product ID']])

# Combine the predictions with the test data
test_data['Predicted Interaction'] = predicted_interactions

# Sort the recommendations by predicted interaction probabilities
recommendations = test_data.sort_values(by='Predicted Interaction', ascending=False)

# Define the number of recommendations to generate for each user
N = 5  # For example, generate top 5 recommendations for each user

# Select the top-N recommendations for each user
top_n_recommendations = recommendations.groupby('User ID').head(N)


# In[ ]:


# Assuming your dataset has 'User ID' and 'Product ID' columns
data_users = df['User ID']
data_items = df['Product ID']

# Split the data into training and test sets
train_users, test_users, train_items, test_items = train_test_split(data_users, data_items,
                                                                    test_size=0.2, random_state=42)

# Define the user for whom you want to generate recommendations
target_user_id = 13  # Example user ID

# Retrieve the user's interactions or purchase history from your dataset
user_interactions = df[df['User ID'] == target_user_id]['Product ID'].unique()

# Filter out products that the user has already interacted with
uninteracted_products = np.setdiff1d(all_product_ids, user_interactions)

# Reshape input data if necessary
user_ids = np.full_like(uninteracted_products, target_user_id)
user_ids = user_ids.reshape((-1, 1))  # Assuming user_ids is a 1D array
uninteracted_products = uninteracted_products.reshape((-1, 1))  # Assuming uninteracted_products is a 1D array

# Predict the interaction probabilities for uninteracted products
predicted_interactions = model.predict([user_ids, uninteracted_products])

# Combine the predictions with the uninteracted products
recommendations_df = pd.DataFrame({'User ID': user_ids.flatten(), 'Product ID': uninteracted_products.flatten(), 'Predicted Interaction': predicted_interactions.flatten()})

# Sort the recommendations by predicted interaction probabilities
sorted_recommendations = recommendations_df.sort_values(by='Predicted Interaction', ascending=False)

# Select the top N recommendations
N = 5  # Number of recommendations to generate
top_recommendations = sorted_recommendations.head(N)

# Print or display the top recommendations
print("Top Recommendations for User", target_user_id)
print(top_recommendations)

#for User ID 13 it recommended Product ID 9 with a prediction interaction of 0.782764


# In[ ]:


# Assuming your dataset has 'User ID' and 'Product ID' columns
data_users = df['User ID']
data_items = df['Product ID']

# Split the data into training and test sets
train_users, test_users, train_items, test_items = train_test_split(data_users, data_items,
                                                                    test_size=0.2, random_state=42)

# Define the user for whom you want to generate recommendations
target_user_id = 13  # Example user ID

# Retrieve the user's interactions or purchase history from your dataset
user_interactions = df[df['User ID'] == target_user_id]['Product ID'].unique()

# Filter out products that the user has already interacted with
uninteracted_products = np.setdiff1d(all_product_ids, user_interactions)

# Reshape input data if necessary
user_ids = np.full_like(uninteracted_products, target_user_id)
user_ids = user_ids.reshape((-1, 1))  # Assuming user_ids is a 1D array
uninteracted_products = uninteracted_products.reshape((-1, 1))  # Assuming uninteracted_products is a 1D array

# Predict the interaction probabilities for uninteracted products
predicted_interactions = model.predict([user_ids, uninteracted_products])

# Combine the predictions with the uninteracted products
recommendations_df = pd.DataFrame({'User ID': user_ids.flatten(), 'Product ID': uninteracted_products.flatten(), 'Predicted Interaction': predicted_interactions.flatten()})

# Sort the recommendations by predicted interaction probabilities
sorted_recommendations = recommendations_df.sort_values(by='Predicted Interaction', ascending=False)

# Select the top N recommendations
N = 5  # Number of recommendations to generate
top_recommendations = sorted_recommendations.head(N)

# Print or display the top recommendations
print("Top Recommendations for User", target_user_id)
print(top_recommendations)

#for User ID 13 it recommended Product ID 9 with a prediction interaction of 0.782764


# # Model Evaluation and Results

# In[353]:


# Access precision and recall from the history object
train_precision = history.history['precision']
train_recall = history.history['recall']

val_precision = history.history['val_precision']
val_recall = history.history['val_recall']

print("Train Precision:", train_precision)
print("Train Recall:", train_recall)

print("Validation Precision:", val_precision)
print("Validation Recall:", val_recall)


# Precision and Recall

# In[354]:


# Precision and recall values for each epoch
precision_values = train_precision
recall_values = train_recall

# Calculate average precision and recall
avg_precision = np.mean(precision_values)
avg_recall = np.mean(recall_values)

print("Average Precision:", avg_precision)
print("Average Recall:", avg_recall)


# In[355]:


# Precision and recall values for each epoch
precision_values = [0.8755, 0.8803, 0.8790, 0.8766, 0.8771, 0.8734, 0.8710, 0.8717, 0.8708, 0.8693, 0.8697, 0.8680, 0.8681, 0.8680, 0.8662, 0.8693, 0.8672, 0.8699, 0.8675, 0.8686, 0.8695, 0.8690, 0.8679, 0.8671, 0.8672, 0.8680, 0.8685, 0.8666, 0.8683, 0.8697, 0.8659, 0.8645, 0.8647, 0.8640, 0.8661, 0.8638, 0.8663, 0.8638, 0.8643, 0.8621, 0.8635, 0.8637]
recall_values = [0.7732, 0.7775, 0.7720, 0.7665, 0.7664, 0.7635, 0.7620, 0.7576, 0.7580, 0.7592, 0.7570, 0.7573, 0.7566, 0.7556, 0.7583, 0.7591, 0.7560, 0.7571, 0.7581, 0.7548, 0.7575, 0.7570, 0.7546, 0.7580, 0.7567, 0.7588, 0.7587, 0.7553, 0.7576, 0.7586, 0.7578, 0.7570, 0.7541, 0.7503, 0.7532, 0.7500, 0.7495, 0.7483, 0.7469, 0.7472, 0.7467, 0.7486]

# Calculate average precision and recall
avg_precision = np.mean(precision_values)
avg_recall = np.mean(recall_values)

print("Average Precision:", avg_precision)
print("Average Recall:", avg_recall)


# CTR

# In[356]:


# Extract relevant metrics from the training history
train_precision = history.history['precision']
train_recall = history.history['recall']

val_precision = history.history['val_precision']
val_recall = history.history['val_recall']

# Calculate the average precision and recall over all epochs
avg_train_precision = np.mean(train_precision)
avg_train_recall = np.mean(train_recall)
avg_val_precision = np.mean(val_precision)
avg_val_recall = np.mean(val_recall)

# Calculate the Click-Through Rate (CTR) using precision and recall
# For example, you can use the average validation precision as the CTR
CTR = avg_val_precision

print("Average Validation Precision (CTR):", CTR)


# In[357]:


import matplotlib.pyplot as plt

# Precision, Recall, and CTR values
precision_value = avg_precision
recall_value = avg_recall
ctr_value = CTR

# Bar chart data
labels = ['Precision', 'Recall', 'CTR']
values = [precision_value, recall_value, ctr_value]

# Plotting the bar chart
# plt.figure(figsize=(8, 6))
# plt.bar(labels, values, color=['navy', 'yellow', 'red'])
# plt.xlabel('Metrics')
# plt.ylabel('Values')
# plt.title('Model Evaluation Metrics')

# Display the values on top of each bar
# for i, value in enumerate(values):
#     plt.text(i, value + 0.01, str(round(value, 4)), ha='center')

# Show plot
# plt.show()


# In[358]:


import matplotlib.pyplot as plt

# Precision, Recall, and CTR values
precision_value = avg_precision
recall_value = avg_recall
ctr_value = CTR

# Bar chart data
labels = ['Precision', 'Recall', 'CTR']
values = [precision_value, recall_value, ctr_value]

# Plotting the bar chart
# plt.figure(figsize=(8, 6))
# plt.bar(labels, values, color=['navy', 'yellow', 'red'])
# plt.xlabel('Metrics')
# plt.ylabel('Values')
# plt.title('Model Evaluation Metrics')

# Display the values on top of each bar with 2 decimal places
# for i, value in enumerate(values):
#     plt.text(i, value + 0.01, "{:.2f}".format(value), ha='center')

# Show plot
# plt.show()


# # Traditional Collaborative Filtering

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.neighbors import NearestNeighbors

# Assuming df contains the DataFrame with user-item interactions
# Assuming we have a user-item matrix where rows are users and columns are items
# Here, we'll use a simple nearest neighbors approach for collaborative filtering

# Replace NaN values with 0
user_item_matrix = df.pivot_table(index='User ID', columns='Product ID', values='Quantity Ordered', fill_value=0)

# Initialize the Nearest Neighbors model
k = 5  # Number of neighbors to consider
nn_model = NearestNeighbors(n_neighbors=k, metric='cosine')

# Fit the model to the user-item matrix
nn_model.fit(user_item_matrix)


# In[ ]:


def recommend_items(user_id, n=10):
    # Get the index of the user in the user-item matrix
    user_index = user_item_matrix.index.get_loc(user_id)
    # Find the k nearest neighbors
    distances, indices = nn_model.kneighbors(user_item_matrix.iloc[user_index, :].values.reshape(1, -1))
    # Combine items purchased by the neighbors
    recommended_items = set()
    for idx in indices.flatten():
        recommended_items.update(user_item_matrix.iloc[idx, :].to_numpy().nonzero()[0])  # Correction made here
    # Remove items already purchased by the user
    recommended_items -= set(user_item_matrix.iloc[user_index, :].to_numpy().nonzero()[0])  # Correction made here
    # Convert indices to actual item IDs
    recommended_items = [user_item_matrix.columns[idx] for idx in recommended_items]
    # Return top n recommended items
    return recommended_items[:n]

# Function to calculate precision
def calculate_precision(actual_items, recommended_items):
    true_positives = len(set(actual_items) & set(recommended_items))
    false_positives = len(recommended_items) - true_positives
    if true_positives + false_positives > 0:
        precision = true_positives / (true_positives + false_positives)
    else:
        precision = 0
    return precision

# Function to calculate recall
def calculate_recall(actual_items, recommended_items):
    true_positives = len(set(actual_items) & set(recommended_items))
    false_negatives = len(set(actual_items) - set(recommended_items))
    if true_positives + false_negatives > 0:
        recall = true_positives / (true_positives + false_negatives)
    else:
        recall = 0
    return recall


# In[ ]:


# Assuming you have ground truth actual_items and recommended_items
# Replace 'your_user_id_here' with the actual user ID for whom you want to generate the list of actual items purchased
user_id = 12
actual_items_df = df[df['User ID'] == user_id]['Product ID']
actual_items = actual_items_df.tolist()

# Get recommended items for the user
recommended_items = recommend_items(user_id)

# Calculate precision and recall
precision = calculate_precision(actual_items, recommended_items)
recall = calculate_recall(actual_items, recommended_items)

# Print precision and recall
print("Precision:", precision)
print("Recall:", recall)


# In[ ]:


# Create a user-item matrix (rows = customers, columns = items)
user_item_matrix = df.pivot_table(index='User ID', columns='Product ID', values='Quantity Ordered', fill_value=0)
user_item_matrix


# In[ ]:


from sklearn.metrics.pairwise import pairwise_distances

# Convert DataFrame to numpy array
user_item_array = user_item_matrix.to_numpy()

# Calculate the item-item Jaccard similarity using pairwise_distances
item_item_similarity = 1 - pairwise_distances(user_item_array.T, metric='jaccard')


# # Creating Product Recommendation Function

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Function to generate product recommendations using the recommendation system


# # Creating a new dataset with all the pre-processed data of the df dataset 

# In[ ]:


# Specify the file path where you want to save the modified dataset
output_file_path = "/Users/engyamr/Downloads/Python/output_dataset.csv"

# Save the modified dataset to a new CSV file
df.to_csv(output_file_path, index=False)

# Confirm the file has been saved
print(f"Dataset saved to: {output_file_path}")


# In[ ]:


df_output= pd.read_csv("/Users/engyamr/Downloads/Python/output_dataset.csv")
df_output.head()







def get_recommendations(user_id, model, all_product_ids, df, product_names, N=5):
    # Retrieve the user's interactions or purchase history from your dataset
    user_interactions = df[df['User ID'] == user_id]['Product ID'].unique()

    # Filter out products that the user has already interacted with
    uninteracted_products = np.setdiff1d(all_product_ids, user_interactions)

    # Reshape input data if necessary
    user_ids = np.full_like(uninteracted_products, user_id)
    user_ids = user_ids.reshape((-1, 1))  # Assuming user_ids is a 1D array
    uninteracted_products = uninteracted_products.reshape((-1, 1))  # Assuming uninteracted_products is a 1D array

    # Convert arrays to tensors
    user_ids_tensor = tf.convert_to_tensor(user_ids, dtype=tf.float32)
    uninteracted_products_tensor = tf.convert_to_tensor(uninteracted_products, dtype=tf.float32)

    # Predict the interaction probabilities for uninteracted products
    predicted_interactions = model.predict([user_ids_tensor, uninteracted_products_tensor])

    # Combine the predictions with the uninteracted products
    recommendations_df = pd.DataFrame({'User ID': user_ids.flatten(), 'Product ID': uninteracted_products.flatten(), 'Predicted Interaction': predicted_interactions.flatten()})

    # Sort the recommendations by predicted interaction probabilities
    sorted_recommendations = recommendations_df.sort_values(by='Predicted Interaction', ascending=False)

    # Select the top N recommendations
    top_recommendations = sorted_recommendations.head(N).copy()  # Create a copy of the DataFrame

    # Merge with product names
    top_recommendations['Product Name'] = top_recommendations['Product ID'].map(product_names)

    return top_recommendations[['Product Name', 'Predicted Interaction']]