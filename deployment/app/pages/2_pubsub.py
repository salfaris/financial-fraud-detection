from concurrent import futures
import json

from google.cloud import pubsub_v1
import streamlit as st

# import pandas as pd
# import joblib


PROJECT_ID = "fraud-detection-10062930"
TOPIC_ID = "single-transactions"
SUBSCRIPTION_ID = "single-transactions-subscription"

# Google Cloud Pub/Sub setup
subscriber = pubsub_v1.SubscriberClient()
subscription_path = f"projects/{PROJECT_ID}/subscriptions/{SUBSCRIPTION_ID}"


# Streamlit app
st.title("Real-time Transaction Publisher")

# # Initialize a Pub/Sub client
# publisher = pubsub_v1.PublisherClient()

# # Create a fully qualified identifier of the topic
# topic_path = publisher.topic_path(PROJECT_ID, TOPIC_ID)


# # def publish_message(data):
# #     data = json.dumps(data).encode("utf-8")
# #     future = publisher.publish(topic_path, data)
# #     return future.result()


# # st.header("Publish a Transaction")

# # # Create a form for transaction data
# # with st.form(key="transaction_form"):
# #     transaction_id = st.text_input("Transaction ID")
# #     amount = st.number_input("Amount", min_value=0.0)
# #     currency = st.text_input("Currency")
# #     timestamp = st.text_input("Timestamp", placeholder="2024-07-20T12:00:00Z")

# #     submit_button = st.form_submit_button(label="Publish")

# # if submit_button:
# #     transaction_data = {
# #         "transaction_id": transaction_id,
# #         "amount": amount,
# #         "currency": currency,
# #         "timestamp": timestamp,
# #     }

# #     message_id = publish_message(transaction_data)
# #     st.success(f"Published message ID: {message_id}")


st.title("Real-time Transaction Prediction")


# # Function to handle incoming messages
# def callback(message):
#     # data = json.loads(message.data)
#     # transaction_df = pd.DataFrame([data])

#     # # Make predictions
#     # prediction = model.predict(transaction_df)
#     # st.write(f"Prediction: {prediction}")

#     # Acknowledge the message
#     st.write(f"Received message: {message.data}")
#     message.ack()


# # Subscribe to the Pub/Sub topic
# streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
# st.write(streaming_pull_future)
# st.write("Listening for transactions...")

# # Keep the Streamlit app running
# try:
#     streaming_pull_future.result()
#     st.write("HELLO WORLD")
# except KeyboardInterrupt:
#     streaming_pull_future.cancel()
