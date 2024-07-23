#!/bin/bash

TOPIC_ID="single-transactions"
gcloud pubsub topics create $TOPIC_ID

# Create a pull subscription
SUBSCRIPTION_ID="single-transactions-subscription"
gcloud pubsub subscriptions create $SUBSCRIPTION_ID --topic=$TOPIC_ID
