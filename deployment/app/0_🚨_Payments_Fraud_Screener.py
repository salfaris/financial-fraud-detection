import time

from millify import millify
import pandas as pd
import plotly.express as px
import streamlit as st

import app_config

st.set_page_config(
    page_title="Payments Fraud Screener",
    page_icon="🚨",
    layout="wide",
)

# `inference.py` is a module that loads streamlit as well so `set.set_page_config`
# has to be set up before importing it.
import inference  # noqa

# dashboard title
st.markdown("## 🚨 Payments Fraud Screener")

with st.expander("Want to see a single transaction streaming sim instead?"):
    st.page_link(
        "pages/1_1️⃣_Single_Transactions.py",
        label="Single Transactions Simulator",
        icon="1️⃣",
        use_container_width=True,
    )


# @st.cache_resource
def load_data():
    data = pd.read_csv(
        app_config.DATA_DIR / "02_staged" / "processed_paysim.csv",
        index_col=0,
        chunksize=1000,
    )
    return data


DATA_CHUNK = load_data()


def logreg_pipeline(data: pd.DataFrame):
    model_data = data.drop(
        columns=[
            # "is_fraud",
            "name_Destination",
            "name_Source",
            "step",
        ]
    )
    data = inference.predict_pipeline(model_data)
    # data["flagged_fraud"] = inference.predict_pipeline(model_data)
    # data["confidence"] = MODEL.predict_proba(model_data)[:, 1]
    return data


placeholder = st.empty()

full_data = []
prev_flagged_fraud = 0
prev_fraud_total_amount = 0
for i in range(1000):
    new_data = next(DATA_CHUNK)
    new_data.drop(columns=["is_fraud"], inplace=True)

    new_data = logreg_pipeline(new_data)
    display_data = new_data
    # display_data = data_display_form(new_data)

    full_data.extend([display_data])
    data = pd.concat(full_data)
    # data["flagged_fraud"] = data["flagged_fraud"].astype(str)

    fraudulent_txn_data = data.query("flagged_fraud == 1")
    fraudulent_total_amount = fraudulent_txn_data.amount.sum()

    with placeholder.container():

        # create three columns
        kpi1, kpi2, kpi3 = st.columns(3)

        # fill in those three columns with respective metrics or KPIs
        kpi1.metric(
            label="Number of transactions scanned",
            value=millify(data.shape[0]),
            # delta=-10 + 30,
        )

        flagged_fraud = fraudulent_txn_data.flagged_fraud.sum()
        kpi2.metric(
            label="Flagged frauds 🚨",
            value=millify(flagged_fraud, precision=2),
            delta=int(flagged_fraud - prev_flagged_fraud),
        )

        # FPR estimated from model training requirements.
        estimated_fpr = 0.01
        kpi3.metric(
            label="Estimated loss prevention ＄",
            value="$" + millify(estimated_fpr * fraudulent_total_amount, precision=1),
            # delta=f"{fraud:.0f}",
        )

        # create two columns for charts
        fig_col1, fig_col2 = st.columns(2)
        with fig_col1:
            st.markdown("### Fraud rate split by transaction type")
            st.caption(
                "Percentage of fraudulent transactions grouped by transaction type"
            )

            t1, t2, t3, t4, t5 = st.columns(5)

            txn_split_by_type_count = (
                data.groupby(["type"])["flagged_fraud"]
                .value_counts(normalize=True)
                .unstack(fill_value=0.0)
                .stack()
            )

            def comp(txn_type: str):
                v = txn_split_by_type_count.xs((txn_type, 1))
                st.metric(" ".join(txn_type.split("_")), value=f"{v* 100:.1f}%")

            with t1:
                comp("TRANSFER")
            with t2:
                comp("CASH_OUT")
            with t3:
                comp("CASH_IN")
            with t4:
                comp("DEBIT")
            with t5:
                comp("PAYMENT")
            # fig = px.scatter(
            #     data_frame=data,
            #     x="step",
            #     y="amount",
            #     # color=data["flagged_fraud"].astype(str),
            #     color=data["type"],
            #     symbol=data["flagged_fraud"],
            #     color_discrete_sequence=px.colors.qualitative.Plotly,
            # )
            # fig = px.density_heatmap(
            #     data_frame=data,
            #     x="type",
            #     y=data["flagged_fraud"].map(
            #         {1: "Flagged fraud 🚨", 0: "Not flagged fraud"}
            #     ),
            #     z="amount",
            #     histfunc="count",
            #     histnorm="percent",
            #     # # color=data["flagged_fraud"].astype(str),
            #     # color=data["type"],
            #     # symbol=data["flagged_fraud"],
            #     # color_discrete_sequence=px.colors.qualitative.Plotly,
            # )
            # st.write(fig)

        # with fig_col2:
        #     st.markdown("### Second Chart")
        #     fig2 = px.histogram(
        #         data_frame=fraudulent_txn_data, x="new_balance_Destination"
        #     )
        #     st.write(fig2)

        with fig_col2:
            st.markdown("### Fraud transaction amount")
            st.caption("Transaction amount for transactions marked as fraud.")
            fig2 = px.histogram(data_frame=fraudulent_txn_data, x="amount")

            # fig2 = go.Figure(
            #     data=[
            #         go.Histogram(
            #             x=fraudulent_txn_data["amount"],
            #             marker=go.histogram.Marker(color="orange"),
            #             name="Fraud",
            #         )
            #     ],
            # )
            # fig2.add_trace(
            #     go.Histogram(
            #         x=data.query("flagged_fraud == 0")["amount"], name="Not fraud"
            #     )
            # )
            # fig2.update_layout(barmode="overlay")
            # fig2.update_traces(opacity=0.75)
            st.write(fig2)

        # st.markdown("### Data streamed")
        # st.dataframe(data)

        st.markdown("### Fraudulent transactions")
        st.dataframe(data.query("flagged_fraud == 1"))

        time.sleep(0.1)

        prev_flagged_fraud = flagged_fraud
        prev_fraud_total_amount = fraudulent_total_amount
