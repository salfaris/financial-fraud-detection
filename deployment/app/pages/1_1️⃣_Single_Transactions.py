import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Single Streaming Transaction Simulator",
    page_icon="🤖",
    layout="wide",
)

# `inference.py` is a module that loads streamlit as well so `set.set_page_config`
# has to be set up before importing it.
import inference  # noqa


def init_session_state(key, default=None):
    if key not in st.session_state:
        st.session_state[key] = default


init_session_state("fraud_dataframe", default=pd.DataFrame())
init_session_state("generate_counter", default=0)

init_session_state("amount")
init_session_state("source_ob")
init_session_state("source_nb")
init_session_state("dest_ob")
init_session_state("dest_nb")
init_session_state("txn_type")
init_session_state("txn_type_index", default=0)

init_session_state("random_amount")
init_session_state("random_source_ob")
init_session_state("random_source_nb")
init_session_state("random_dest_ob")
init_session_state("random_dest_nb")
init_session_state("random_txn_type")
init_session_state("random_txn_type_index", default=0)


TRANSACTION_TYPES = [
    "TRANSFER",
    "CASH_OUT",
    "CASH_IN",
    "DEBIT",
    "PAYMENT",
]
UX_TXN_TYPES = {k: k.replace("_", " ").title() for k in TRANSACTION_TYPES}
UX_TXN_TYPES_INVERSE = {v: k for k, v in UX_TXN_TYPES.items()}

st.markdown("## 1️⃣ Single Streaming Transaction Simulator")
st.write(
    """
This demo simulates how the service process a single transaction in the backend.
"""
)

with st.expander("Want to see a full streaming simulator instead?"):
    st.page_link(
        "0_🚨_Payments_Fraud_Screener.py",
        label="Payments Fraud Screener",
        icon="🚨",
        use_container_width=True,
    )

# with st.expander("Definition: transaction data"):
st.caption(
    """
Definition. A transaction of type ***T*** and amount ***A*** flows from a :blue[SOURCE] account to a :red[DESTINATION] account.
Depending on the type ***T***, this transaction can increase or decrease the balance of the :blue[SOURCE] account by ***A***; and similarly for the :red[DESTINATION] account.
"""  # noqa
)


def generate_choice() -> np.ndarray:
    return np.random.choice(TRANSACTION_TYPES)


def generate_amount() -> np.ndarray:
    sample = np.random.exponential(scale=100000, size=1)
    return sample[0]  # okay to get 0-index element because `size=1`.


min_amount = 0.0
step_amount = 1000.0

container = st.empty()
with container.container(border=True):
    st.markdown("#### Transaction data")

    generate = st.button("Give me a random transaction data")

    if generate:
        st.session_state.generate_counter += 1
        st.session_state.random_amount = generate_amount()
        st.session_state.random_source_ob = generate_amount()
        st.session_state.random_source_nb = generate_amount()
        st.session_state.random_dest_ob = generate_amount()
        st.session_state.random_dest_nb = generate_amount()
        st.session_state.random_txn_type = generate_choice()
        st.session_state.random_txn_type_index = list(UX_TXN_TYPES.keys()).index(
            st.session_state.random_txn_type
        )

    st.session_state.amount = st.number_input(
        "Transaction amount",
        value=st.session_state.random_amount,
        min_value=min_amount,
        step=step_amount,
        placeholder="Put the amount of transaction",
    )

    source_ob_col, source_nb_col = st.columns(2)
    with source_ob_col:
        st.session_state.source_ob = st.number_input(
            "Old balance in :blue[SOURCE] ➡️ account",
            value=st.session_state.random_source_ob,
            min_value=min_amount,
            step=step_amount,
            placeholder="Put the amount in the SOURCE account before transaction",
        )
    with source_nb_col:
        st.session_state.source_nb = st.number_input(
            "New balance in :blue[SOURCE] ➡️ account",
            value=st.session_state.random_source_nb,
            min_value=min_amount,
            step=step_amount,
            placeholder="Put the amount in the SOURCE account after transaction",
        )

    dest_ob_col, dest_nb_col = st.columns(2)
    with dest_ob_col:
        st.session_state.dest_ob = st.number_input(
            "Old balance in :red[DESTINATION] ⬅️ account",
            value=st.session_state.random_dest_ob,
            min_value=min_amount,
            step=step_amount,
            placeholder="Put the amount in the DESTINATION account before transaction",
        )
    with dest_nb_col:
        st.session_state.dest_nb = st.number_input(
            "New balance in :red[DESTINATION] ⬅️ account",
            value=st.session_state.random_dest_nb,
            min_value=min_amount,
            step=step_amount,
            placeholder="Put the amount in the DESTINATION account after transaction",
        )
    st.session_state.txn_type = st.selectbox(
        "Transaction type",
        index=st.session_state.random_txn_type_index,
        options=UX_TXN_TYPES.values(),
        placeholder="Choose a transaction type",
    )
    st.session_state.txn_type_index = list(UX_TXN_TYPES_INVERSE.keys()).index(
        st.session_state.txn_type
    )

    st.markdown("#### Is transaction a potential fraud?")

    def fire():
        message = {
            "amount": st.session_state.amount,
            "old_balance_Source": st.session_state.source_ob,
            "new_balance_Source": st.session_state.source_nb,
            "old_balance_Destination": st.session_state.dest_ob,
            "new_balance_Destination": st.session_state.dest_nb,
            "type": UX_TXN_TYPES_INVERSE[st.session_state.txn_type],
        }
        # st.write(message)
        found_one_null = False
        for k, v in message.items():
            if v is None:
                found_one_null = True

        if found_one_null:
            st.write("Cannot predict fraud with no data")

        else:
            # Convert message JSON into ML model input format.
            batch = pd.Series(message).to_frame().T

            batch = inference.predict_pipeline(batch)
            batch.set_index(pd.Index([st.session_state.generate_counter]), inplace=True)
            pred = batch.flagged_fraud.iloc[0]

            if pred == 1:
                st.session_state.fraud_dataframe = pd.concat(
                    [st.session_state.fraud_dataframe, batch]
                )
            st.write(
                "🚨 YES, transaction is possibly fraud!"
                if pred == 1
                else "🟢 NO, unlikely transaction is fraud"
            )

    fire()

st.markdown("### 🚨 Potential fraud activities")

df = st.session_state.fraud_dataframe
if df.empty:
    st.write(
        "No fraud activity detected yet, keep clicking the generate "
        "random data button above."
    )
else:
    st.dataframe(df)
