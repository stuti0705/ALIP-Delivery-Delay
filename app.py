import streamlit as st
import pandas as pd
import joblib

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("delivery_delay_model.pkl")

model = load_model()

st.set_page_config(
    page_title="Delivery Delay Prediction",
    page_icon="📦",
    layout="centered"
)

st.title("📦 Delivery Delay Prediction")
st.write("Predict whether an order is likely to be delayed.")

st.subheader("Enter Order Details")

with st.form("prediction_form"):
    order_type = st.selectbox(
        "Order Type",
        ["DEBIT", "TRANSFER", "PAYMENT", "CASH"]
    )

    category = st.text_input("Category Name", "Cleats")
    
    customer_segment = st.selectbox(
        "Customer Segment",
        ["Consumer", "Corporate", "Home Office"]
    )

    region = st.text_input("Order Region", "West")
    state = st.text_input("Order State", "California")

    shipping_mode = st.selectbox(
        "Shipping Mode",
        ["Standard Class", "Second Class", "First Class", "Same Day"]
    )

    days_scheduled = st.number_input(
        "Days for Shipment (Scheduled)",
        min_value=0,
        value=4
    )

    quantity = st.number_input(
        "Order Item Quantity",
        min_value=1,
        value=1
    )

    price = st.number_input(
        "Order Item Product Price",
        min_value=0.0,
        value=100.0
    )

    discount = st.number_input(
        "Order Item Discount Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.0
    )

    submitted = st.form_submit_button("Predict Delivery Status")

if submitted:
    input_data = pd.DataFrame([{
        "Type": order_type,
        "Category Name": category,
        "Customer Segment": customer_segment,
        "Order Region": region,
        "Order State": state,
        "Shipping Mode": shipping_mode,
        "Days for shipment (scheduled)": days_scheduled,
        "Order Item Quantity": quantity,
        "Order Item Product Price": price,
        "Order Item Discount Rate": discount
    }])

    try:
        probability = model.predict_proba(input_data)[0][1]
        prediction = int(probability >= 0.40)
       
        st.subheader("Order Risk Assessment")

        st.metric(
            "Estimated Delay Risk",
            f"{probability:.1%}"
        )

        if probability >= 0.40:
            st.warning(
                "This order has a higher predicted risk of delay. "
                "Consider reviewing the shipment plan."
            )
        else:
            st.success(
                "This order has a lower predicted risk of delay."
            )

        st.caption(
            "Risk estimates are model predictions and are not guarantees."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
