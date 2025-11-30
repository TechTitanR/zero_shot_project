# streamlit_app.py
import streamlit as st
from transformers import pipeline
import pandas as pd
import io

# Cache the model to avoid reloading every run
@st.cache_resource
def get_classifier(model_name="facebook/bart-large-mnli"):
    return pipeline("zero-shot-classification", model=model_name)

# Web app settings
st.set_page_config(page_title="Zero-Shot Classifier", layout="centered")
st.title("🧠 Zero-Shot Text Classification Demo")
st.write("Classify any text into custom labels **without training** using a pretrained Transformer model.")

# Model input
with st.expander("⚙️ Model Settings", expanded=False):
    model_name = st.text_input("Model name", value="facebook/bart-large-mnli")
classifier = get_classifier(model_name)

# User input area
text = st.text_area("✏️ Enter text to classify:", 
                    value="Apple announced a new iPhone with improved battery life.")

labels_input = st.text_input("🏷️ Candidate labels (comma-separated):", 
                             value="technology, sports, politics, food")

labels = [l.strip() for l in labels_input.split(",") if l.strip()]

# Classify Button
if st.button("🚀 Classify"):
    if not text.strip():
        st.error("Please enter text.")
    elif not labels:
        st.error("Please provide labels.")
    else:
        with st.spinner("Running model..."):
            result = classifier(text, candidate_labels=labels)

        st.success("Classification complete!")

        # Extract results
        top_label = result["labels"][0]
        scores = {lab: float(score) for lab, score in zip(result["labels"], result["scores"])}

        st.subheader("🏆 Top Prediction:")
        st.markdown(f"### **{top_label}**")

        # Create DataFrame for display + visualization
        df = pd.DataFrame({
            "Label": list(scores.keys()),
            "Score": [round(s, 4) for s in scores.values()]
        }).sort_values("Score", ascending=False)

        # Display Table
        st.subheader("📊 Score Table")
        st.dataframe(df, use_container_width=True)

        # Display Bar Chart
        st.subheader("📈 Confidence Bar Chart")
        st.bar_chart(df.set_index("Label"))

        # Download as CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)

        st.download_button(
            label="💾 Download results as CSV",
            data=csv_buffer.getvalue(),
            file_name="zero_shot_result.csv",
            mime="text/csv",
        )

# Small footer
st.markdown("---")
st.caption("Built using Hugging Face Transformers + Streamlit")
