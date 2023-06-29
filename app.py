import streamlit as st
import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

# Load the fine-tuned GPT-2 model and tokenizer
model_path = "tf_model.h5"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = TFGPT2LMHeadModel.from_pretrained(model_path)

# Set the maximum length for generated responses
max_length = 100

# Define a function to generate responses for user queries
def generate_response(query):
    input_ids = tokenizer.encode(query, add_special_tokens=True, return_tensors="tf")
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Streamlit app
def main():
    st.title("Medical Query Assistant")
    query = st.text_input("Enter your medical query")
    if st.button("Submit"):
        if query.strip() == "":
            st.error("Please enter a valid query.")
        else:
            response = generate_response(query)
            st.success("Response:")
            st.write(response)

if __name__ == "__main__":
    main()
