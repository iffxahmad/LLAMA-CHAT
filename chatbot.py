pip install transformers
import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Function to generate responses
def generate_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Streamlit UI
st.title("GPT-2 Chatbot")

# Create a text input for the user
user_input = st.text_input("You: ", "Ask Question?")

# Generate and display the response
if st.button("Send"):
    response = generate_response(user_input)
    st.text_area("Bot:", value=response, height=200)
