from transformers import AutoModelForCausalLM, AutoTokenizer
import streamlit as st
import os

# Tải mô hình GPT-J
model_name = "EleutherAI/gpt-j-6B"  # Hoặc thay bằng "bigscience/bloom-560m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

st.title("Chatbot GPT-J với Ngữ Cảnh")

# Lưu trữ lịch sử hội thoại
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = ""

# Đầu vào từ người dùng
user_input = st.text_input("Hãy nhập câu hỏi của bạn:")

if user_input:
    # Thêm câu hỏi vào lịch sử hội thoại
    st.session_state["chat_history"] += f"Người dùng: {user_input}\n"

    # Tạo prompt với lịch sử hội thoại
    inputs = tokenizer(st.session_state["chat_history"], return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=500, pad_token_id=tokenizer.eos_token_id)

    # Phản hồi từ bot
    bot_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    bot_response = bot_response[len(st.session_state["chat_history"]):].strip()

    # Thêm phản hồi vào lịch sử hội thoại
    st.session_state["chat_history"] += f"Chatbot: {bot_response}\n"

    # Hiển thị phản hồi
    st.write(f"**Chatbot:** {bot_response}")
