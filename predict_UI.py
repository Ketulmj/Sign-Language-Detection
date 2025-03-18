import time
import numpy as np
import extract_features
import config
import model
from langchain_ollama import ChatOllama
import streamlit as st
from tempfile import NamedTemporaryFile

llm = ChatOllama(
    model="gemma3",
    temperature=0.3
)

class VideoDescriptionRealTime:
    def __init__(self, config):
        self.latent_dim = config.latent_dim
        self.num_encoder_tokens = config.num_encoder_tokens
        self.num_decoder_tokens = config.num_decoder_tokens
        self.time_steps_encoder = config.time_steps_encoder
        self.max_probability = config.max_probability
        self.tokenizer, self.inf_encoder_model, self.inf_decoder_model = model.inference_model()
        self.search_type = config.search_type
        self.num = 0

    def greedy_search(self, loaded_array):
        inv_map = {value: key for key, value in self.tokenizer.word_index.items()}
        states_value = self.inf_encoder_model.predict(loaded_array.reshape(1, 80, 4096))
        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        target_seq[0, 0, self.tokenizer.word_index['bos']] = 1
        
        final_sentence = []
        prev_word = None  # Track previous word to remove consecutive duplicates
        for _ in range(10):  # Reduced iterations for speed
            output_tokens, h, c = self.inf_decoder_model.predict([target_seq] + states_value, batch_size=1)
            states_value = [h, c]
            y_hat = np.argmax(output_tokens[0, 0])
            if y_hat == 0 or inv_map.get(y_hat) in [None, 'eos']:
                break
            current_word = inv_map[y_hat]
            if current_word != prev_word:  # Avoid consecutive duplicates
                final_sentence.append(current_word)
            prev_word = current_word
            target_seq[0, 0, y_hat] = 1
        return ' '.join(final_sentence)

    def process_video(self, video_path):
        model_cnn = extract_features.model_cnn_load()
        features = extract_features.extract_features(video_path, model_cnn)
        return self.greedy_search(features.reshape(1, 80, 4096))

# Streamlit UI
st.title("🎥 Real-Time Video Captioning")
st.write("Upload a video, and the model will generate a caption for it!")
video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if video_file:
    temp_video = NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video.write(video_file.read())
    video_path = temp_video.name
    
    video_captioner = VideoDescriptionRealTime(config)
    with st.spinner("Generating caption..."):
        start_time = time.time()
        caption = video_captioner.process_video(video_path)
        end_time = time.time()
    st.success(f"Caption Generated in {end_time - start_time:.2f} seconds!")
    st.write(f"**Predicted Caption:** {caption}")
    st.video(video_path)
