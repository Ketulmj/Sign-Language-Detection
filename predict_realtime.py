import os
import time
import cv2
import numpy as np
from langchain_ollama import ChatOllama
import config
import extract_features
import model

llm = ChatOllama(
    model="gemma3",
    temperature=0.3
)

class VideoDescriptionRealTime(object):
    """Initialize the parameters for the model."""

    def __init__(self, config):
        self.latent_dim = config.latent_dim
        self.num_encoder_tokens = config.num_encoder_tokens
        self.num_decoder_tokens = config.num_decoder_tokens
        self.time_steps_encoder = config.time_steps_encoder
        self.max_probability = config.max_probability
        # models
        self.tokenizer, self.inf_encoder_model, self.inf_decoder_model = model.inference_model()
        self.inf_decoder_model = None
        self.save_model_path = config.save_model_path
        self.test_path = config.test_path
        self.train_path = config.train_path
        self.search_type = config.search_type
        self.num = 0

    def greedy_search(self, loaded_array):
        """Predict sentence using greedy search approach.

        Args:
            loaded_array: The loaded numpy array after creating videos to frames and extracting features.

        Returns:
            The final sentence which has been predicted greedily.
        """
        inv_map = self.index_to_word()
        states_value = self.inf_encoder_model.predict(loaded_array.reshape(-1, 80, 4096))
        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        final_sentence = ''
        target_seq[0, 0, self.tokenizer.word_index['bos']] = 1
        self.tokenizer, self.inf_encoder_model, self.inf_decoder_model = model.inference_model()

        for i in range(15):
            output_tokens, h, c = self.inf_decoder_model.predict([target_seq] + states_value)
            states_value = [h, c]
            output_tokens = output_tokens.reshape(self.num_decoder_tokens)
            y_hat = np.argmax(output_tokens)

            if y_hat == 0:
                continue
            if inv_map[y_hat] is None or inv_map[y_hat] == 'eos':
                break

            final_sentence = final_sentence + inv_map[y_hat] + ' '
            target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            target_seq[0, 0, y_hat] = 1
        return final_sentence

    def decode_sequence2bs(self, input_seq):
        """Decode sequence using beam search."""
        states_value = self.inf_encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        target_seq[0, 0, self.tokenizer.word_index['bos']] = 1
        self.beam_search(target_seq, states_value, [], [], 0)
        return decode_seq

    def decoded_sentence_tuning(self, decoded_sentence):
        """Tune the decoded sentence by removing duplicates and special tokens."""
        decode_str = []
        filter_string = ['bos', 'eos']
        uni_gram = {}
        last_string = ""

        for idx2, c in enumerate(decoded_sentence):
            if c in uni_gram:
                uni_gram[c] += 1
            else:
                uni_gram[c] = 1

            if last_string == c and idx2 > 0:
                continue
            if c in filter_string:
                continue
            if len(c) > 0:
                decode_str.append(c)
            if idx2 > 0:
                last_string = c

        return decode_str

    def index_to_word(self):
        """Convert index to word mapping."""
        return {value: key for key, value in self.tokenizer.word_index.items()}

    def get_test_data(self, file_name):
        """Load and process test data."""
        path = os.path.join(self.train_path, 'feat', file_name + '.npy')
        if not os.path.exists(path):
            path = os.path.join(self.test_path, 'feat', file_name + '.npy')
        if os.path.exists(path):
            f = np.load(path)
        else:
            model = extract_features.model_cnn_load()
            f = extract_features.extract_features(file_name, model)

        return f, file_name

    def test(self, file_name):
        """Run inference on test data."""
        X_test, filename = self.get_test_data(file_name)

        if self.search_type == 'greedy':
            sentence_predicted = self.greedy_search(X_test.reshape((-1, 80, 4096)))
        else:
            sentence_predicted = ''
            decoded_sentence = self.decode_sequence2bs(X_test.reshape((-1, 80, 4096)))
            decode_str = self.decoded_sentence_tuning(decoded_sentence)
            sentence_predicted = ' '.join(decode_str)

        self.max_probability = -1
        return sentence_predicted, filename

    def main(self, filename, caption):
        """Display video with caption."""
        video_path = os.path.join(self.train_path, 'video', filename)
        if not os.path.exists(video_path):
            video_path = os.path.join(self.test_path, 'video', filename)

        # print(f"Full video path: {os.path.abspath(video_path)}")
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {filename}")
            return

        caption = '[ ' + ' '.join(caption.split()[:]) + ']'
        cv2.namedWindow('Video Caption', cv2.WINDOW_NORMAL)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Reached end of video")
                break
            # Resize and add caption
            frame = cv2.resize(frame, (800, 600))
            # Get text size to calculate center position
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2
            text_size = cv2.getTextSize(caption, font, font_scale, thickness)[0]
            
            # Calculate position for bottom-center alignment
            text_x = (frame.shape[1] - text_size[0]) // 2  # Center horizontally
            text_y = frame.shape[0] - 30  # Bottom with 30px padding
            
            cv2.putText(frame, caption, (text_x, text_y), 
                        font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
            cv2.imshow('Video Caption', frame)

            if cv2.waitKey(25) & 0xFF in [27, ord('q')]:  # ESC or Q key
                break

        cap.release()
        cv2.waitKey(1000)  # Wait 2 seconds before closing
        cv2.destroyAllWindows()


if __name__ == "__main__":
    video_to_text = VideoDescriptionRealTime(config)
    file_name = ["thank you so much (6).MP4", "i am hungry   (5).MP4", "congratulations (7).MP4", "agree.MP4", "who are you (6).MP4"]
    for file in file_name:
        print('\n.........................\n')
        print(f"Generating Caption for {file} :\n")
        start = time.time()
        video_caption, video_file = video_to_text.test(file)
        end = time.time()
        sentence = ' '.join(video_caption.split())
        print('\n.........................\n')
        print(f"Predicted Sentence: {sentence}")
        print('\n.........................')
        print('It took {:.2f} seconds to generate caption'.format(end-start))
        video_to_text.main(video_file, sentence)
        print('\n.........................')
        # print("Structured Sentence using LLM....\n", llm.invoke(f"""Take the given response: '{sentence}' and rephrase it into a natural, human-like sentence while maintaining its original meaning and clarity. Only respond to well-formed sentences like 'I want a relationship built on love and respect.""").content)
        print("Do you want to play next video? (y/n)")
        a = input()
        if a == 'y':
            continue
        else:
            break