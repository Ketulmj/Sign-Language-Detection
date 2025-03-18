import shutil
import tqdm
import numpy as np
import cv2
import os
import mediapipe as mp
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
import config
import tensorflow as tf

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def video_to_frames(video):
    path = os.path.join(config.train_path, 'temporary_images')
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    video_path = os.path.join(config.train_path, 'video', video)
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    count = 0
    image_list = []
    # Path to video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")
    
    # Initialize MediaPipe hands
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is False:
                break

            # Convert BGR to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame and detect hands
            results = hands.process(frame_rgb)
            
            # Draw hand landmarks on the frame
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
            
            # Save the frame with hand landmarks
            cv2.imwrite(os.path.join(config.train_path, 'temporary_images', f'frame{count}.jpg'), frame)
            image_list.append(os.path.join(config.train_path, 'temporary_images', f'frame{count}.jpg'))
            count += 1
    cap.release()
    return image_list

def model_cnn_load():
    model = VGG16(weights="imagenet", include_top=True, input_shape=(224, 224, 3))
    out = model.layers[-2].output
    model_final = Model(inputs=model.input, outputs=out)
    return model_final

def load_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    return img

def extract_features(video, model):
    """
    :param video: The video whose frames are to be extracted to convert into a numpy array
    :param model: the pretrained vgg16 model
    :return: numpy array of size 4096x80
    """
    video_id = video.split(".")[0]
    print(video_id)
    print(f'Processing video {video}')

    try:
        image_list = video_to_frames(video)
        samples = np.round(np.linspace(0, len(image_list) - 1, 80))
        image_list = [image_list[int(sample)] for sample in samples]
        
        # Process images in smaller batches
        batch_size = 16  # Reduced batch size to prevent OOM
        num_images = len(image_list)
        img_feats_list = []

        for i in range(0, num_images, batch_size):
            batch_images = np.zeros((min(batch_size, num_images - i), 224, 224, 3))
            
            # Load and preprocess batch
            for j, idx in enumerate(range(i, min(i + batch_size, num_images))):
                img = load_image(image_list[idx])
                batch_images[j] = img

            # Clear memory
            tf.keras.backend.clear_session()
            
            # Process batch
            batch_feats = model.predict(batch_images, batch_size=batch_size)
            img_feats_list.append(batch_feats)

        # Combine results
        img_feats = np.concatenate(img_feats_list, axis=0)
        
    except Exception as e:
        print(f"Error processing video {video}: {str(e)}")
        raise
    finally:
        # Cleanup
        if os.path.exists(os.path.join(config.train_path, 'temporary_images')):
            shutil.rmtree(os.path.join(config.train_path, 'temporary_images'))           
    return img_feats


def extract_feats_pretrained_cnn():
    """
    saves the numpy features from all the videos
    """
    model = model_cnn_load()
    print('Model loaded')

    if not os.path.isdir(os.path.join(config.train_path, 'mediapipe_feat')):
        os.mkdir(os.path.join(config.train_path, 'mediapipe_feat'))

    video_list = os.listdir(os.path.join(config.train_path, 'video'))
    
    #ًWhen running the script on Colab an item called '.ipynb_checkpoints' 
    #is added to the beginning of the list causing errors later on, so the next line removes it.
    # video_list.remove('.ipynb_checkpoints')
    
    for video in video_list:
        outfile = os.path.join(config.train_path, 'mediapipe_feat', video + '.npy')
        img_feats = extract_features(video, model)
        np.save(outfile, img_feats)

if __name__ == "__main__":
    extract_feats_pretrained_cnn()