import tensorflow as tf
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

import tensorflow as tf
import numpy as np
import logging
from PIL import Image

# Configurer le logger
logging.basicConfig(
    level=logging.DEBUG,  # ou logging.DEBUG
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


@st.cache_resource
def load_model():
   
    model = tf.keras.models.load_model("model/best_efficientnetb0.keras",compile=False)
    
    return EfficientNetWrapper(model)

@st.cache_resource
def load_tflite_dyna_model():
    model =TFLiteDynamicModel("model/efficientnetb0_fp32.tflite", img_size=224)
    return model

@st.cache_resource
def load_tflite_ptq_model():
    model =TFLiteDynamicModel("model/efficientnetb0_ptq_int8.tflite", img_size=224)
    return model



def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((480, 480))
    array = np.array(image) / 255.0
    return np.expand_dims(array, axis=0)

def predict(model, image: Image.Image):
    input_array = preprocess_image(image)
    predictions = model.predict(input_array)
    return np.argmax(predictions[0])



class EfficientNetWrapper:
    def __init__(self, model):
        self.model = model
        self.img_size = 224  # taille attendue par EfficientNetV2
        print("🔧 [INIT] EfficientNetWrapper initialisé avec modèle et taille d’image 224x224.")

    def get_model(self):
        print("📦 [GET MODEL] Récupération du modèle encapsulé.")
        return self.model

    def preprocess(self, pil_img):
        print("🖼️ [PREPROCESS] Début du prétraitement de l’image...")
        # Redimensionne
        img_resized = pil_img.resize((self.img_size, self.img_size))
        print("📐 [PREPROCESS] Image redimensionnée à", self.img_size, "x", self.img_size)

        # Convertit en array numpy
        x = image.img_to_array(img_resized)
        print("🔄 [PREPROCESS] Image convertie en tableau numpy.")

        # Ajoute une dimension batch
        x = np.expand_dims(x, axis=0)
        print("➕ [PREPROCESS] Ajout de la dimension batch :", x.shape)

        # Preprocessing EfficientNet
        x = tf.keras.applications.efficientnet.preprocess_input(x)
        print("✅ [PREPROCESS] Normalisation effectuée selon EfficientNet.")

        return x

    def predict(self, pil_img):
        print("🔮 [PREDICT] Prédiction en cours...")
        x = self.preprocess(pil_img)
        preds = self.model.predict(x)
        print("📊 [PREDICT] Prédiction terminée. Forme des prédictions :", preds.shape)
        return preds[0], x


class TFLiteDynamicModel:
    def __init__(self, tflite_path, img_size=224):
        logger.info(f"🚀 Chargement du modèle TFLite depuis : {tflite_path}")
        self.img_size = img_size
        self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.interpreter.allocate_tensors()

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        self.input_index = input_details[0]['index']
        self.input_dtype = input_details[0]['dtype']
        self.input_scale, self.input_zero_point = input_details[0]['quantization']
        self.output_index = output_details[0]['index']

        logger.info(f"🔍 Input tensor index : {self.input_index}, dtype : {self.input_dtype}, scale : {self.input_scale}, zero_point : {self.input_zero_point}")
        logger.info(f"🔍 Output tensor index : {self.output_index}")


    def preprocess(self, pil_image):
        logger.info(f"🎨 Prétraitement image, redimension à {self.img_size}x{self.img_size}")
        img = pil_image.resize((self.img_size, self.img_size))
        img = np.array(img)

        # 📸 Gestion des images grayscale ou RGBA
        if img.ndim == 2:  # Grayscale -> RGB
            logger.debug("⚪ Image grayscale détectée, conversion en RGB")
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[-1] == 4:  # RGBA -> RGB
            logger.debug("🖼️ Image RGBA détectée, suppression canal alpha")
            img = img[..., :3]

        if self.input_dtype in [np.uint8, np.int8]:
            logger.info("🗜️ Modèle quantifié PTQ détecté (entrée int8 ou uint8)")

            # Pas de division par 255 ici ! On quantifie directement selon l'échelle et le zéro-point
            img = img.astype(np.float32)
            img = img / self.input_scale + self.input_zero_point

            # On clip selon le type
            if self.input_dtype == np.uint8:
                img = np.clip(img, 0, 255)
            else:  # np.int8
                img = np.clip(img, -128, 127)

            img = img.astype(self.input_dtype)

        else:
            logger.info("🌊 Modèle dynamique ou float32 détecté (entrée float32 normalisée)")
            img = img.astype(self.input_dtype)  # Normalisation classique

        input_data = np.expand_dims(img, axis=0)
        logger.info(f"✅ Image prétraitée avec forme {input_data.shape} et dtype {input_data.dtype}")
        return input_data


    def preprocess_old(self, pil_image):
        logger.info(f"🎨 Prétraitement image, redimension à {self.img_size}x{self.img_size}")
        img = pil_image.resize((self.img_size, self.img_size))
        img = np.array(img).astype(np.float32)

        if img.ndim == 2:  # grayscale -> RGB
            logger.debug("⚪ Image grayscale détectée, conversion en RGB")
            img = np.stack([img]*3, axis=-1)
        elif img.shape[-1] == 4:  # RGBA -> RGB
            logger.debug("🖼️ Image RGBA détectée, suppression canal alpha")
            img = img[..., :3]

        if self.input_dtype in [np.uint8, np.int8]:
            if self.input_scale > 0:
                logger.debug(f"⚙️ Application quantification dynamique avec scale {self.input_scale} et zero_point {self.input_zero_point}")
                img = img / 255.0
                img = img / self.input_scale + self.input_zero_point
                img = np.clip(img, 0, 255 if self.input_dtype == np.uint8 else 127)
            img = img.astype(self.input_dtype)
        else:
            img = img.astype(self.input_dtype)

        input_data = np.expand_dims(img, axis=0)
        logger.info(f"✅ Image prétraitée avec forme {input_data.shape} et dtype {input_data.dtype}")
        return input_data

    

    def predict_dyna(self, pil_image):
        logger.info("⚡ Début de prédiction (modèle dynamique ou float32)")

        # Prétraitement
        logger.info("🔄 Prétraitement de l'image en cours")
        input_data = self.preprocess(pil_image)
        logger.debug(f"✅ Image prétraitée - Shape : {input_data.shape} - Dtype : {input_data.dtype}")

        # Injection des données dans le modèle
        logger.info("📥 Injection des données dans le modèle")
        self.interpreter.set_tensor(self.input_index, input_data)

        # Invocation du modèle
        logger.info("🚀 Exécution du modèle TFLite")
        self.interpreter.invoke()

        # Récupération de la sortie
        logger.info("📤 Récupération des résultats bruts")
        output_data = self.interpreter.get_tensor(self.output_index)
        logger.debug(f"✅ Logits récupérés - Shape : {output_data.shape} - Dtype : {output_data.dtype}")

        # Calcul des probabilités
        logger.info("🧮 Calcul des probabilités")
        probas=output_data[0]
        logger.debug(f"✅ Probabilités : {probas}")

        # # Classe prédite
        # predicted_class = np.argmax(probas)
        # confidence = probas[predicted_class]
        # logger.info(f"🏷️ Classe prédite : {predicted_class} avec une probabilité de {confidence:.4f}")

        logger.info("🎯 Prédiction terminée")

        return  probas


    def predict_ptq(self, pil_image):
        logger.info("⚡ Début de prédiction")
        
        # Prétraitement
        logger.info("🔄 Prétraitement de l'image en cours")
        input_data = self.preprocess(pil_image)
        logger.debug(f"✅ Image prétraitée - Shape : {input_data.shape} - Dtype : {input_data.dtype}")

        # Injection des données dans le modèle
        logger.info("📥 Injection des données dans le modèle")
        self.interpreter.set_tensor(self.input_index, input_data)

        # Invocation du modèle
        logger.info("🚀 Exécution du modèle TFLite")
        self.interpreter.invoke()

        # Récupération de la sortie
        logger.info("📤 Récupération des résultats bruts")
        output_details = self.interpreter.get_output_details()[0]
        output_data = self.interpreter.get_tensor(output_details['index'])
        logger.debug(f"✅ Logits quantifiés récupérés - Shape : {output_data.shape} - Dtype : {output_data.dtype}")

        # Paramètres de quantification
        output_scale, output_zero_point = output_details['quantization']
        logger.debug(f"ℹ️ Paramètres de quantification - Scale: {output_scale}, Zero Point: {output_zero_point}")

        # Déquantification
        logger.info("🔓 Déquantification des logits")
        logits = (output_data.astype(np.float32) - output_zero_point) * output_scale
        logger.debug(f"✅ Logits déquantifiés : {logits}")

        # Calcul des probabilités
        logger.info("🧮 Calcul des probabilités avec softmax")
        #probas = tf.nn.softmax(logits[0]/temperature).numpy()
        probas = logits[0]
        logger.debug(f"✅ Probabilités : {probas}")

        logger.info("🎯 Prédiction terminée")

        return probas
    
    def predict (self, pil_image):
        if self.input_dtype in [np.uint8, np.int8]:
            logger.info("🗜️ Modèle quantifié PTQ détecté")
            return self.predict_ptq(pil_image)
        else:
            logger.info("🌊 Modèle dynamique ou float32 détecté")
            return self.predict_dyna(pil_image)