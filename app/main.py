import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model import load_model, predict,load_tflite_dyna_model,load_tflite_ptq_model
import cv2
import streamlit as st
from PIL import Image
import numpy as np
import os
import requests
#from tensorflow.keras.applications.efficientnet import  preprocess_input
from tensorflow.keras.preprocessing import image
import time
from io import BytesIO
import base64
import tensorflow as tf

from streamlit_autorefresh import st_autorefresh

import logging

logging.basicConfig(
    level=logging.INFO,  # ou logging.DEBUG
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger(__name__)
model_float32= load_model()
model_dynamique=load_tflite_dyna_model()
model_ptq=load_tflite_ptq_model()
API_HEALTH = "http://0.0.0.0:8590/health"
API_URL = "http://0.0.0.0:8590/predict"
class_names = [
    'Apple_Apple_scab',
    'Apple_Black_rot',
    'Apple_Cedar_apple_rust',
    'Apple_Healthy',
    'Blueberry_Healthy',
    'Cassava_Healthy',
    'Cassava_bacterial_blight',
    'Cassava_brown_streak_disease',
    'Cassava_green_mottle',
    'Cassava_mosaic_disease',
    'Cherry_Healthy',
    'Cherry_Powdery_mildew',
    'Corn_Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_Common_rust_',
    'Corn_Healthy',
    'Corn_Northern_Leaf_Blight',
    'Grape_Black_rot',
    'Grape_Esca_(Black_Measles)',
    'Grape_Healthy',
    'Grape_Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Orange_Haunglongbing_(Citrus_greening)',
    'Peach_Bacterial_spot',
    'Peach_Healthy',
    'Pepper,_Bacterial_spot',
    'Pepper,_Healthy',
    'Potato_Early_blight',
    'Potato_Healthy',
    'Potato_Late_blight',
    'Raspberry_Healthy',
    'Rice_Bacterialblight',
    'Rice_Blast',
    'Rice_Brownspot',
    'Rice_Tungro',
    'Rose_Healthy',
    'Rose_Rose_Rust',
    'Rose_Rose_sawfly_Rose_slug',
    'Soybean_Healthy',
    'Squash_Powdery_mildew',
    'Strawberry_Healthy',
    'Strawberry_Leaf_scorch',
    'Sugarcane_Healthy',
    'Sugarcane_Mosaic',
    'Sugarcane_RedRot',
    'Sugarcane_Rust',
    'Sugarcane_Yellow',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Healthy',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites Two-spotted_spider_mite',
    'Tomato_Target_Spot',
    'Tomato_Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato_Tomato_mosaic_virus'
]

st.title("🌿 Projet reconnaissance plantes et maladies")

# # st.markdown(
#     """
#     <h1 style='
#         color: green; 
#         font-family: "Arial Black", Gadget, sans-serif; 
#         text-align: center; 
#         text-shadow: 2px 2px #b0e57c;
#         padding: 10px;
#     '>
#         Projet reconnaissance plantes et maladies
#     </h1>
#     """, 
#     unsafe_allow_html=True
# )


st.sidebar.title("Navigation")
pages=["Le projet","Dataset et Exploration",  "Modélisation","Analyses","Prédiction","Perspectives","About"]
page=st.sidebar.radio("Aller vers", pages)

if page == pages[0] : 
  st.write("### Home")
  

if page == pages[1] : 
   st.write("### DataVizualization")


if page == pages[2] : 
  st.write("### Modélisation")
  


def check_api_status(url="http://localhost:8000/health"):
    try:
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            return True
    except Exception:
        pass
    return False
def extract_class_name(filename):
    """
    Extrait 'Plante___Maladie' depuis un nom de fichier du type :
    'Plante___Maladie___<ID>.JPG'
    """
    return "___".join(filename.split("___")[:2])

def predict_via_api(image_pil, api_url, mode="single",show_heatmap=False):
    logger.info("🖼️ Préparation de l'image pour l'envoi à l'API...")

    # Conversion de l'image en base64
    buffered = BytesIO()
    image_pil.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    
    # Construction de l'URL
    url = f"{api_url}?show_heatmap={str(show_heatmap).lower()}"
    url = f"{url}&mode={str(mode).lower()}"
    logger.info(f"🌐 Envoi de la requête POST à l'API : {url}")

    payload = {"image": img_b64}
    
    try:
        response = requests.post(url, json=payload)
    except Exception as e:
        logger.error(f"❌ Erreur lors de la requête API : {e}")
        return None, None, [], [], []

    if response.status_code != 200:
        logger.error(f"❌ Erreur API : code {response.status_code} – {response.text}")
        return None, None, [], [], []

    result = response.json()

    logger.info("✅ Réponse reçue avec succès.")

    # Interprétation du nom de classe
    if isinstance(result["predicted_class"], int):
        pred_class_name = class_names[result["predicted_class"]]
    else:
        pred_class_name = result["predicted_class"]

    logger.info(f"📌 Prédiction : {pred_class_name} avec confiance {result['confidence']:.4f}")

    gradcam_images = []

    if show_heatmap:
        logger.info(f"🔥 Génération des heatmaps pour {len(result['models_heatmaps'])} modèles...")
        for idx, heatmap in enumerate(result["models_heatmaps"]):
            heatmap_array = np.array(heatmap)
            cam_overlay = apply_heatmap_on_image(image_pil, heatmap_array)
            gradcam_images.append(cam_overlay)
            logger.info(f"🖼️ Heatmap générée pour le modèle {result['models'][idx]}")

    return (
        pred_class_name,
        result["confidence"],
        result["models"],
        result["models_predictions"],
        result["models_confidences"],
        gradcam_images
    )
    
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Calcule la Grad-CAM heatmap pour une image donnée et un modèle Keras.

    Args:
      img_array (numpy array): image d'entrée preprocessée, shape (1, H, W, C)
      model (tf.keras.Model): modèle Keras complet (avec la tête softmax)
      last_conv_layer_name (str): nom de la dernière couche conv (ex: 'top_conv' dans EfficientNetB0)
      pred_index (int, optional): indice de la classe ciblée. Par défaut la classe prédite par le modèle.

    Returns:
      heatmap (numpy array): heatmap 2D normalisée entre 0 et 1
    """

    # 1. Récupérer la dernière couche conv
    #back_bone=model.get_layer("efficientnetv2-m")
    back_bone=model
    last_conv_layer = back_bone.get_layer(last_conv_layer_name)

    # 2. Créer un modèle intermédiaire qui donne les activations de la dernière couche conv
    # et la sortie finale du modèle
    grad_model = tf.keras.models.Model(
        [model.inputs], [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # 3. Calculer le gradient des logits de la classe cible par rapport aux activations conv
    grads = tape.gradient(class_channel, conv_outputs)

    # 4. Moyenne globale des gradients sur les axes spatiaux (H, W)
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))

    # 5. Pondérer chaque canal des activations par les gradients moyens
    conv_outputs = conv_outputs[0]
    pooled_grads = pooled_grads[0]

    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # 6. Normaliser la heatmap entre 0 et 1
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    return heatmap

def compute_saliency_map(model, image_array, class_index=None):
    """
    Calcule la carte de saillance (saliency map) d'une image pour un modèle donné.

    Args:
        model : modèle Keras TensorFlow.
        image_array : np.array, image d'entrée pré-traitée, shape (H, W, 3), float32.
        class_index : int ou None, index de la classe cible. 
                      Si None, on prend la prédiction la plus probable.

    Returns:
        saliency_map : np.array, carte de saillance normalisée (valeurs entre 0 et 1), shape (H, W).
    """
    if image_array.ndim == 3:
        image_tensor = tf.expand_dims(image_array, axis=0)
    else:
        image_tensor = tf.convert_to_tensor(image_array)

    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        preds = model(image_tensor)
        if class_index is None:
            class_index = tf.argmax(preds[0])
        class_score = preds[:, class_index]

    grads = tape.gradient(class_score, image_tensor)  # gradient de la sortie par rapport à l'image
    saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0]  # max sur canaux couleur, retirer batch

    # Normaliser entre 0 et 1
    saliency = (saliency - tf.reduce_min(saliency)) / (tf.reduce_max(saliency) - tf.reduce_min(saliency) + 1e-10)
    return saliency.numpy()

def apply_heatmap_on_image(image_pil, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Superpose le heatmap (2D float entre 0 et 1) sur l’image PIL d'origine.
    """
    # Convert PIL -> array
    image = np.array(image_pil.resize((heatmap.shape[1], heatmap.shape[0]))).copy()  # (H, W, 3)
    # Convert heatmap to [0, 255] uint8
    heatmap = np.uint8(255 * heatmap)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap, colormap)  # (H, W, 3), BGR
    
    # Convert BGR to RGB
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Superpose avec transparence
    superimposed_img = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return Image.fromarray(superimposed_img)

def overlay_heatmap(heatmap, image_pil, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap).resize(image_pil.size)
    heatmap = heatmap.convert("RGBA")
    image_pil = image_pil.convert("RGBA")
    blended = Image.blend(image_pil, heatmap, alpha)
    return blended
if page == pages[4]:
    st.title("🧠 Prédictions sur plusieurs images")

    is_online = st.toggle("Mode en ligne", value=True)
    api_up = check_api_status(API_HEALTH) and is_online

    if api_up:
        plan = st.radio("Sélectionnez votre plan :", ["Standard", "Premium"])

    if api_up:
        st.success("✅ Service API disponible")
    else:
        st.error("❌ Service API indisponible. Utilisation du modèle local.")
        is_online = False

    # 🎛️ Sélecteur de modèle avec infobulle
        model_choice = st.radio(
        "Sélectionnez le modèle local (hors API) :",
        (
            "Quantifié Dynamique – rapide – mobile standard",
            "Quantifié PTQ – ultra rapide – mobile ancien",
            "Float32 – lourd – mobile puissant"
        ),
        index=0,
        help="Choisissez en fonction de la puissance de votre appareil : PTQ pour une compatibilité maximale, Dynamique pour un bon compromis, Float32 pour une visualisation avancée."
        )

        # Description dynamique
        if "Dynamique" in model_choice:
            st.info("🔹 Modèle Quantifié Dynamique : rapide, idéal pour les mobiles d'entrée de gamme avec des ressources limitées. Grad-CAM indisponible.")
            model_tflite=model_dynamique
        elif "PTQ" in model_choice:
            st.info("🔹 Modèle Quantifié PTQ : ultra rapide, parfait pour les mobiles très bas de gamme ou anciens. Grad-CAM indisponible.")
            model_tflite=model_ptq
        else:
            st.info("🔹 Modèle Float32 : plus lourd, nécessite un mobile puissant ou récent. Grad-CAM disponible pour visualiser les activations.")

    # 👉 Tu peux continuer ici avec ton chargement de modèle en fonction du choix

    show_gradcam = False
    if is_online or (not is_online and "Float32" in model_choice):
        show_gradcam = st.checkbox("Afficher Grad-CAM", value=False)
    else:
        st.info("ℹ️ Grad-CAM non disponible avec modèle quantifié")

    uploaded_files = st.file_uploader(
        "Déposez plusieurs images ou choisissez-les :",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if st.button("🚀 Lancer les prédictions sur toutes les images"):
        if uploaded_files:
            total = len(uploaded_files)
            progress_bar = st.progress(0)
            progress_text = st.empty()
            summary_placeholder = st.empty()

            start_time = time.time()
            results = []

            def predict_one(uploaded_file):
                img = Image.open(uploaded_file).convert("RGB")
                file_name = os.path.basename(uploaded_file.name)
                true_class = extract_class_name(file_name)
                confidence = 0.0

                try:
                    if is_online and api_up:
                        mode_call = "single" if plan == "Standard" else "voting"
                        pred_class_name, confidence, models_names, models_predictions, models_confidences, gradcam_images = predict_via_api(
                            img, API_URL, mode_call, show_gradcam
                        )
                    else:
                        if model_choice.startswith("Quantifié"):
                            preds = model_tflite.predict(img.resize((224, 224)))
                            pred_class_idx = np.argmax(preds)
                            confidence = float(np.max(preds))
                            pred_class_name = class_names[pred_class_idx]
                            models_names = ["Modèle quantifié (TFLite)"]
                            gradcam_images = []
                        else:
                            preds, image_data = model_float32.predict(img.resize((224, 224)))
                            pred_class_idx = np.argmax(preds)
                            confidence = float(np.max(preds))
                            pred_class_name = class_names[pred_class_idx]
                            models_names = ["Modèle float32"]
                            gradcam_images = []
                            if show_gradcam:
                                heatmap = compute_saliency_map(model_float32.get_model(), image_data)
                                cam_overlay = apply_heatmap_on_image(img.resize((224, 224)), heatmap)
                                gradcam_images.append(cam_overlay)

                    return {
                        "file_name": file_name,
                        "image_obj":img,
                        "true_class": true_class,
                        "pred_class_name": pred_class_name,
                        "confidence": confidence,
                        "models_names": models_names,
                        "models_predictions": models_predictions if is_online and api_up else [],
                        "models_confidences": models_confidences if is_online and api_up else [],
                        "gradcam_images": gradcam_images,
                        "error": None
                    }
                except Exception as e:
                    return {
                        "file_name": file_name,
                        "error": str(e)
                    }

            if is_online and api_up:
                # Mode API avec parallélisation
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    futures = {executor.submit(predict_one, f): f for f in uploaded_files}
                    for i, future in enumerate(concurrent.futures.as_completed(futures)):
                        result = future.result()
                        results.append(result)
                        progress = int((i + 1) / total * 100)
                        progress_bar.progress(progress)
                        progress_text.text(f"🖼️ Image {i+1}/{total} traitée...")
            else:
                # Mode local séquentiel (non parallélisé)
                for i, f in enumerate(uploaded_files):
                    result = predict_one(f)
                    results.append(result)
                    progress = int((i + 1) / total * 100)
                    progress_bar.progress(progress)
                    progress_text.text(f"🖼️ Image {i+1}/{total} traitée...")

            elapsed = time.time() - start_time

            # Affichage résumé
            summary_placeholder.markdown(
                f"""
                🎉 **Toutes les prédictions sont terminées !**  
                🕒 {total} image(s) traitée(s) en {elapsed:.2f} secondes.
                """
            )

            # Affichage des résultats détaillés après la boucle
            for res in results:
                if res.get("error"):
                    st.error(f"Erreur pour `{res['file_name']}` : {res['error']}")
                    continue

                st.image(res['image_obj'], caption=f"Image : {res['file_name']}", use_container_width=True)
                st.markdown("### 🔍 Résultat de la prédiction")
                st.write(f"📁 Nom du fichier : `{res['file_name']}`")
                st.write(f"🏷️ Vraie classe (extrait nom) : `{res['true_class']}`")
                st.write(f"✅ Classe prédite : `{res['pred_class_name']}`")
                st.write(f"📊 Confiance : **{res['confidence']*100:.2f}%**")

                if is_online:
                    if plan == "Standard":
                        st.info(f"🧠 Mode : **Standard** — 1 modèle utilisé : `{res['models_names'][0]}`")
                    elif plan == "Premium":
                        st.success(f"🌟 Mode : **Premium** — Modèles utilisés : {', '.join(res['models_names'])}")
                        if len(res['models_names']) > 1:
                            with st.expander("🔎 Détails des votes de chaque modèle"):
                                for i, model_name in enumerate(res['models_names']):
                                    model_pred_class = res['models_predictions'][i]
                                    model_confidence = 100 * res['models_confidences'][i]
                                    st.write(f"🧠 **{model_name}** : classe `{class_names[model_pred_class]}` avec confiance **{model_confidence:.2f}%**")
                else:
                    st.warning(f"⚠️ Mode local : modèle utilisé `{res['models_names'][0]}`")

                if res['gradcam_images']:
                    with st.expander("🖼️ Visualisations Grad-CAM"):
                        for i, img_cam in enumerate(res['gradcam_images']):
                            st.image(img_cam, caption=f"Salience - Modèle : {res['models_names'][i]}")

        else:
            st.warning("⚠️ Veuillez uploader au moins une image.")
