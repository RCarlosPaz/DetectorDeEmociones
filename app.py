import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import gradio as gr

# Configuración de dispositivo (GPU si está disponible, CPU si no)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Definir la misma arquitectura de modelo que se usó para entrenar
# Cargamos un ResNet18 con pesos preentrenados (aunque los vamos a sobrescribir)
pesos = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=pesos)

# Reemplazamos la última capa (la cabeza de clasificación)
num_caracteristicas = model.fc.in_features
model.fc = nn.Linear(num_caracteristicas, 2) # 2 clases (fire, nofire)

# 2. Cargar los pesos entrenados del modelo
# Asegúrate de que este path coincida con donde subas el modelo en HF Space
model_path = 'my_pytorch_model.pth' 
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval() # Poner el modelo en modo evaluación

# 3. Definir las transformaciones para las imágenes de entrada (las mismas que para test)
transformaciones_inferencia = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 4. Definir la función de inferencia
def predict_image(image):
    if image is None:
        return "Por favor, sube una imagen."
    
    # Convertir a PIL Image si no lo es (Gradio puede pasar numpy arrays)
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    image_tensor = transformaciones_inferencia(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted_idx = torch.max(outputs, 1)

    # Mapear los índices a las clases originales
    clases = ['fire', 'nofire'] # Asegúrate de que esto coincida con tu entrenamiento
    predicted_class = clases[predicted_idx.item()]
    confidence = probabilities[0][predicted_idx.item()].item()
    
    return f"Predicción: {predicted_class} (Confianza: {confidence:.2f})"

# 5. Configurar la interfaz de Gradio
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Sube una imagen"),
    outputs=gr.Textbox(label="Resultado de la Predicción"),
    title="Clasificador de Fuego con Transfer Learning (ResNet18)",
    description="Sube una imagen para predecir si contiene fuego o no."
)

# Para ejecutar la interfaz (solo si se ejecuta directamente, no en Hugging Face Spaces)
# if __name__ == '__main__':
#     interface.launch()
