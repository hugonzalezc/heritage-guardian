# Heritage Guardian: Mural vs. Grafiti

Proyecto Aplicado — Aprendizaje Automático II (PUCV)
Clasificación binaria (mural patrimonial vs. grafiti no autorizado) comparando 
ResNet18 y ViT-Small.

## Autores
Hugo González, Rodrigo Venegas

## Estructura
- `proyecto_avance.ipynb`: notebook principal (preprocesamiento, entrenamiento 
  ResNet18, entrenamiento ViT-Small, métricas, experimento adicional con Random Forest).
- `resnet18_heritage_guardian.pth`: pesos entrenados de ResNet18.
- `vit_small_heritage_guardian.pth`: pesos entrenados de ViT-Small.
- `requirements.txt`: dependencias.

## Cómo reproducir
1. Crear entorno e instalar dependencias:
pip install -r requirements.txt
2. Correr `proyecto_avance.ipynb` de principio a fin (requiere GPU con CUDA para 
   tiempos de entrenamiento razonables; ~2-3 min por modelo en GPU tipo T4).
3. El notebook descarga automáticamente los datos (`graffiti.zip`, `muralism.zip`) 
   desde Google Drive, los descomprime, y construye la estructura 
   `dataset_estructurado/{train,val,test}/{mural,grafiti}` con split 70/15/15.
4. Los checkpoints de ambos modelos se guardan automáticamente al final de cada 
   entrenamiento.

## Datos
- Grafiti: subconjunto derivado de datasets públicos tipo "Graffiti Classification" 
  (Kaggle) + Roboflow Universe.
- Murales: subconjunto derivado de WikiArt (composición/estilo).
- Licencia y atribución completa: pendiente, se documentará en la Data Card 
  (entregable Final).
- Reproducibilidad: semilla fija (`SEED = 42`) para PyTorch, NumPy y random.

## Resultados preliminares (test set, 246 imágenes)

| Modelo | Accuracy | F1 grafiti | F1 mural | AUC-ROC | MCC |
|---|---|---|---|---|---|
| ResNet18 | 0.74 | 0.81 | 0.59 | 0.7651 | 0.3993 |
| ViT-Small | 0.81 | 0.85 | 0.73 | 0.8787 | 0.5942 |

Ver la sección "Riesgos identificados" en el notebook para limitaciones conocidas 
(desbalance de clases, overfitting en ViT, domain gap geográfico).

## Estado del proyecto
Entrega de Avance (5%). Pendiente para la etapa Final: Data Card completa, 
interpretabilidad (Grad-CAM/attention rollout), demo funcional, informe técnico.