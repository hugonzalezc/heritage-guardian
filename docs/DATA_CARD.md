# Data Card — Heritage Guardian

**Versión:** entrega final, noviembre de 2026 · **Responsables:** equipo Heritage Guardian, Aprendizaje Automático II, PUCV.

## 1. Propósito y alcance
Dataset para clasificación binaria de fotografías de intervenciones urbanas mediante proxies visuales: `grafiti` (0) y `mural` (1). Se utiliza para comparar CNN, Vision Transformers y un clasificador clásico. No contiene evidencia de autorización, protección UNESCO, valor patrimonial o vandalismo; no permite inferir dichos atributos. Uso previsto: investigación académica y evaluación exploratoria, no decisiones municipales automatizadas.

## 2. Procedencia y composición
El avance declara fuentes derivadas de Graffiti Classification (Kaggle), Roboflow Universe y WikiArt. Los archivos de trabajo son `graffiti.zip` y `muralism.zip`; no se ha recuperado la trazabilidad individual a los conjuntos primarios ni sus versiones/licencias exactas. La disponibilidad pública no equivale a permiso de redistribución.

| Partición | Grafiti | Mural | Total |
|---|---:|---:|---:|
| Train | 765 | 362 | 1127 |
| Validación | 164 | 78 | 242 |
| Test | 164 | 78 | 242 |
| Total | 1093 | 518 | 1611 |

El 67,85 % corresponde a grafiti y el 32,15 % a mural. El avance registró 1637 imágenes; la reconstrucción final contiene 26 menos, todas de la clase grafiti. No se dispone de una conciliación individual de esas diferencias; la comparación histórica es orientativa.

## 3. Construcción y etiquetado original
`src/prepare_data.py` extrae los archivos, asigna la clase por carpeta y reclasifica los nombres con prefijo `vandalico_` de `muralista` como grafiti. Verifica imágenes con Pillow, deduplica por SHA-256 y divide por clase aproximadamente 70/15/15 con semilla 42. El manifiesto `results/dataset_manifest.csv` vincula origen dentro del archivo, destino, clase, split y hash. La auditoría final no detectó duplicados exactos entre particiones. No se ha verificado cada etiqueta ni la independencia por obra, fotógrafo o ubicación; los hashes no detectan variantes recomprimidas o recortadas.

## 4. Evaluación externa de Valparaíso
68 fotografías recopiladas de fuentes web, etiquetadas antes de la inferencia y sin incorporarlas al entrenamiento. El CSV `data/valparaiso_metadata.csv` registra ID, nombre sin extensión, URL, ubicación declarada, etiqueta, autor, licencia y notas. Las dos últimas propiedades de procedencia (autor y licencia) no están verificadas; el identificador de una cuenta no acredita autoría. La ubicación se declara según la recopilación, no mediante una auditoría geográfica independiente.

| Etiqueta | Cantidad | Uso |
|---|---:|---|
| Grafiti | 33 | Test externo |
| Mural | 30 | Test externo |
| Dudoso | 5 | Análisis cualitativo sin métricas de acierto |

66 URLs provienen de Flickr, una de Instagram y una de Tur.com. 30 de los 33 grafitis provienen de la cuenta Flickr `58947252@N05`; 25 de los 30 murales provienen de `duncan`. Esta correlación entre fuente y clase puede producir sesgos de encuadre, cámara o estilo. La selección es de conveniencia, no probabilística; no representa necesariamente toda la ciudad. No se midió acuerdo entre anotadores; hubo revisión humana con asistencia conversacional para casos fronterizos, sin validación independiente de etiquetas.

**Criterio operacional local:** grafiti si predominan firmas, tags, rayados o letras estilizadas; mural si predomina una composición pictórica, figurativa o abstracta; dudoso cuando coexisten ambos sin predominio claro. No se considera la calidad artística, técnica, legalidad o valor patrimonial. Los cinco casos ambiguos no se utilizaron para métricas binarias.

`src/label_local.py` valida nombres y formatos, decodificación y SHA-256, copia los archivos sin alterar originales y genera `results/metrics/local_dataset_manifest.csv`. Resultado: 68 válidas, ninguna faltante, corrupta, sin registro o duplicada exactamente respecto de las otras imágenes locales y las 1611 originales. No se verificaron sistemáticamente casi duplicados ni obras repetidas. La evaluación local se mantuvo separada del ajuste de modelos, pesos del ensemble y umbrales.

## 5. Preprocesamiento y acceso
RGB, 224×224, normalización ImageNet; aumento aleatorio solo en entrenamiento. Baseline: resize y flip horizontal; improved/balanced: random resized crop, flip, rotación y color jitter. Balanced utiliza muestreo ponderado solo en train. Validación, test y extracción de embeddings son deterministas en su transformación. Los originales locales permanecen en `data/valparaiso_raw`; `data/test_valparaiso/{grafiti,mural}` contiene 63 copias y `data/valparaiso_ambiguous` contiene cinco.

## 6. Calidad, sesgos y límites
- Desbalance original; presencia observada de collages, marcas de agua, diferencias de resolución, fondos urbanos y etiquetas potencialmente ambiguas.
- Las fuentes y definiciones pueden diferir entre conjuntos; no se midió cuánto aporta cada factor a la brecha de dominio.
- Una etiqueta por fotografía no describe adecuadamente escenas con múltiples intervenciones; no hay segmentación ni localización anotada.
- Una sola semilla y muestra local pequeña; no se reporta significancia estadística ni acuerdo interanotador.
- ViT Improved: Macro F1 0,8177 en test original y 0,5055 en local; recall mural 0,8205 y 0,1667, respectivamente. Los tres modelos principales predijeron grafiti en los cinco casos dudosos. Las puntuaciones no son garantías de acierto.

## 7. Licencias, privacidad y mantenimiento
Estado: **pendiente de verificación de derechos de las imágenes originales y locales**. No redistribuir fotografías, collages, mapas superpuestos ni checkpoints derivados sin revisar las condiciones correspondientes. Mantener URLs y atribuciones verificadas en el CSV; registrar licencia individual y fecha de consulta cuando se confirme. No publicar datos privados, rostros identificables innecesarios ni metadatos sensibles. Los hashes y métricas agregadas pueden documentarse sin publicar imágenes. El código propio y las dependencias tienen condiciones distintas; no atribuir una licencia abierta a los datos por licenciar el código. Para acceso académico, solicitar instrucciones al equipo y verificar permisos. Cualquier versión posterior debe preservar los splits originales, registrar cambios y reservar un nuevo test externo para evaluar adaptación de dominio.
