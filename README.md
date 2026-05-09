# Extendiendo DDColor: Análisis e investigación de las limitaciones del modelo.

> Grupo 7: Alejandro Parody Quirós, Guillermo Rodríguez Narbona , José Carlos Mora López

Este repositorio contiene el trabajo realizado por el grupo 7 de la asignatura de "Procesamiento de Imágenes Digitales", del grado Ingeniería Informatica del Software, de la Universidad de Sevilla.

Fundamentalmente se trata de una extensión de la implementación de DDColor, propuesto en el paper "DDColor: Towards Photo-Realistic Image Colorization via Dual Decoders", de los autores Xiaoyang Kang, Tao Yang, Wenqi Ouyang, Peiran Ren, Lingzhi Li, Xuansong Xie.

Artículo disponible en: https://arxiv.org/abs/2212.11613

Implementación disponible en: https://github.com/piddnad/DDColor

## Propuesta y objetivos

>Importante: tanto los cambios en el código como los reentrenamientos del modelo están todavía en ejecución y, de momento, puede ocurrir que no satisfaga las propuesta planteada en este apartado.

Este trabajo pretende analizar dos limitaciones del modelo original. 

Por un lado trataremos el comportamiento del modelo en ámbitos en los que no ha sido entrenado. Con esto nos referimos concretamente a evaluar el funcionamiento del modelo ante imágenes de
naturaleza *no fotorrealista* (otros estilos gráficos, i.e. viñetas de comic). 

Por el otro lado, se analiza una limitación propuesta en el paper orignal, sobre artefactos que alteran la colorizacion cuando aparecen elementos que presentan trasparencias. 

Para ello resulta fundamental a su vez conocer las métricas de evaluación de imágenes para hacer una correcta comparativa de los resultados obtenidos. Se añade al código a su vez los cálculos de dichas métricas.

Finalmente, trabajamos también sobre una interfaz gráfica que permita usar el modelo de una forma más sencilla.

## Instalación
### Requisitos
- Python >= 3.7
- PyTorch >= 1.7

### Instalación con Conda (Recomendado)

```
conda create -n ddcolor python=3.9
conda activate ddcolor
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt

# Para realizar entrenamientos, instala el siguiente contenido adicional, dependencias y basicsr

pip install -r requirements.train.txt
python3 setup.py develop
```

## Nuestro modelo

Aquí aparecerá información sobre como descargar y utilizar el modelo que nosotros hemos entrenado en el dominio *no fotorrealista*  tan pronto como consigamos subirlo a Internet.

## Inicio rápido del modelo original
### Usar el modelo entrenado original con un script local (No `basicsr` Required)

1. Descargar el modelo preentrenado. Para ello hacer y correr un script de python con el siguiente contenido.

```python
from modelscope.hub.snapshot_download import snapshot_download

model_dir = snapshot_download('damo/cv_ddcolor_image-colorization', cache_dir='./modelscope')
print('model assets saved to %s' % model_dir)
```

2.	Correr inference con:

```sh
python scripts/infer.py --model_path ./modelscope/damo/cv_ddcolor_image-colorization/pytorch_model.pt --input ./assets/test_images
```
o
```sh
sh scripts/inference.sh
```

### Usar modelo original ya entrenado desde Hugging Face 
Descargar el modelo via Hugging Face Hub:

```python
from huggingface_hub import PyTorchModelHubMixin
from ddcolor import DDColor

class DDColorHF(DDColor, PyTorchModelHubMixin):
    def __init__(self, config=None, **kwargs):
        if isinstance(config, dict):
            kwargs = {**config, **kwargs}
        super().__init__(**kwargs)

ddcolor_paper_tiny = DDColorHF.from_pretrained("piddnad/ddcolor_paper_tiny")
ddcolor_paper      = DDColorHF.from_pretrained("piddnad/ddcolor_paper")
ddcolor_modelscope = DDColorHF.from_pretrained("piddnad/ddcolor_modelscope")
ddcolor_artistic   = DDColorHF.from_pretrained("piddnad/ddcolor_artistic")
```

O correr el modelo preentrenado directamente, ejecutando: 

```sh
python scripts/infer.py --model_name ddcolor_modelscope --input ./assets/test_images
# model_name: [ddcolor_paper | ddcolor_modelscope | ddcolor_artistic | ddcolor_paper_tiny]
```

### Usar modelo original ya entrenado desde ModelScope
1. Instalar modelscope:

```sh
pip install modelscope
```

2. Correr el modelo:

```python
import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

img_colorization = pipeline(Tasks.image_colorization, model='damo/cv_ddcolor_image-colorization')
result = img_colorization('https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/audrey_hepburn.jpg')
cv2.imwrite('result.png', result[OutputKeys.OUTPUT_IMG])
```

Este código descargará automáticamente el modelo ddcolor_modelscope (ver ModelZoo) y realizará la inferencia. El archivo del modelo pytorch_model.pt se puede encontrar en la ruta local ~/.cache/modelscope/hub/damo.

## Aplicación de escritorio: DDColor Lab

DDColor Lab es una interfaz gráfica que permite colorizar imágenes y comparar los resultados de los tres modelos de forma visual y cuantitativa.

### Instalación

Se recomienda usar el entorno Conda del proyecto (ver sección Instalación). Si se prefiere un entorno virtual independiente:

```sh
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```

Instalar las dependencias de la app:

```sh
pip install torch torchvision opencv-python pillow scikit-image
python setup.py develop
```

### Pesos de los modelos

Colocar los pesos en `./models/` con los siguientes nombres:

| Fichero | Origen | Descarga |
|---|---|---|
| `ddcolor_comic.pth` | Fine-tune propio | [Aleparqui/PID_proyect](https://huggingface.co/Aleparqui/PID_proyect) → `net_g_latest.pth` (renombrar) |
| `ddcolor_realistic.pth` | DDColor oficial | [piddnad/DDColor-models](https://huggingface.co/piddnad/DDColor-models) → `ddcolor_modelscope.pth` (renombrar) |
| `ddcolor_artistic.pth` | DDColor oficial | [piddnad/DDColor-models](https://huggingface.co/piddnad/DDColor-models) → `ddcolor_artistic.pth` |

Descarga programática:

```python
from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="Aleparqui/PID_proyect",
                filename="net_g_latest.pth", local_dir="./models/")
# Renombrar a ddcolor_comic.pth

hf_hub_download(repo_id="piddnad/DDColor-models",
                filename="ddcolor_modelscope.pth", local_dir="./models/")
# Renombrar a ddcolor_realistic.pth

hf_hub_download(repo_id="piddnad/DDColor-models",
                filename="ddcolor_artistic.pth", local_dir="./models/")
```

Si algún peso no está disponible, la app opera en **modo DEMO** para ese modelo.

### Uso

```sh
python ddcolor_desktop.py
```

1. **Abrir imagen** — formatos admitidos: PNG, JPG, BMP, TIFF, WebP. Se usa automáticamente como ground truth para las métricas.
2. **Abrir GT (opcional)** — carga una imagen de referencia distinta para las métricas.
3. **Seleccionar modelos** — activar/desactivar Cómic FT, Realista HF y Artístico HF individualmente.
4. **▶ Ejecutar** — los resultados aparecen en el panel derecho al finalizar.
5. **Métricas** — pestaña con tabla comparativa (ΔCF y ΔE₂₀₀₀); el mejor resultado de cada fila se marca con ✓.
6. **Guardar resultados** — exporta las imágenes colorizadas como `result_comic_ft.png`, `result_realistic_hf.png` y `result_artistic_hf.png`.

## Más información
Para acceder a información sobre otros modelos entrenados, como entrenar tu modelo propio, o exportar a ONXX, recomendamos acceder al repositorio original disponible al inicio de este documento.
