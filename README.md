# Heart sound segmentation

This repository contains the codes used for the development of a heart sound prediction system from phonocardiograms (PCG) using Convoloutional Neural Networks (CNN). This model allows prediction of the time of occurence of the first heart sound (S1) and the second heart sound (S2), from which the systolic and diastolic intervals can be inferred.

The development of this project was performed in the context of my Master of Engineering Sciences research entitled "[*Design of a preprocessing system for sounds obtained from chest auscultation*](https://repositorio.uc.cl/handle/11534/60994)" at Pontificia Universidad Catolica de Chile, which derived in the paper entitled "Study of hyperparameters in the semantic segmentation of heart sounds".

## 1. Theoretical background

The phonocardiogram (PCG) is a recording of the heart sounds produced by the opening and closing of the heart valves. The main components of the PCG are the first heart sound (S1) and the second heart sound (S2).

S1 is generated during ventricular systole (closure of the atrioventricular valves: mitral/bicuspid and tricuspid), in which the ventricles contract and
allow blood to be pumped from the heart to the rest of the body through the aorta and pulmonary arteries. S2 occurs during ventricular diastole (closure of the sigmoid/semilunar valves: aortic and pulmonary) in which the ventricles relax and allow blood to flow in from the atria. In comparison, S1 is a lower-pitched sound of longer duration, while S2 is a higher-pitched sound of shorter duration. 

In this work, CNNs are used to detect the presence of fundamental heart sounds. One of the advantages of this type of the networks is that they allow independence of the temporal relationships of the signal. Each convolutional layer can be understood as a filter that is adjusted to detect the segments of interest within the heart sound. A semantic segmentation architecture based on the [SegNet](https://arxiv.org/abs/1511.00561) network is proposed (see figure 1).

<figure>
	<div style="text-align:center">
		<img src="imgs/CNN_architectures-SegNet_SPA.png" width="100%">
    </div>
	<figcaption align = "center"><b>Figure 1: Semantic Segmentation CNN.</b></figcaption>
</figure>

Based on the results of this study, the architecture available in the `models` folder is defined to address the problem of heart sound segmentation (for more details on the findings of this work, please refer to chapter 2 of the thesis "[*Design of a preprocessing system for sounds obtained from chest auscultation*](https://repositorio.uc.cl/handle/11534/60994)".


## 2. Database

Para el entrenamiento de este sistema, se utilizó una base de datos de sonidos cardiacos disponible en la implementación de Springer titulada "[Logistic Regression-HSMM-based Heart Sound Segmentation](https://physionet.org/content/hss/1.0/)", la cual fue presentada para la etapa de segmentación de sonidos cardiacos en el contexto del desafío PhysioNet/CinC del año 2016. 

Esta base de datos cuenta con 792 registros de audio de obtenidos a partir de 135 pacientes distintos, los cuales son auscultados en distintas posiciones del pecho. Cada uno de estos archivos de audio se encuentra muestreado a 1000 Hz, y posee etiquetas muestreadas a 50 Hz que indican 4 posibles estados: S1, sístole, S2 y diástole. Estas etiquetas son definidas con el *peak*-R y el final de la onda T de un electrocardiograma (ECG) sincronizado con el estetoscopio con el que se grabaron los sonidos cardiacos. Sin embargo, ninguna de las etiquetas proporcionadas poseen corrección humana.

En la figura 2 es posible apreciar un ejemplo de un sonido cardiaco en conjunto con sus etiquetas.

<figure>
	<div style="text-align:center">
		<img src="imgs/Database_sound_and_labels.png" width="70%">
    </div>
	<figcaption align = "center"><b>Figure 2: Heart sounds and their labels.</b></figcaption>
</figure>

## 3. Repository contents

The folders and files that comprise this project are:

* `hsp_utils`: Contiene funciones que permiten operar las funciones principales de segmentación.
* `imgs`: Carpeta con imágenes que se incluyen en este `README`.
* `jupyter_test`: Contiene el archivo `testing_notebook.ipynb` que permite realizar experimentos del modelo entrenado sobre los archivos disponibles en la carpeta `samples_test`.
* `models`: Contiene el modelo final entrenado en formato `.h5`.
* `samples_test`: Contiene una pequeña muestra de la base de datos presentada en la [sección 2](#2-base-de-datos).
* `training_scripts`: Contiene algunos de los archivos utilizados para el entrenamiento de las redes. Sin embargo, no se asegura el correcto funcionamiento de estos archivos. Se incluyen simplemente para tener una noción de cómo se implementó este proyecto. Para más detalles se recomienda revisar en el repositorio [`Scripts_magister`](https://github.com/cmescobar/Scripts_Magister) la carpeta [`Heart_sound_segmentation_v2`](https://github.com/cmescobar/Scripts_Magister/tree/master/Heart_sound_segmentation_v2) para ver el historial de cambios sobre este experimento (:warning:**Se advierte de manera previa que esa carpeta corresponde a una etapa experimental/borrador del trabajo realizado, y por ende, no se encuentra ordenada ni es apta para utilizar directamente los códigos. En caso de estar interesado en más detalle aún, se sugiere contactar a mi correo personal**:warning:).
* `main.py`: Archivo que contiene un ejemplo de ejecución para la función que realiza la predicción de los instantes de ocurrencia de los sonidos cardiacos.
* `prediction_functions.py`: Archivo que contiene las funciones que permiten aplicar la predicción de las posiciones de los sonidos cardiacos utilizando la red CNN con arquitectura *encoder-decoder*.

## 4. Requirements

For the development of these modules the following list of libraries were used. Since the correct functioning of the repository cannot be ensured for later versions of these libraries, the version of each library will also be incorporated. This implementation was developed using Python 3.7.

* [NumPy](https://numpy.org/) (1.18.4)
* [SciPy](https://scipy.org/) (1.5.4)
* [Tensorflow](https://www.tensorflow.org/) (2.3.1) 
* [Matplotlib](https://matplotlib.org/) (3.3.2)
* [Soundfile](https://pysoundfile.readthedocs.io/en/latest/) (0.10.3)
* [PyWavelets](https://pywavelets.readthedocs.io/en/latest/) (1.0.3)
* [PyEMD](https://pyemd.readthedocs.io/en/latest/intro.html) (0.2.10)

## 5. Ejemplo de prueba

An example is provided in the notebook located at jupyter_test/testing_notebook.ipynb, which contains a guided execution of the prediction function.

The following code is similar to that available in the main.py file.

```python
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.io import loadmat
from prediction_functions import hss_segmentation

# Parameters
lowpass_params = {'freq_pass': 140, 'freq_stop': 150}
model_name = 'definitive_segnet_based'
db_folder = 'samples_test'

# Opening an audio sample
filename = 'samples_test/435_Patient081_Dis1'
audio, samplerate = sf.read(f'{filename}.wav')
labels = loadmat(f'{filename}.mat')['PCG_states']

# Getting the outputs of the network
_, y_hat_to, (y_out2, y_out3, y_out4) = \
        hss_segmentation(audio, samplerate, model_name,
        length_desired=len(audio),
        lowpass_params=lowpass_params,
        plot_outputs=False)
```
