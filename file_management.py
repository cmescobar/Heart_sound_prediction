import os
import numpy as np
import soundfile as sf
from ast import literal_eval

DIAGNOSIS_CSV = 'Respiratory_Sound_Database/patient_diagnosis.csv'
AUDIO_FILES = 'Respiratory_Sound_Database/audio_and_txt_files'


def get_patient_by_symptom(symptom):
    '''Esta función permite retornar una lista de pacientes que tiene la
    etiqueta de la enfermedad respiratoria pedida en la variable
    "symptom".
    
    Parámetros
    - "symptom": (string) Enfermedad a buscar

    Retorna una lista (out) con los ID de los pacientes que estén con ese
    síntoma.'''

    # Definición de la variable de salida
    out = []

    with open(DIAGNOSIS_CSV, 'r', encoding='utf8') as file:
        for line in file:
            id, patient_symptom = line.strip('\n').split(',')
            if patient_symptom == symptom:
                out.append(id)
    return out


def get_dir_audio_by_id(id_list):
    '''Esta función permite retornar una lista de la dirección de los audios de
    los pacientes que se encuentran en la lista de identificadores "id_list"
     
    Parámetros
    - "id_list": (list) Lista de id's de los pacientes a buscar

    Retorna una lista (out) con las direcciones de los audios de esta lista
    de id's.'''

    # Definición del vector de salida
    out = []

    files = os.listdir(AUDIO_FILES)
    for i in files:
        if i.endswith('.wav') and i[:3] in id_list:
            out.append(AUDIO_FILES + '/' + i)

    return out


def get_dir_audiotxt_by_id(id_list):
    '''Esta función permite retornar una lista de la dirección de los audios de
     los pacientes que se encuentran en la lista de identificadores "id_list"
     
    Parámetros
    - "id_list": (list) Lista de id's de los pacientes a buscar

    Retorna una lista (out) con las direcciones de los txt de los audios de esta
    lista de id's.'''

    # Definición del vector de salida
    out = []

    files = os.listdir(AUDIO_FILES)
    for i in files:
        if i.endswith('.txt') and i[:3] in id_list:
            out.append(AUDIO_FILES + '/' + i)

    return out


def get_dir_audiotxt_by_symptom(symptom, wav=True):
    '''Esta función permite retornar una lista de pacientes que tiene la
    etiqueta de la enfermedad respiratoria pedida en la variable
    "symptom".
    
    Parámetros
    - "symptom": (string) Enfermedad a buscar

    Retorna una lista (out) con direcciones de los audios de los pacientes que
    estén con ese síntoma. '''

    # Lista las id de pacientes por síntoma
    id_list = get_patient_by_symptom(symptom)

    if wav:
        return get_dir_audio_by_id(id_list)
    else:
        return get_dir_audiotxt_by_id(id_list)


def get_audio_folder_by_symptom(symptom, sep_type='all'):
    # Corroborar la opción de separación
    if sep_type not in ['all', 'tracheal', 'toracic']:
        print('The option is not valid. Please try again.')
        return

    # Obtener las id de los pacientes
    symptom_list = get_patient_by_symptom(symptom)

    # Luego obtener las direcciones de los archivos de audio para cada caso
    symptom_dirs = get_dir_audio_by_id(symptom_list)

    # Definición de la carpeta dónde se guardará la información
    folder_data = f'Interest_Audios/{symptom}/{sep_type}'
    
    # Preguntar si es que la carpeta que almacenará los sonidos se ha
    # creado. En caso de que no exista, se crea una carpeta
    if not os.path.isdir(folder_data):
        os.makedirs(folder_data)

    # Lectura de cada uno de los archivos de audios para la reescritura
    # normalizada del archivo de audio
    for i in symptom_dirs:
        if sep_type == 'tracheal'and 'Tc' not in i:
            continue
        elif sep_type == 'toracic' and 'Tc' in i:
            continue

        # Lectura del audio
        audio, samplerate = sf.read(i)

        # Normalizando al archivo de audio
        audio /= max(abs(audio))

        # Definición de la carpeta donde se guardará
        folder_to_save = f"{folder_data}/{i.split('/')[-1]}"
        
        # Escribiendo
        print(f"Re-writing file sound {i.split('/')[-1]}...")
        sf.write(folder_to_save, audio, samplerate, 'PCM_24')
        print("Re-writing complete!\n")


def get_segmentation_points_by_filename(symptom, filename):
    # Dirección de los puntos originales
    filepath = f"Results/{symptom}/{symptom}_resp_original_points.txt"
    # Abriendo el archivo
    with open(filepath, 'r', encoding='utf8') as file:
        for i in file:
            # Obteniendo el nombre
            name_in_file = i.split(';')[0]
            if name_in_file == filename:
                # Se obtiene la lista en texto
                list_txt = i.split(';')[1].strip()
                # Transformando a lista
                return literal_eval(list_txt)


def get_heart_sound_files():
    '''
        Obsoleto desde que existe la función de abajo
    '''
    # Dirección del archivo de sonidos cardíacos
    filename = "labels/Heart_sound_present_signals.csv"
    
    # Definición de la carpeta dónde se guardará la información
    folder_data = f'Interest_Audios/Heart_sound_files'
    
    # Preguntar si es que la carpeta que almacenará los sonidos se ha
    # creado. En caso de que no exista, se crea una carpeta
    if not os.path.isdir(folder_data):
        os.makedirs(folder_data)
    
    # Abriendo el archivo
    with open(filename, 'r', encoding='utf8') as file:
        for i in file:
            # Nombre del archivo .wav
            audio_name = i.strip() + ".wav"
            
            # Dirección del archivo en la carpeta madre. Este archivo es el que 
            # se copiará
            dir_to_copy = f"Respiratory_Sound_Database/audio_and_txt_files/"\
                          f"{audio_name}"
                          
            # Dirección en la cual se almacenará este nuevo archivo
            dir_to_paste = f"{folder_data}/{audio_name}"
            
            # Abriendo y re grabando
            audio, samplerate = sf.read(dir_to_copy)
            
            # Normalizando
            audio = audio / max(abs(audio))
            
            # Re grabando
            sf.write(dir_to_paste, audio, samplerate)


def get_heart_sound_by_presence(level=4):
    ''' Función que permite separar los sonidos respiratorios más cardíacos por su
    indicador de presencia cardíaca, siendo 1 el más bajo (menos presente), y 4 el
    más alto (más presente).
    
    Parámetros
    - level: Nivel de separabilidad
        - ['all']: Crea una carpeta de todos los sonidos de esa base
        - [int]: Crea una carpeta del nivel indicado
    '''
    # Dirección del archivo de sonidos cardíacos
    filename = "labels/Heart_sound_present_signals.csv"
    
    # Definición de la carpeta dónde se guardará la información
    if isinstance(level, int):
        folder_data = f'Interest_Audios/Heart_sound_files/Level {level}'
    elif level == 'all':
        folder_data = f'Interest_Audios/Heart_sound_files/all'
    else:
        print('Opción no válida. Por favor intente nuevamente.')
        return
    
    # Preguntar si es que la carpeta que almacenará los sonidos se ha
    # creado. En caso de que no exista, se crea una carpeta
    if not os.path.isdir(folder_data):
        os.makedirs(folder_data)
    
    with open(filename, 'r', encoding='utf8') as file:
        for i in file:
            # Eliminando el caracter de salto
            line = i.strip()
            # Revisando temas de formato (si la línea no está vacía, comienza con
            # un salto vacío o parte con una seña de comentario, entonces es 
            # candidato a ser el nombre de un archivo)
            if not (line == '' or line[0] == '' or line[0] == '#'):
                audio_name, audio_level = line.split(',')
            else:
                continue
            
            # Filtro para los archivos del nivel de interés
            if level == int(audio_level):
                print(f'Recording {audio_name}.wav...')
                # Dirección del archivo en la carpeta madre. Este archivo es el que 
                # se copiará
                dir_to_copy = f"Respiratory_Sound_Database/audio_and_txt_files/"\
                              f"{audio_name}.wav"
                              
                # Dirección en la cual se almacenará este nuevo archivo
                dir_to_paste = f"{folder_data}/{audio_name}.wav"
                
                # Se abre el archivo de interés
                audio, samplerate = sf.read(dir_to_copy)
                
                # Normalizando
                audio = audio / max(abs(audio))
                
                # Re grabando
                sf.write(dir_to_paste, audio, samplerate)
                print('Completed!\n')


def get_heart_sound_files_heartbeatdb(datatype='normal', setdef='a'):
    '''Función que genera una base de datos de sonidos cardiacos basados en
    la carpeta "Heartbeat sounds", eligiendo qué tipo de señal se quiere 
    obtener y el set
    
    Parámetros
    - datatype: Tipo de sonido cardíaco a obtener
    - setdef: Es posible elegir entre la base de datos "a" o "b"
    '''
    # Dirección de ubicación de archivos cardíacos
    filepath = f'Heartbeat sounds/set_{setdef}'
    
    # Definición de la carpeta a guardar
    folder_data = f'Heartbeat sounds/Generated/{datatype}_{setdef}'
    
    # Preguntar si es que la carpeta que almacenará los sonidos se ha
    # creado. En caso de que no exista, se crea una carpeta
    if not os.path.isdir(folder_data):
        os.makedirs(folder_data)
    
    # Cargando los archivos de audio
    filenames = os.listdir(filepath)
    
    for audio_name in filenames:
        # Discriminación de archivos por tipo
        if datatype in audio_name:
            print(f'Recording {audio_name}...')
            # Dirección del archivo en la carpeta madre. Este archivo es el que 
            # se copiará
            dir_to_copy = f'{filepath}/{audio_name}'
                              
            # Dirección en la cual se almacenará este nuevo archivo
            dir_to_paste = f"{folder_data}/{audio_name}"
            
            audio, samplerate = sf.read(dir_to_copy)
            
            # Normalizando
            audio = audio / max(abs(audio))
            
            # Re grabando
            sf.write(dir_to_paste, audio, samplerate)
            print('Completed!\n')


def get_labeled_heart_sound_files_heartbeatdb(record=True):
    '''Función que genera una base de datos de sonidos cardiacos basados en
    la carpeta "Heartbeat sounds", eligiendo solamente las señales que están
    etiquetadas en el archivo "set_a_timing.csv". Retorna una lista con los 
    nombres de los archivos de audio etiquetado
    
    Parámetros
    - record: Booleano que indica si el archivo es además generado. Si es True,
              se crea en la carpeta indicada en "folder_data". Si es False, no
    '''
    # Dirección de ubicación de archivos cardíacos
    filepath = f'Heartbeat sounds/set_a'
    
    # Dirección del archivo de etiquetas
    filelabel = f'Heartbeat sounds/set_a_timing.csv'
    
    # Definición de la carpeta a guardar
    folder_data = f'Heartbeat sounds/Generated/normal_a_labeled'
    
    # Preguntar si es que la carpeta que almacenará los sonidos se ha
    # creado. En caso de que no exista, se crea una carpeta
    if not os.path.isdir(folder_data):
        os.makedirs(folder_data)
    
    # Definición del set en el que se almacenarán los nombres
    filenames = set()
    
    with open(filelabel, 'r', encoding='utf8') as file:
        # Saltando la primera línea
        file.readline()
        for line in file:
            # Obtención del nombres del archivo
            audio_name = line.strip().split(',')[0].split('/')[1]
            
            # Agregando al set de datos
            filenames.add(audio_name)
            
    # Una vez completado se transforma este set en una lista ordenada
    filenames_out = list(filenames)
    filenames_out.sort()
    
    if record:
        for audio_name in filenames_out:
            print(f'Recording {audio_name}...')
            # Dirección del archivo en la carpeta madre. Este archivo es el que 
            # se copiará
            dir_to_copy = f'{filepath}/{audio_name}'
                            
            # Dirección en la cual se almacenará este nuevo archivo
            dir_to_paste = f"{folder_data}/{audio_name}"
            
            audio, samplerate = sf.read(dir_to_copy)
            
            # Normalizando
            audio = audio / max(abs(audio))
            
            # Re grabando
            sf.write(dir_to_paste, audio, samplerate)
            print('Completed!\n')
    
    return filenames_out
    

def get_heartbeat_points(filename):
    '''Función que retorna los puntos etiquetados de los sonidos cardíacos
    de la base de datos "Heartbeat sounds", para los sonidos normales del 
    set a.
    
    Parameters
    ----------
    filename : str
        Nombre del archivo a procesar.
    '''
    filepath = 'Heartbeat sounds/set_a_timing.csv'
    
    # Definición de la lista de puntos a obtener
    point_list = list()
    
    with open(filepath, 'r', encoding='utf8') as file:
        # Saltando el encabezado
        file.readline()
        
        # Si el nombre está en la línea, se recupera la ubicación del punto
        for line in file:
            if filename in line:
                # Se formatea
                line_info = line.strip().split(',')
                
                # Se añade a la lista
                point_list.append(int(line_info[-1]))
                
    return point_list


def get_heartbeat_points_created_db(filename):
    '''Función que retorna los puntos etiquetados de los sonidos cardíacos
    de la base de datos creada "Database_manufacturing/db_HR", los cuales 
    se encuentran separados en la carpeta "Database_manufacturing/db_heart"
    en "Manual combinations".
    
    Parameters
    ----------
    filename : str
        Nombre del archivo a procesar.
    '''
    filepath = 'Database_manufacturing/db_heart/Manual combinations'
    
    # Revisión de cada uno de los archivos en la carpeta de interés
    segment_files = [i for i in os.listdir(filepath) if i.endswith('.txt')]
    
    # Revisando cada archivo
    for rev_file in segment_files:
        # Si el nombre está en la carpeta, se recupera la ubicación del punto
        if filename in rev_file:
            with open(f'{filepath}/{rev_file}', 'r', encoding='utf8') as file:
                # Saltando el encabezado
                interval_list = literal_eval(file.readline())
    
    try:
        # Definición de la lista de puntos de interés
        label_points = [int(np.mean(i)) for i in interval_list]
    except:
        raise Exception('No se ha encontrado la lista de intervalos para el audio especificado.')
                
    return label_points



# Test module
#a = get_heartbeat_points_created_db('Seed[47651]_S1[61]_S2[62]')
#print(a)

# get_labeled_heart_sound_files_heartbeatdb(record=True)
# get_heart_sound_files_heartbeatdb()

'''symptom = "Pneumonia"
get_audio_folder_by_symptom(symptom, sep_type='all')
get_audio_folder_by_symptom(symptom, sep_type='tracheal')
get_audio_folder_by_symptom(symptom, sep_type='toracic')
get_segmentation_points_by_filename("Healthy")'''
# import matplotlib.pyplot as plt 
# get_heart_sound_files()

'''sep_types = ['all', 'tracheal', 'toracic']
for i in sep_types:
    get_audio_folder_by_symptom('URTI', sep_type=i)
    get_audio_folder_by_symptom('Asthma', sep_type=i)
    get_audio_folder_by_symptom('COPD', sep_type=i)
    get_audio_folder_by_symptom('LRTI', sep_type=i)
    get_audio_folder_by_symptom('Bronchiectasis', sep_type=i)
    get_audio_folder_by_symptom('Bronchiolitis', sep_type=i)'''
    
# get_heart_sound_by_presence(level=1)
    