# Este script incorpora la posibilidad de poder obtener los indices de train y test 
# de forma externa
__author__ = 'Christian Escobar Arce'

import os, gc
import numpy as np
import tensorflow as tf
from ast import literal_eval
from paper_DNN_models import cnn_dnn_1_1, cnn_dnn_1_2, segnet_based_1_1, segnet_based_1_2, \
    segnet_based_2_1, segnet_based_2_2, segnet_based_2_3, segnet_based_2_4, segnet_based_2_5, \
    segnet_based_2_6, segnet_based_2_7, segnet_based_2_8, segnet_based_2_9, segnet_based_2_10, \
    segnet_based_2_11, segnet_based_2_12, cnn_dnn_1_3, cnn_dnn_1_4, segnet_based_1_3, \
    segnet_based_1_4, segnet_based_2_13, segnet_based_2_14, segnet_based_2_15, segnet_based_2_16, \
    segnet_based_2_17, segnet_based_2_18, segnet_based_2_19, segnet_based_2_20, segnet_based_2_21, \
    segnet_based_1_1_all, segnet_based_1_2_all, segnet_based_1_3_all, segnet_based_1_4_all, \
    segnet_based_3_1, segnet_based_3_2, segnet_based_3_3, segnet_based_3_4, segnet_based_3_5, \
    segnet_based_3_6, segnet_based_3_7, segnet_based_3_8, segnet_based_3_9, segnet_based_3_10, \
    segnet_based_3_11, segnet_based_3_12, segnet_based_3_13, segnet_based_3_14, segnet_based_3_15, \
    segnet_based_4_1, segnet_based_4_2, segnet_based_4_3, segnet_based_4_4, segnet_based_5_x, \
    segnet_based_6_1, segnet_based_6_2, segnet_based_6_3, segnet_based_6_4, segnet_based_6_5, \
    segnet_based_6_6, segnet_based_6_7, segnet_based_6_8, segnet_based_6_9, segnet_based_6_10, \
    segnet_based_7_1, segnet_based_7_2, segnet_based_7_3, segnet_based_7_4, segnet_based_7_5, \
    segnet_based_7_6, segnet_based_7_7, segnet_based_7_8, segnet_based_7_9, segnet_based_7_10, \
    segnet_based_8_x, segnet_based_9_x, segnet_based_10_1, segnet_based_10_2, segnet_based_11_1, \
    segnet_based_11_2, segnet_based_12_x, definitive_segnet_based, segnet_based_2_22, segnet_based_2_23
from heart_sound_physionet_management import get_model_data_idxs, \
    get_model_data_idxs_2, get_training_weights


# Definición de la carpeta con la base de datos
db_folder = 'PhysioNet 2016 CINC Heart Sound Database'


# Función que permitirá iterar sobre cada modelo, sin sobrepasar los límites de memoria
def model_train_iteration(model, model_name, index_list, epoch_train):
    # Definición de los datos de entrenamiento
    if model_name in ['segnet_based_11_1', 'segnet_based_11_2']:
        X_train, Y_train = \
            get_model_data_idxs_2(db_folder, snr_list=snr_list, index_list=index_list, N=N, 
                                noverlap=N-step, padding_value=padding_value, 
                                activation_percentage=activation_percentage, 
                                append_audio=append_audio, 
                                append_envelopes=append_envelopes, 
                                apply_bpfilter=apply_bpfilter, bp_parameters=bp_parameters, 
                                homomorphic_dict=homomorphic_dict, hilbert_dict=hilbert_dict,
                                simplicity_dict=simplicity_dict, vfd_dict=vfd_dict, 
                                multiscale_wavelet_dict=multiscale_wavelet_dict, 
                                spec_track_dict=spec_track_dict, spec_energy_dict=spec_energy_dict,
                                wavelet_dict=wavelet_dict, append_fft=append_fft, 
                                print_indexes=False, return_indexes=True)
            
    else:
        X_train, Y_train = \
            get_model_data_idxs(db_folder, snr_list=snr_list, index_list=index_list, N=N, 
                                noverlap=N-step, padding_value=padding_value, 
                                activation_percentage=activation_percentage, 
                                append_audio=append_audio, 
                                append_envelopes=append_envelopes, 
                                apply_bpfilter=apply_bpfilter, bp_parameters=bp_parameters, 
                                homomorphic_dict=homomorphic_dict, hilbert_dict=hilbert_dict,
                                simplicity_dict=simplicity_dict, vfd_dict=vfd_dict, 
                                multiscale_wavelet_dict=multiscale_wavelet_dict, 
                                spec_track_dict=spec_track_dict, spec_energy_dict=spec_energy_dict,
                                wavelet_dict=wavelet_dict, append_fft=append_fft, 
                                print_indexes=False, return_indexes=True)
    
    # Entrenando
    if model_name in ['Model_6_1', 'Model_6_1_noised', 'Model_6_1_onechannel', 
                      'Model_6_2', 'Model_6_3', 'Model_6_4_onechannel', 
                      'Model_6_4_typicalchannels', 'Model_6_1_pro_total',
                      'Model_8_1_pro_total', 'Model_8_1_pro_total_augmented', 
                      'Model_8_1_pro_total_augmented_2', 'Model_6_1_pro_total_augmented',
                      'Model_8_2_pro_total_augmented', 'Model_8_3_pro_total_augmented',
                      'segnet_based_1_1', 'segnet_based_2_1', 'segnet_based_2_2', 
                      'segnet_based_2_3', 'segnet_based_2_4', 'segnet_based_2_5',
                      'segnet_based_2_6', 'segnet_based_2_7', 'segnet_based_2_8',
                      'segnet_based_2_9', 'segnet_based_2_10', 'segnet_based_2_11',
                      'segnet_based_2_12', 'segnet_based_1_3', 'segnet_based_2_13',
                      'segnet_based_2_14', 'segnet_based_2_15', 'segnet_based_2_16',
                      'segnet_based_2_17', 'segnet_based_2_18', 'segnet_based_2_19',
                      'segnet_based_2_20', 'segnet_based_2_21', 'segnet_based_1_1_all',
                      'segnet_based_1_3_all', 'segnet_based_3_1', 'segnet_based_3_2',
                      'segnet_based_3_3', 'segnet_based_3_4', 'segnet_based_3_5',
                      'segnet_based_3_6', 'segnet_based_3_7', 'segnet_based_3_8',
                      'segnet_based_3_9', 'segnet_based_3_10', 'segnet_based_3_11',
                      'segnet_based_3_12', 'segnet_based_3_13', 'segnet_based_3_14',
                      'segnet_based_3_15', 'segnet_based_4_1', 'segnet_based_4_2',
                      'segnet_based_4_3', 'segnet_based_4_4', 'segnet_based_5_1',
                      'segnet_based_5_2', 'segnet_based_5_3', 'segnet_based_5_4',
                      'segnet_based_5_5', 'segnet_based_5_6', 'segnet_based_5_7',
                      'segnet_based_6_1', 'segnet_based_6_2', 'segnet_based_6_3',
                      'segnet_based_6_4', 'segnet_based_6_5', 'segnet_based_6_6',
                      'segnet_based_6_7', 'segnet_based_6_8', 'segnet_based_6_9',
                      'segnet_based_6_10', 'segnet_based_7_1', 'segnet_based_7_2', 
                      'segnet_based_7_3', 'segnet_based_7_4', 'segnet_based_7_5', 
                      'segnet_based_7_6', 'segnet_based_7_7', 'segnet_based_7_8',
                      'segnet_based_7_9', 'segnet_based_7_10', 'segnet_based_8_1', 
                      'segnet_based_8_2', 'segnet_based_8_3', 'segnet_based_8_4', 
                      'segnet_based_8_5', 'segnet_based_8_6', 'segnet_based_8_7', 
                      'segnet_based_8_8', 'segnet_based_8_9', 'segnet_based_8_10', 
                      'segnet_based_8_11', 'segnet_based_8_12', 'segnet_based_8_13',
                      'segnet_based_9_1', 'segnet_based_10_1', 'segnet_based_10_2',
                      'segnet_based_12_1', 'segnet_based_12_2', 'segnet_based_12_3',
                      'segnet_based_12_4', 'segnet_based_12_5', 'segnet_based_12_6',
                      'segnet_based_12_7', 'segnet_based_12_8', 'segnet_based_12_9',
                      'segnet_based_12_10', 'definitive_segnet_based', 'segnet_based_2_22',
                      'segnet_based_2_23']:
        print('\nTraining time\n------------\n')
        # Definición de las etiquetas de entrenamiento
        y1 = Y_train[:, :, 0]
        y2 = Y_train[:, :, 1]
        y0 = np.ones(Y_train.shape[:-1]) - y1 - y2

        # Acondicionando las etiquetas para entrenar el modelo
        y0 = np.expand_dims(y0, -1)     # Segmentos intermedios
        y1 = np.expand_dims(y1, -1)     # S1
        y2 = np.expand_dims(y2, -1)     # S2

        # Concatenando las etiquetas para el modelo
        y_to = np.concatenate((y0, y1, y2), axis=-1)
        
        # Entrenando
        history = model.fit(x=X_train, y=y_to, epochs=epoch_train, batch_size=batch_size, 
                            verbose=1)

    elif model_name in ['segnet_based_11_1']:
        print('\nTraining time\n------------\n')
        # Definición de las etiquetas de testeo (solo 2 en este caso)
        s1_labels = (Y_train == 1)
        s2_labels = (Y_train == 3)
        sys_labels = (Y_train == 2)
        dia_labels = (Y_train == 4)
        
        # Definición de las etiquetas
        y1 = s1_labels + s2_labels
        y0 = sys_labels + dia_labels
        
        # Concatenando las etiquetas para el modelo
        y_to = np.concatenate((y0, y1), axis=-1)
        
        # Evaluando
        history = model.fit(x=X_train, y=y_to, epochs=epoch_train, batch_size=batch_size, 
                            verbose=1)
    
    
    elif model_name in ['segnet_based_11_2']:
        print('\nTraining time\n------------\n')
        # Definición de las etiquetas de testeo (solo 2 en este caso)
        y0 = (Y_train == 1)
        y1 = (Y_train == 2)
        y2 = (Y_train == 3)
        y3 = (Y_train == 4)
        
        # Concatenando las etiquetas para el modelo
        y_to = np.concatenate((y0, y1, y2, y3), axis=-1)
        
        # Evaluando
        history = model.fit(x=X_train, y=y_to, epochs=epoch_train, batch_size=batch_size, 
                            verbose=1)
    
    
    elif model_name in ['Model_7_1_pro_total', 'Model_7_2_pro_total', 'Model_7_1_pro_total_epochs',
                        'segnet_based_1_2', 'segnet_based_1_4', 'segnet_based_1_2_all',
                        'segnet_based_1_4_all']:
        print('\nTraining time\n------------\n')
        # Definición de las etiquetas de entrenamiento
        y1 = Y_train[:, :, 0]
        y2 = Y_train[:, :, 1]
        y0 = np.ones(Y_train.shape[:-1]) - y1 - y2

        # Acondicionando las etiquetas para entrenar el modelo
        y0 = np.expand_dims(y0, -1)     # Segmentos intermedios
        y1 = np.expand_dims(y1, -1)     # S1
        y2 = np.expand_dims(y2, -1)     # S2

        # Concatenando las etiquetas para el modelo
        y_to = np.concatenate((y0, y1, y2), axis=-1)
        
        # Entrenando
        history = model.fit(x=[X_train[:, :, i] for i in range(X_train.shape[2])], 
                            y=y_to, epochs=epoch_train, batch_size=batch_size, 
                            verbose=1)


    elif model_name in ['cnn_dnn_1_1', 'cnn_dnn_1_3']:
        print('\nTraining time\n------------\n')
        # Definición de las etiquetas de testeo
        y1 = Y_train[:, 0]
        y2 = Y_train[:, 1]
        y0 = np.ones(Y_train.shape[0]) - y1 - y2

        # Acondicionando las etiquetas para testear el modelo
        y0 = np.expand_dims(y0, -1)     # Segmentos intermedios
        y1 = np.expand_dims(y1, -1)     # S1
        y2 = np.expand_dims(y2, -1)     # S2

        # Concatenando las etiquetas para el modelo
        y_to = np.concatenate((y0, y1, y2), axis=-1)

        # Entrenando
        history = model.fit(x=X_train, y=y_to, epochs=epoch_train, batch_size=batch_size, 
                            verbose=1)
    
    
    elif model_name in ['cnn_dnn_1_2', 'cnn_dnn_1_4']:
        print('\nTraining time\n------------\n')
        # Definición de las etiquetas de testeo
        y1 = Y_train[:, 0]
        y2 = Y_train[:, 1]
        y0 = np.ones(Y_train.shape[0]) - y1 - y2

        # Acondicionando las etiquetas para testear el modelo
        y0 = np.expand_dims(y0, -1)     # Segmentos intermedios
        y1 = np.expand_dims(y1, -1)     # S1
        y2 = np.expand_dims(y2, -1)     # S2

        # Concatenando las etiquetas para el modelo
        y_to = np.concatenate((y0, y1, y2), axis=-1)

        # Entrenando
        history = model.fit(x=[X_train[:, :, i] for i in range(X_train.shape[2])], 
                            y=y_to, epochs=epoch_train, batch_size=batch_size, 
                            verbose=1)
    
    
    # Y guardando la información del entrenamiento con el testeo
    with open(f'{filepath_to_save}/{model_name}.txt', 'a', encoding='utf8') as file:
        file.write(f'{history.history}\n')
    
    # Eliminando las variables de entrenamiento de la memoria
    del Y_train
    del X_train
    
    # Eliminando las variables registradas que no se referencian en memoria
    print("Recolectando registros de memoria sin uso...")
    n = gc.collect()
    print("Número de objetos inalcanzables recolectados por el GC:", n)
    print("Basura incoleccionable:", gc.garbage)
    
    return model


def model_bigbatch_iteration(model, model_name, train_list_iter, big_batch_size,
                             epoch):
    # Definición de una lista de iteración auxiliar para cada loop
    # train_list_iter = train_list
    
    # Realizando las iteraciones
    while train_list_iter.size > 0:
        # Selección de archivos
        train_sel = train_list_iter[:big_batch_size]
        
        # Cortando los archivos seleccionados
        if big_batch_size is None:
            train_list_iter = train_list_iter[:0]
        else:
            train_list_iter = train_list_iter[big_batch_size:]
        
        # Mensaje de progreso
        print(f'Epoch {epoch+1}: Faltan {train_list_iter.size} sonidos por procesar...\n')
        
        # Aplicando la iteración
        model = model_train_iteration(model, model_name, train_sel, epoch_train=1)
        # print(train_sel)
               
    return model


def model_bigbatch_evaluation(model, model_name, index_list, epoch, type_op):
    if type_op == 'val':
        to_print = '\nValidation time\n---------------\n'
        print('Etapa de validación\n-------------------')
        snr_list_to = snr_list
    elif type_op == 'test':
        to_print = '\nTesting time\n------------\n'
        print('Etapa de testeo\n---------------')
        snr_list_to = list()
    
    # Definición de los datos de validación
    if model_name in ['segnet_based_11_1', 'segnet_based_11_2']:
        X_data, Y_data = \
            get_model_data_idxs_2(db_folder, snr_list=snr_list_to, index_list=index_list, N=N, 
                                noverlap=N-step, padding_value=padding_value, 
                                activation_percentage=activation_percentage, 
                                append_audio=append_audio, 
                                append_envelopes=append_envelopes, 
                                apply_bpfilter=apply_bpfilter, bp_parameters=bp_parameters, 
                                homomorphic_dict=homomorphic_dict, hilbert_dict=hilbert_dict,
                                simplicity_dict=simplicity_dict, 
                                vfd_dict=vfd_dict, multiscale_wavelet_dict=multiscale_wavelet_dict, 
                                spec_track_dict=spec_track_dict, spec_energy_dict=spec_energy_dict,
                                wavelet_dict=wavelet_dict, 
                                append_fft=append_fft, print_indexes=False, return_indexes=True)
    
    else:
        X_data, Y_data = \
            get_model_data_idxs(db_folder, snr_list=snr_list_to, index_list=index_list, N=N, 
                                noverlap=N-step, padding_value=padding_value, 
                                activation_percentage=activation_percentage, 
                                append_audio=append_audio, 
                                append_envelopes=append_envelopes, 
                                apply_bpfilter=apply_bpfilter, bp_parameters=bp_parameters, 
                                homomorphic_dict=homomorphic_dict, hilbert_dict=hilbert_dict,
                                simplicity_dict=simplicity_dict, 
                                vfd_dict=vfd_dict, multiscale_wavelet_dict=multiscale_wavelet_dict, 
                                spec_track_dict=spec_track_dict, spec_energy_dict=spec_energy_dict,
                                wavelet_dict=wavelet_dict, 
                                append_fft=append_fft, print_indexes=False, return_indexes=True)
    
    
    if model_name in ['Model_6_1', 'Model_6_1_noised', 'Model_6_1_onechannel', 'Model_6_2',
                      'Model_6_3', 'Model_6_4_onechannel', 'Model_6_4_typicalchannels', 
                      'Model_6_1_pro_total', 'Model_8_1_pro_total', 'Model_8_1_pro_total_augmented', 
                      'Model_8_1_pro_total_augmented_2', 'Model_6_1_pro_total_augmented',
                      'Model_8_2_pro_total_augmented', 'Model_8_3_pro_total_augmented',
                      'segnet_based_1_1', 'segnet_based_1_3', 'segnet_based_2_1', 
                      'segnet_based_2_2', 'segnet_based_2_3', 'segnet_based_2_4', 
                      'segnet_based_2_5', 'segnet_based_2_6', 'segnet_based_2_7', 
                      'segnet_based_2_8', 'segnet_based_2_9', 'segnet_based_2_10', 
                      'segnet_based_2_11', 'segnet_based_2_12', 'segnet_based_2_13', 
                      'segnet_based_2_14', 'segnet_based_2_15', 'segnet_based_2_16',
                      'segnet_based_2_17', 'segnet_based_2_18', 'segnet_based_2_19',
                      'segnet_based_2_20', 'segnet_based_2_21', 'segnet_based_1_1_all',
                      'segnet_based_1_3_all', 'segnet_based_3_1', 'segnet_based_3_2',
                      'segnet_based_3_3', 'segnet_based_3_4', 'segnet_based_3_5',
                      'segnet_based_3_6', 'segnet_based_3_7', 'segnet_based_3_8',
                      'segnet_based_3_9', 'segnet_based_3_10', 'segnet_based_3_11',
                      'segnet_based_3_12', 'segnet_based_3_13', 'segnet_based_3_14',
                      'segnet_based_3_15', 'segnet_based_4_1', 'segnet_based_4_2',
                      'segnet_based_4_3', 'segnet_based_4_4', 'segnet_based_5_1',
                      'segnet_based_5_2', 'segnet_based_5_3', 'segnet_based_5_4',
                      'segnet_based_5_5', 'segnet_based_5_6', 'segnet_based_5_7',
                      'segnet_based_6_1', 'segnet_based_6_2', 'segnet_based_6_3',
                      'segnet_based_6_4', 'segnet_based_6_5', 'segnet_based_6_6',
                      'segnet_based_6_7', 'segnet_based_6_8', 'segnet_based_6_9',
                      'segnet_based_6_10', 'segnet_based_7_1', 'segnet_based_7_2',
                      'segnet_based_7_3', 'segnet_based_7_4', 'segnet_based_7_5', 
                      'segnet_based_7_6', 'segnet_based_7_7', 'segnet_based_7_8',
                      'segnet_based_7_9', 'segnet_based_7_10', 'segnet_based_8_1', 
                      'segnet_based_8_2', 'segnet_based_8_3', 'segnet_based_8_4', 
                      'segnet_based_8_5', 'segnet_based_8_6', 'segnet_based_8_7', 
                      'segnet_based_8_8', 'segnet_based_8_9', 'segnet_based_8_10', 
                      'segnet_based_8_11', 'segnet_based_8_12', 'segnet_based_8_13',
                      'segnet_based_9_1', 'segnet_based_10_1', 'segnet_based_10_2',
                      'segnet_based_12_1', 'segnet_based_12_2', 'segnet_based_12_3',
                      'segnet_based_12_4', 'segnet_based_12_5', 'segnet_based_12_6',
                      'segnet_based_12_7', 'segnet_based_12_8', 'segnet_based_12_9',
                      'segnet_based_12_10', 'definitive_segnet_based', 'segnet_based_2_22',
                      'segnet_based_2_23']:
        print(to_print)
        # Definición de las etiquetas de testeo
        y1 = Y_data[:, :, 0]
        y2 = Y_data[:, :, 1]
        y0 = np.ones(Y_data.shape[:-1]) - y1 - y2

        # Acondicionando las etiquetas para testear el modelo
        y0 = np.expand_dims(y0, -1)     # Segmentos intermedios
        y1 = np.expand_dims(y1, -1)     # S1
        y2 = np.expand_dims(y2, -1)     # S2

        # Concatenando las etiquetas para el modelo
        y_to = np.concatenate((y0, y1, y2), axis=-1)
        
        # Evaluando
        eval_info = model.evaluate(x=X_data, y=y_to, verbose=1, return_dict=True)


    elif model_name in ['segnet_based_11_1']:
        print(to_print)
        # Definición de las etiquetas de testeo (solo 2 en este caso)
        s1_labels = (Y_data == 1)
        s2_labels = (Y_data == 3)
        sys_labels = (Y_data == 2)
        dia_labels = (Y_data == 4)
        
        # Definición de las etiquetas
        y1 = s1_labels + s2_labels
        y0 = sys_labels + dia_labels
        
        # Concatenando las etiquetas para el modelo
        y_to = np.concatenate((y0, y1), axis=-1)
        
        # Evaluando
        eval_info = model.evaluate(x=X_data, y=y_to, verbose=1, return_dict=True)
        
    elif model_name in ['segnet_based_11_2']:
        print(to_print)
        # Definición de las etiquetas de testeo (solo 2 en este caso)
        y0 = (Y_data == 1)
        y1 = (Y_data == 2)
        y2 = (Y_data == 3)
        y3 = (Y_data == 4)
        
        # Concatenando las etiquetas para el modelo
        y_to = np.concatenate((y0, y1, y2, y3), axis=-1)
        
        # Evaluando
        eval_info = model.evaluate(x=X_data, y=y_to, verbose=1, return_dict=True)
        
        
    
    elif model_name in ['Model_7_1_pro_total', 'Model_7_2_pro_total', 'Model_7_1_pro_total_epochs',
                        'segnet_based_1_2', 'segnet_based_1_4', 'segnet_based_1_2_all',
                        'segnet_based_1_4_all']:
        print(to_print)
        # Definición de las etiquetas de testeo
        y1 = Y_data[:, :, 0]
        y2 = Y_data[:, :, 1]
        y0 = np.ones(Y_data.shape[:-1]) - y1 - y2

        # Acondicionando las etiquetas para testear el modelo
        y0 = np.expand_dims(y0, -1)     # Segmentos intermedios
        y1 = np.expand_dims(y1, -1)     # S1
        y2 = np.expand_dims(y2, -1)     # S2

        # Concatenando las etiquetas para el modelo
        y_to = np.concatenate((y0, y1, y2), axis=-1)
        
        # Evaluando
        eval_info = model.evaluate(x=[X_data[:, :, i] for i in range(X_data.shape[2])], 
                                y=y_to, verbose=1, return_dict=True)


    elif model_name in ['cnn_dnn_1_1', 'cnn_dnn_1_3']:
        print(to_print)
        # Definición de las etiquetas de testeo
        y1 = Y_data[:, 0]
        y2 = Y_data[:, 1]
        y0 = np.ones(Y_data.shape[0]) - y1 - y2

        # Acondicionando las etiquetas para testear el modelo
        y0 = np.expand_dims(y0, -1)     # Segmentos intermedios
        y1 = np.expand_dims(y1, -1)     # S1
        y2 = np.expand_dims(y2, -1)     # S2

        # Concatenando las etiquetas para el modelo
        y_to = np.concatenate((y0, y1, y2), axis=-1)

        # Evaluando
        eval_info = model.evaluate(x=X_data, y=y_to, verbose=1, return_dict=True)


    elif model_name in ['cnn_dnn_1_2', 'cnn_dnn_1_4']:
        print(to_print)
        # Definición de las etiquetas de testeo
        y1 = Y_data[:, 0]
        y2 = Y_data[:, 1]
        y0 = np.ones(Y_data.shape[0]) - y1 - y2

        # Acondicionando las etiquetas para testear el modelo
        y0 = np.expand_dims(y0, -1)     # Segmentos intermedios
        y1 = np.expand_dims(y1, -1)     # S1
        y2 = np.expand_dims(y2, -1)     # S2

        # Concatenando las etiquetas para el modelo
        y_to = np.concatenate((y0, y1, y2), axis=-1)

        # Evaluando
        eval_info = model.evaluate(x=[X_data[:, :, i] for i in range(X_data.shape[2])], 
                                   y=y_to, verbose=1, return_dict=True)
    
    
    # Y guardando la información del entrenamiento con el testeo
    with open(f'{filepath_to_save}/{model_name}.txt', 'a', encoding='utf8') as file:
        if type_op == 'val':
            file.write(f'Validation_info_epoch_{epoch+1}: {eval_info}\n')
        elif type_op == 'test':
            file.write(f'Testing_info: {eval_info}\n')



###############       Definición de parámetros       ###############

# Definición de la GPU con la que se trabajará
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Carpeta a guardar
filepath_to_save = 'Paper_models'

# Parámetros de get_model_data
snr_list = []
big_batch_size = 160 #// 4
padding_value = 2

# Definición de los largos de cada ventana
N_env_vfd = 64
step_env_vfd = 8
N_env_spec = 64
step_env_spec = 8
N_env_energy = 128
step_env_energy = 16


# Parámetros filtro pasabanda
apply_bpfilter = True
bp_parameters = [20, 30, 180, 190]

# Parámetros de Red neuronal
validation_split = 0.1
batch_size = 70
epochs = 20
model_name = 'segnet_based_2_23'

# Parámetros de la función objetivo
optimizer = 'Adam'
loss_func = 'categorical_crossentropy'
metrics = ['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
loss_weights = None



# Parámetros de envolvente para cada caso
if model_name == 'cnn_dnn_1_1':
    # Definición de las ventanas a revisar
    N = 128
    step = 16
    activation_percentage = 0.5
    
    # Parámetros de envolvente
    append_audio = False
    append_envelopes = True
    homomorphic_dict = {'cutoff_freq': 10, 'delta_band': 5}
    hilbert_dict = {'analytic_env': True, 'analytic_env_mod': False, 'inst_phase': False, 
                    'inst_freq': False}
    simplicity_dict = None
    vfd_dict = None
    multiscale_wavelet_dict = None
    spec_track_dict = None
    spec_energy_dict = {'band_limits': [30, 120], 'alpha': 1, 'N': N_env_energy, 
                        'noverlap': N_env_energy - step_env_energy, 'padding': 0, 
                        'repeat': 0 , 'window': 'hann'}
    wavelet_dict = {'wavelet': 'db6', 'levels': [4], 'start_level': 0, 'end_level': 4}
    append_fft = False


elif model_name == 'cnn_dnn_1_2':
    # Definición de las ventanas a revisar
    N = 128
    step = 16
    activation_percentage = 0.5
    
    # Parámetros de envolvente
    append_audio = False
    append_envelopes = True
    homomorphic_dict = {'cutoff_freq': 10, 'delta_band': 5}
    hilbert_dict = {'analytic_env': True, 'analytic_env_mod': False, 'inst_phase': False, 
                    'inst_freq': False}
    simplicity_dict = None
    vfd_dict = None
    multiscale_wavelet_dict = None
    spec_track_dict = None
    spec_energy_dict = {'band_limits': [30, 120], 'alpha': 1, 'N': N_env_energy, 
                        'noverlap': N_env_energy - step_env_energy, 'padding': 0, 
                        'repeat': 0 , 'window': 'hann'}
    wavelet_dict = {'wavelet': 'db6', 'levels': [4], 'start_level': 0, 'end_level': 4}
    append_fft = False


elif model_name == 'cnn_dnn_1_3':
    # Definición de las ventanas a revisar
    N = 128
    step = 16
    activation_percentage = 0.5
    
    # Parámetros de envolvente
    append_audio = False
    append_envelopes = True
    homomorphic_dict = {'cutoff_freq': 10, 'delta_band': 5}
    hilbert_dict = {'analytic_env': True, 'analytic_env_mod': False, 'inst_phase': False, 
                    'inst_freq': False}
    simplicity_dict = None
    vfd_dict = None
    multiscale_wavelet_dict = None
    spec_track_dict = None
    spec_energy_dict = {'band_limits': [30, 120], 'alpha': 1, 'N': N_env_energy, 
                        'noverlap': N_env_energy - step_env_energy, 'padding': 0, 
                        'repeat': 0 , 'window': 'hann'}
    wavelet_dict = {'wavelet': 'db6', 'levels': [4], 'start_level': 0, 'end_level': 4}
    append_fft = False


elif model_name == 'cnn_dnn_1_4':
    # Definición de las ventanas a revisar
    N = 128
    step = 16
    activation_percentage = 0.5
    
    # Parámetros de envolvente
    append_audio = False
    append_envelopes = True
    homomorphic_dict = {'cutoff_freq': 10, 'delta_band': 5}
    hilbert_dict = {'analytic_env': True, 'analytic_env_mod': False, 'inst_phase': False, 
                    'inst_freq': False}
    simplicity_dict = None
    vfd_dict = None
    multiscale_wavelet_dict = None
    spec_track_dict = None
    spec_energy_dict = {'band_limits': [30, 120], 'alpha': 1, 'N': N_env_energy, 
                        'noverlap': N_env_energy - step_env_energy, 'padding': 0, 
                        'repeat': 0 , 'window': 'hann'}
    wavelet_dict = {'wavelet': 'db6', 'levels': [4], 'start_level': 0, 'end_level': 4}
    append_fft = False


elif model_name in ['segnet_based_1_1', 'segnet_based_1_1_all']:
    # Definición de las ventanas a revisar
    N = 1024
    step = 64
    activation_percentage = None
    
    # Parámetros de envolvente
    append_audio = False
    append_envelopes = True
    homomorphic_dict = {'cutoff_freq': 10, 'delta_band': 5}
    hilbert_dict = {'analytic_env': True, 'analytic_env_mod': False, 'inst_phase': False, 
                    'inst_freq': False}
    simplicity_dict = None
    vfd_dict = None
    multiscale_wavelet_dict = None
    spec_track_dict = None
    spec_energy_dict = {'band_limits': [30, 120], 'alpha': 1, 'N': N_env_energy, 
                        'noverlap': N_env_energy - step_env_energy, 'padding': 0, 
                        'repeat': 0 , 'window': 'hann'}
    wavelet_dict = {'wavelet': 'db6', 'levels': [4], 'start_level': 0, 'end_level': 4}
    append_fft = False


elif model_name in ['segnet_based_1_2', 'segnet_based_1_2_all']:
    # Definición de las ventanas a revisar
    N = 1024
    step = 64
    activation_percentage = None
    
    # Parámetros de envolvente
    append_audio = False
    append_envelopes = True
    homomorphic_dict = {'cutoff_freq': 10, 'delta_band': 5}
    hilbert_dict = {'analytic_env': True, 'analytic_env_mod': False, 'inst_phase': False, 
                    'inst_freq': False}
    simplicity_dict = None
    vfd_dict = None
    multiscale_wavelet_dict = None
    spec_track_dict = None
    spec_energy_dict = {'band_limits': [30, 120], 'alpha': 1, 'N': N_env_energy, 
                        'noverlap': N_env_energy - step_env_energy, 'padding': 0, 
                        'repeat': 0 , 'window': 'hann'}
    wavelet_dict = {'wavelet': 'db6', 'levels': [4], 'start_level': 0, 'end_level': 4}
    append_fft = False


elif model_name in ['segnet_based_1_3', 'segnet_based_1_3_all']:
    # Definición de las ventanas a revisar
    N = 128
    step = 16
    activation_percentage = None
    
    # Parámetros de envolvente
    append_audio = False
    append_envelopes = True
    homomorphic_dict = {'cutoff_freq': 10, 'delta_band': 5}
    hilbert_dict = {'analytic_env': True, 'analytic_env_mod': False, 'inst_phase': False, 
                    'inst_freq': False}
    simplicity_dict = None
    vfd_dict = None
    multiscale_wavelet_dict = None
    spec_track_dict = None
    spec_energy_dict = {'band_limits': [30, 120], 'alpha': 1, 'N': N_env_energy, 
                        'noverlap': N_env_energy - step_env_energy, 'padding': 0, 
                        'repeat': 0 , 'window': 'hann'}
    wavelet_dict = {'wavelet': 'db6', 'levels': [4], 'start_level': 0, 'end_level': 4}
    append_fft = False


elif model_name in ['segnet_based_1_4', 'segnet_based_1_4_all']:
    # Definición de las ventanas a revisar
    N = 128
    step = 16
    activation_percentage = None
    
    # Parámetros de envolvente
    append_audio = False
    append_envelopes = True
    homomorphic_dict = {'cutoff_freq': 10, 'delta_band': 5}
    hilbert_dict = {'analytic_env': True, 'analytic_env_mod': False, 'inst_phase': False, 
                    'inst_freq': False}
    simplicity_dict = None
    vfd_dict = None
    multiscale_wavelet_dict = None
    spec_track_dict = None
    spec_energy_dict = {'band_limits': [30, 120], 'alpha': 1, 'N': N_env_energy, 
                        'noverlap': N_env_energy - step_env_energy, 'padding': 0, 
                        'repeat': 0 , 'window': 'hann'}
    wavelet_dict = {'wavelet': 'db6', 'levels': [4], 'start_level': 0, 'end_level': 4}
    append_fft = False


elif model_name == 'segnet_based_2_1':
    # Definición de las ventanas a revisar
    N = 1024
    step = 64
    activation_percentage = None
    
    # Parámetros de envolvente
    append_audio = True
    append_envelopes = True
    homomorphic_dict = {'cutoff_freq': 10, 'delta_band': 5}
    hilbert_dict = {'analytic_env': True, 'analytic_env_mod': False, 'inst_phase': False, 
                    'inst_freq': False}
    simplicity_dict = None
    vfd_dict = None
    multiscale_wavelet_dict = None
    spec_track_dict = None
    spec_energy_dict = {'band_limits': [30, 120], 'alpha': 1, 'N': N_env_energy, 
                        'noverlap': N_env_energy - step_env_energy, 'padding': 0, 
                        'repeat': 0 , 'window': 'hann'}
    wavelet_dict = {'wavelet': 'db6', 'levels': [4], 'start_level': 0, 'end_level': 4}
    append_fft = False


elif model_name == 'segnet_based_2_2':
    # Definición de las ventanas a revisar
    N = 1024
    step = 64
    activation_percentage = None
    
    # Parámetros de envolvente
    append_audio = True
    append_envelopes = True
    homomorphic_dict = {'cutoff_freq': 10, 'delta_band': 5}
    hilbert_dict = {'analytic_env': True, 'analytic_env_mod': False, 
                    'inst_phase': False, 'inst_freq': False}
    simplicity_dict = None
    vfd_dict = None
    multiscale_wavelet_dict = None
    spec_track_dict =  {'freq_obj': [40, 60], 'N': N_env_spec, 
                        'noverlap': N_env_spec - step_env_spec, 
                        'padding': 0, 'repeat': 0, 'window': 'hann'}
    spec_energy_dict = {'band_limits': [30, 120], 'alpha': 1, 'N': N_env_energy, 
                        'noverlap': N_env_energy - step_env_energy, 'padding': 0, 
                        'repeat': 0 , 'window': 'hann'}
    wavelet_dict = {'wavelet': 'db6', 'levels': [4], 'start_level': 0, 'end_level': 4}
    append_fft = False


elif model_name == 'segnet_based_2_3':
    # Definición de las ventanas a revisar
    N = 1024
    step = 64
    activation_percentage = None
    
    # Parámetros de envolvente
    append_audio = True
    append_envelopes = True
    homomorphic_dict = {'cutoff_freq': 10, 'delta_band': 5}
    hilbert_dict = {'analytic_env': True, 'analytic_env_mod': False, 
                    'inst_phase': False, 'inst_freq': False}
    simplicity_dict = None
    vfd_dict = {'N': N_env_vfd, 'noverlap': N_env_vfd - step_env_vfd, 'kmin': 4, 
                'kmax': 4, 'step_size_method': 'unit', 'inverse': True}
    multiscale_wavelet_dict = None
    spec_track_dict =  {'freq_obj': [40, 60], 'N': N_env_spec, 
                        'noverlap': N_env_spec - step_env_spec, 
                        'padding': 0, 'repeat': 0, 'window': 'hann'}
    spec_energy_dict = {'band_limits': [30, 120], 'alpha': 1, 'N': N_env_energy, 
                        'noverlap': N_env_energy - step_env_energy, 'padding': 0, 
                        'repeat': 0 , 'window': 'hann'}
    wavelet_dict = {'wavelet': 'db6', 'levels': [4], 'start_level': 0, 'end_level': 4}
    append_fft = False


elif model_name == 'segnet_based_2_4':
    # Definición de las ventanas a revisar
    N = 1024
    step = 64
    activation_percentage = None
    
    # Parámetros de envolvente
    append_audio = True
    append_envelopes = True
    homomorphic_dict = {'cutoff_freq': 10, 'delta_band': 5}
    hilbert_dict = {'analytic_env': True, 'analytic_env_mod': False, 
                    'inst_phase': False, 'inst_freq': False}
    simplicity_dict = None
    vfd_dict = {'N': N_env_vfd, 'noverlap': N_env_vfd - step_env_vfd, 'kmin': 4, 'kmax': 4, 
                'step_size_method': 'unit', 'inverse': True}
    multiscale_wavelet_dict = {'wavelet': 'db6', 'levels': [4,5], 'start_level': 0, 'end_level': 4}
    spec_track_dict =  {'freq_obj': [40, 60], 'N': N_env_spec, 
                        'noverlap': N_env_spec - step_env_spec, 
                        'padding': 0, 'repeat': 0, 'window': 'hann'}
    spec_energy_dict = {'band_limits': [30, 120], 'alpha': 1, 'N': N_env_energy, 
                        'noverlap': N_env_energy - step_env_energy, 'padding': 0, 
                        'repeat': 0 , 'window': 'hann'}
    wavelet_dict = {'wavelet': 'db6', 'levels': [4], 'start_level': 0, 'end_level': 4}
    append_fft = False


elif model_name == 'segnet_based_2_5':
    # Definición de las ventanas a revisar
    N = 1024
    step = 64
    activation_percentage = None
    
    # Parámetros de envolvente
    append_audio = True
    append_envelopes = True
    homomorphic_dict = {'cutoff_freq': 10, 'delta_band': 5}
    hilbert_dict = {'analytic_env': True, 'analytic_env_mod': True, 
                    'inst_phase': False, 'inst_freq': False}
    simplicity_dict = None
    vfd_dict = {'N': N_env_vfd, 'noverlap': N_env_vfd - step_env_vfd, 'kmin': 4, 'kmax': 4, 
                'step_size_method': 'unit', 'inverse': True}
    multiscale_wavelet_dict = {'wavelet': 'db6', 'levels': [4,5], 'start_level': 0, 'end_level': 4}
    spec_track_dict =  {'freq_obj': [40, 60], 'N': N_env_spec, 
                        'noverlap': N_env_spec - step_env_spec, 
                        'padding': 0, 'repeat': 0, 'window': 'hann'}
    spec_energy_dict = {'band_limits': [30, 120], 'alpha': 1, 'N': N_env_energy, 
                        'noverlap': N_env_energy - step_env_energy, 'padding': 0, 
                        'repeat': 0 , 'window': 'hann'}
    wavelet_dict = {'wavelet': 'db6', 'levels': [4], 'start_level': 0, 'end_level': 4}
    append_fft = False


elif model_name == 'segnet_based_2_6':
    # Definición de las ventanas a revisar
    N = 1024
    step = 64
    activation_percentage = None
    
    # Parámetros de envolvente
    append_audio = True
    append_envelopes = True
    homomorphic_dict = None
    hilbert_dict = {'analytic_env': True, 'analytic_env_mod': True, 
                    'inst_phase': False, 'inst_freq': False}
    simplicity_dict = None
    vfd_dict = {'N': N_env_vfd, 'noverlap': N_env_vfd - step_env_vfd, 'kmin': 4, 'kmax': 4, 
                'step_size_method': 'unit', 'inverse': True}
    multiscale_wavelet_dict = {'wavelet': 'db6', 'levels': [4,5], 'start_level': 0, 'end_level': 4}
    spec_track_dict =  {'freq_obj': [40, 60], 'N': N_env_spec, 
                        'noverlap': N_env_spec - step_env_spec, 
                        'padding': 0, 'repeat': 0, 'window': 'hann'}
    spec_energy_dict = {'band_limits': [30, 120], 'alpha': 1, 'N': N_env_energy, 
                        'noverlap': N_env_energy - step_env_energy, 'padding': 0, 
                        'repeat': 0 , 'window': 'hann'}
    wavelet_dict = {'wavelet': 'db6', 'levels': [4], 'start_level': 0, 'end_level': 4}
    append_fft = False


elif model_name == 'segnet_based_2_7':
    # Definición de las ventanas a revisar
    N = 1024
    step = 64
    activation_percentage = None
    
    # Parámetros de envolvente
    append_audio = False
    append_envelopes = True
    homomorphic_dict = None
    hilbert_dict = {'analytic_env': True, 'analytic_env_mod': True, 
                    'inst_phase': False, 'inst_freq': False}
    simplicity_dict = None
    vfd_dict = {'N': N_env_vfd, 'noverlap': N_env_vfd - step_env_vfd, 'kmin': 4, 'kmax': 4, 
                'step_size_method': 'unit', 'inverse': True}
    multiscale_wavelet_dict = {'wavelet': 'db6', 'levels': [4,5], 'start_level': 0, 'end_level': 4}
    spec_track_dict =  {'freq_obj': [40, 60], 'N': N_env_spec, 
                        'noverlap': N_env_spec - step_env_spec, 
                        'padding': 0, 'repeat': 0, 'window': 'hann'}
    spec_energy_dict = {'band_limits': [30, 120], 'alpha': 1, 'N': N_env_energy, 
                        'noverlap': N_env_energy - step_env_energy, 'padding': 0, 
                        'repeat': 0 , 'window': 'hann'}
    wavelet_dict = {'wavelet': 'db6', 'levels': [4], 'start_level': 0, 'end_level': 4}
    append_fft = False


elif model_name == 'segnet_based_2_8':
    # Definición de las ventanas a revisar
    N = 1024
    step = 64
    activation_percentage = None
    
    # Parámetros de envolvente
    append_audio = False
    append_envelopes = True
    homomorphic_dict = None
    hilbert_dict = {'analytic_env': True, 'analytic_env_mod': True, 
                    'inst_phase': False, 'inst_freq': False}
    simplicity_dict = None
    vfd_dict = {'N': N_env_vfd, 'noverlap': N_env_vfd - step_env_vfd, 'kmin': 4, 'kmax': 4, 
                'step_size_method': 'unit', 'inverse': True}
    multiscale_wavelet_dict = {'wavelet': 'db6', 'levels': [4,5], 'start_level': 0, 'end_level': 4}
    spec_track_dict =  None
    spec_energy_dict = {'band_limits': [30, 120], 'alpha': 1, 'N': N_env_energy, 
                        'noverlap': N_env_energy - step_env_energy, 'padding': 0, 
                        'repeat': 0 , 'window': 'hann'}
    wavelet_dict = {'wavelet': 'db6', 'levels': [4], 'start_level': 0, 'end_level': 4}
    append_fft = False


elif model_name == 'segnet_based_2_9':
    # Definición de las ventanas a revisar
    N = 1024
    step = 64
    activation_percentage = None
    
    # Parámetros de envolvente
    append_audio = False
    append_envelopes = True
    homomorphic_dict = None
    hilbert_dict = {'analytic_env': True, 'analytic_env_mod': False, 
                    'inst_phase': False, 'inst_freq': False}
    simplicity_dict = None
    vfd_dict = {'N': N_env_vfd, 'noverlap': N_env_vfd - step_env_vfd, 'kmin': 4, 'kmax': 4, 
                'step_size_method': 'unit', 'inverse': True}
    multiscale_wavelet_dict = {'wavelet': 'db6', 'levels': [4,5], 'start_level': 0, 'end_level': 4}
    spec_track_dict =  None
    spec_energy_dict = {'band_limits': [30, 120], 'alpha': 1, 'N': N_env_energy, 
                        'noverlap': N_env_energy - step_env_energy, 'padding': 0, 
                        'repeat': 0 , 'window': 'hann'}
    wavelet_dict = {'wavelet': 'db6', 'levels': [4], 'start_level': 0, 'end_level': 4}
    append_fft = False


elif model_name == 'segnet_based_2_10':
    # Definición de las ventanas a revisar
    N = 1024
    step = 64
    activation_percentage = None
    
    # Parámetros de envolvente
    append_audio = False
    append_envelopes = True
    homomorphic_dict = None
    hilbert_dict = {'analytic_env': True, 'analytic_env_mod': False, 
                    'inst_phase': False, 'inst_freq': False}
    simplicity_dict = None
    vfd_dict = None
    multiscale_wavelet_dict = {'wavelet': 'db6', 'levels': [4,5], 'start_level': 0, 'end_level': 4}
    spec_track_dict =  None
    spec_energy_dict = {'band_limits': [30, 120], 'alpha': 1, 'N': N_env_energy, 
                        'noverlap': N_env_energy - step_env_energy, 'padding': 0, 
                        'repeat': 0 , 'window': 'hann'}
    wavelet_dict = {'wavelet': 'db6', 'levels': [4], 'start_level': 0, 'end_level': 4}
    append_fft = False


elif model_name == 'segnet_based_2_11':
    # Definición de las ventanas a revisar
    N = 1024
    step = 64
    activation_percentage = None
    
    # Parámetros de envolvente
    append_audio = False
    append_envelopes = True
    homomorphic_dict = None
    hilbert_dict = {'analytic_env': False, 'analytic_env_mod': True, 
                    'inst_phase': False, 'inst_freq': False}
    simplicity_dict = None
    vfd_dict = None
    multiscale_wavelet_dict = {'wavelet': 'db6', 'levels': [4,5], 'start_level': 0, 'end_level': 4}
    spec_track_dict =  None
    spec_energy_dict = {'band_limits': [30, 120], 'alpha': 1, 'N': N_env_energy, 
                        'noverlap': N_env_energy - step_env_energy, 'padding': 0, 
                        'repeat': 0 , 'window': 'hann'}
    wavelet_dict = {'wavelet': 'db6', 'levels': [4], 'start_level': 0, 'end_level': 4}
    append_fft = False


elif model_name == 'segnet_based_2_12':
    # Definición de las ventanas a revisar
    N = 1024
    step = 64
    activation_percentage = None
    
    # Parámetros de envolvente
    append_audio = False
    append_envelopes = True
    homomorphic_dict = {'cutoff_freq': 10, 'delta_band': 5}
    hilbert_dict = {'analytic_env': True, 'analytic_env_mod': True, 
                    'inst_phase': False, 'inst_freq': False}
    simplicity_dict = None
    vfd_dict = None
    multiscale_wavelet_dict = {'wavelet': 'db6', 'levels': [4,5], 'start_level': 0, 'end_level': 4}
    spec_track_dict = None
    spec_energy_dict = None
    wavelet_dict = {'wavelet': 'db6', 'levels': [4], 'start_level': 0, 'end_level': 4}
    append_fft = False


elif model_name == 'segnet_based_2_13':
    # Definición de las ventanas a revisar
    N = 1024
    step = 64
    activation_percentage = None
    
    # Parámetros de envolvente
    append_audio = False
    append_envelopes = True
    homomorphic_dict = None
    hilbert_dict = None
    simplicity_dict = None
    vfd_dict = vfd_dict = {'N': N_env_vfd, 'noverlap': N_env_vfd - step_env_vfd, 'kmin': 4, 'kmax': 4, 
                           'step_size_method': 'unit', 'inverse': True}
    multiscale_wavelet_dict = None
    spec_track_dict = None
    spec_energy_dict = None
    wavelet_dict = None
    append_fft = False


elif model_name == 'segnet_based_2_14':
    # Definición de las ventanas a revisar
    N = 1024
    step = 64
    activation_percentage = None
    
    # Parámetros de envolvente
    append_audio = False
    append_envelopes = True
    homomorphic_dict = None
    hilbert_dict = {'analytic_env': False, 'analytic_env_mod': True, 
                    'inst_phase': False, 'inst_freq': False}
    simplicity_dict = None
    vfd_dict = vfd_dict = {'N': N_env_vfd, 'noverlap': N_env_vfd - step_env_vfd, 'kmin': 4, 'kmax': 4, 
                           'step_size_method': 'unit', 'inverse': True}
    multiscale_wavelet_dict = None
    spec_track_dict = None
    spec_energy_dict = None
    wavelet_dict = None
    append_fft = False


elif model_name == 'segnet_based_2_15':
    # Definición de las ventanas a revisar
    N = 1024
    step = 64
    activation_percentage = None
    
    # Parámetros de envolvente
    append_audio = False
    append_envelopes = True
    homomorphic_dict = None
    hilbert_dict = {'analytic_env': False, 'analytic_env_mod': True, 
                    'inst_phase': False, 'inst_freq': False}
    simplicity_dict = None
    vfd_dict = vfd_dict = {'N': N_env_vfd, 'noverlap': N_env_vfd - step_env_vfd, 'kmin': 4, 'kmax': 4, 
                           'step_size_method': 'unit', 'inverse': True}
    multiscale_wavelet_dict = None
    spec_track_dict = None
    spec_energy_dict = {'band_limits': [30, 120], 'alpha': 1, 'N': N_env_energy, 
                        'noverlap': N_env_energy - step_env_energy, 'padding': 0, 
                        'repeat': 0 , 'window': 'hann'}
    wavelet_dict = None
    append_fft = False


elif model_name == 'segnet_based_2_16':
    # Definición de las ventanas a revisar
    N = 1024
    step = 64
    activation_percentage = None
    
    # Parámetros de envolvente
    append_audio = False
    append_envelopes = True
    homomorphic_dict = None
    hilbert_dict = {'analytic_env': False, 'analytic_env_mod': True, 
                    'inst_phase': False, 'inst_freq': False}
    simplicity_dict = None
    vfd_dict = vfd_dict = {'N': N_env_vfd, 'noverlap': N_env_vfd - step_env_vfd, 'kmin': 4, 'kmax': 4, 
                           'step_size_method': 'unit', 'inverse': True}
    multiscale_wavelet_dict = None
    spec_track_dict =  {'freq_obj': [40], 'N': N_env_spec, 
                        'noverlap': N_env_spec - step_env_spec, 
                        'padding': 0, 'repeat': 0, 'window': 'hann'}
    spec_energy_dict = {'band_limits': [30, 120], 'alpha': 1, 'N': N_env_energy, 
                        'noverlap': N_env_energy - step_env_energy, 'padding': 0, 
                        'repeat': 0 , 'window': 'hann'}
    wavelet_dict = None
    append_fft = False


elif model_name == 'segnet_based_2_17':
    # Definición de las ventanas a revisar
    N = 1024
    step = 64
    activation_percentage = None
    
    # Parámetros de envolvente
    append_audio = False
    append_envelopes = True
    homomorphic_dict = {'cutoff_freq': 10, 'delta_band': 5}
    hilbert_dict = {'analytic_env': False, 'analytic_env_mod': True, 
                    'inst_phase': False, 'inst_freq': False}
    simplicity_dict = None
    vfd_dict = vfd_dict = {'N': N_env_vfd, 'noverlap': N_env_vfd - step_env_vfd, 'kmin': 4, 'kmax': 4, 
                           'step_size_method': 'unit', 'inverse': True}
    multiscale_wavelet_dict = None
    spec_track_dict =  {'freq_obj': [40], 'N': N_env_spec, 
                        'noverlap': N_env_spec - step_env_spec, 
                        'padding': 0, 'repeat': 0, 'window': 'hann'}
    spec_energy_dict = {'band_limits': [30, 120], 'alpha': 1, 'N': N_env_energy, 
                        'noverlap': N_env_energy - step_env_energy, 'padding': 0, 
                        'repeat': 0 , 'window': 'hann'}
    wavelet_dict = None
    append_fft = False


elif model_name == 'segnet_based_2_18':
    # Definición de las ventanas a revisar
    N = 1024
    step = 64
    activation_percentage = None
    
    # Parámetros de envolvente
    append_audio = False
    append_envelopes = True
    homomorphic_dict = {'cutoff_freq': 10, 'delta_band': 5}
    hilbert_dict = {'analytic_env': False, 'analytic_env_mod': True, 
                    'inst_phase': False, 'inst_freq': False}
    simplicity_dict = None
    vfd_dict = vfd_dict = {'N': N_env_vfd, 'noverlap': N_env_vfd - step_env_vfd, 'kmin': 4, 'kmax': 4, 
                           'step_size_method': 'unit', 'inverse': True}
    multiscale_wavelet_dict = None
    spec_track_dict =  {'freq_obj': [40, 60], 'N': N_env_spec, 
                        'noverlap': N_env_spec - step_env_spec, 
                        'padding': 0, 'repeat': 0, 'window': 'hann'}
    spec_energy_dict = {'band_limits': [30, 120], 'alpha': 1, 'N': N_env_energy, 
                        'noverlap': N_env_energy - step_env_energy, 'padding': 0, 
                        'repeat': 0 , 'window': 'hann'}
    wavelet_dict = None
    append_fft = False


elif model_name == 'segnet_based_2_19':
    # Definición de las ventanas a revisar
    N = 1024
    step = 64
    activation_percentage = None
    
    # Parámetros de envolvente
    append_audio = False
    append_envelopes = True
    homomorphic_dict = {'cutoff_freq': 10, 'delta_band': 5}
    hilbert_dict = {'analytic_env': True, 'analytic_env_mod': True, 
                    'inst_phase': False, 'inst_freq': False}
    simplicity_dict = None
    vfd_dict = vfd_dict = {'N': N_env_vfd, 'noverlap': N_env_vfd - step_env_vfd, 'kmin': 4, 'kmax': 4, 
                           'step_size_method': 'unit', 'inverse': True}
    multiscale_wavelet_dict = None
    spec_track_dict =  {'freq_obj': [40, 60], 'N': N_env_spec, 
                        'noverlap': N_env_spec - step_env_spec, 
                        'padding': 0, 'repeat': 0, 'window': 'hann'}
    spec_energy_dict = {'band_limits': [30, 120], 'alpha': 1, 'N': N_env_energy, 
                        'noverlap': N_env_energy - step_env_energy, 'padding': 0, 
                        'repeat': 0 , 'window': 'hann'}
    wavelet_dict = None
    append_fft = False


elif model_name == 'segnet_based_2_20':
    # Definición de las ventanas a revisar
    N = 1024
    step = 64
    activation_percentage = None
    
    # Parámetros de envolvente
    append_audio = False
    append_envelopes = True
    homomorphic_dict = {'cutoff_freq': 10, 'delta_band': 5}
    hilbert_dict = {'analytic_env': True, 'analytic_env_mod': True, 
                    'inst_phase': False, 'inst_freq': False}
    simplicity_dict = None
    vfd_dict = vfd_dict = {'N': N_env_vfd, 'noverlap': N_env_vfd - step_env_vfd, 'kmin': 4, 'kmax': 4, 
                           'step_size_method': 'unit', 'inverse': True}
    multiscale_wavelet_dict = None
    spec_track_dict =  {'freq_obj': [40, 60], 'N': N_env_spec, 
                        'noverlap': N_env_spec - step_env_spec, 
                        'padding': 0, 'repeat': 0, 'window': 'hann'}
    spec_energy_dict = {'band_limits': [30, 120], 'alpha': 1, 'N': N_env_energy, 
                        'noverlap': N_env_energy - step_env_energy, 'padding': 0, 
                        'repeat': 0 , 'window': 'hann'}
    wavelet_dict = {'wavelet': 'db6', 'levels': [4], 'start_level': 0, 'end_level': 4}
    append_fft = False


elif model_name == 'segnet_based_2_21':
    # Definición de las ventanas a revisar
    N = 1024
    step = 64
    activation_percentage = None
    
    # Parámetros de envolvente
    append_audio = False
    append_envelopes = True
    homomorphic_dict = {'cutoff_freq': 10, 'delta_band': 5}
    hilbert_dict = {'analytic_env': True, 'analytic_env_mod': True, 
                    'inst_phase': False, 'inst_freq': False}
    simplicity_dict = None
    vfd_dict = {'N': N_env_vfd, 'noverlap': N_env_vfd - step_env_vfd, 'kmin': 4, 'kmax': 4, 
                'step_size_method': 'unit', 'inverse': True}
    multiscale_wavelet_dict = {'wavelet': 'db6', 'levels': [4,5], 'start_level': 0, 'end_level': 4}
    spec_track_dict =  {'freq_obj': [40, 60], 'N': N_env_spec, 
                        'noverlap': N_env_spec - step_env_spec, 
                        'padding': 0, 'repeat': 0, 'window': 'hann'}
    spec_energy_dict = {'band_limits': [30, 120], 'alpha': 1, 'N': N_env_energy, 
                        'noverlap': N_env_energy - step_env_energy, 'padding': 0, 
                        'repeat': 0 , 'window': 'hann'}
    wavelet_dict = {'wavelet': 'db6', 'levels': [4], 'start_level': 0, 'end_level': 4}
    append_fft = False


elif model_name == 'segnet_based_2_22':
    # Definición de las ventanas a revisar
    N = 1024
    step = 64
    activation_percentage = None
    
    # Parámetros de envolvente
    append_audio = True
    append_envelopes = True
    homomorphic_dict = None
    hilbert_dict = None
    simplicity_dict = None
    vfd_dict = None
    multiscale_wavelet_dict = None
    spec_track_dict =  None
    spec_energy_dict = None
    wavelet_dict = None
    append_fft = False


elif model_name == 'segnet_based_2_23':
    # Definición de las ventanas a revisar
    N = 1024
    step = 64
    activation_percentage = None
    
    # Parámetros de envolvente
    append_audio = False
    append_envelopes = True
    homomorphic_dict = {'cutoff_freq': 10, 'delta_band': 5}
    hilbert_dict = {'analytic_env': True, 'analytic_env_mod': False, 'inst_phase': False, 
                    'inst_freq': False}
    simplicity_dict = None
    vfd_dict = None
    multiscale_wavelet_dict = None
    spec_track_dict = None
    spec_energy_dict = {'band_limits': [30, 120], 'alpha': 1, 'N': N_env_energy, 
                        'noverlap': N_env_energy - step_env_energy, 'padding': 0, 
                        'repeat': 0 , 'window': 'hann'}
    wavelet_dict = {'wavelet': 'db6', 'levels': [4], 'start_level': 0, 'end_level': 4}
    append_fft = False


elif model_name in ['segnet_based_3_1', 'segnet_based_3_2', 'segnet_based_3_3',
                    'segnet_based_3_4', 'segnet_based_3_5', 'segnet_based_3_6',
                    'segnet_based_3_7', 'segnet_based_3_8', 'segnet_based_3_9',
                    'segnet_based_3_10', 'segnet_based_3_11', 'segnet_based_3_12',
                    'segnet_based_3_13', 'segnet_based_3_14', 'segnet_based_3_15']:
    # Definición de las ventanas a revisar
    N = 1024
    step = 64
    activation_percentage = None
    
    # Parámetros de envolvente
    append_audio = True
    append_envelopes = True
    homomorphic_dict = {'cutoff_freq': 10, 'delta_band': 5}
    hilbert_dict = {'analytic_env': True, 'analytic_env_mod': True, 
                    'inst_phase': False, 'inst_freq': False}
    simplicity_dict = None
    vfd_dict = {'N': N_env_vfd, 'noverlap': N_env_vfd - step_env_vfd, 'kmin': 4, 'kmax': 4, 
                'step_size_method': 'unit', 'inverse': True}
    multiscale_wavelet_dict = {'wavelet': 'db6', 'levels': [4,5], 'start_level': 0, 'end_level': 4}
    spec_track_dict =  {'freq_obj': [40, 60], 'N': N_env_spec, 
                        'noverlap': N_env_spec - step_env_spec, 
                        'padding': 0, 'repeat': 0, 'window': 'hann'}
    spec_energy_dict = {'band_limits': [30, 120], 'alpha': 1, 'N': N_env_energy, 
                        'noverlap': N_env_energy - step_env_energy, 'padding': 0, 
                        'repeat': 0 , 'window': 'hann'}
    wavelet_dict = {'wavelet': 'db6', 'levels': [4], 'start_level': 0, 'end_level': 4}
    append_fft = False


elif model_name in ['segnet_based_4_1', 'segnet_based_4_2', 'segnet_based_4_3',
                    'segnet_based_4_4']:
    # Definición de las ventanas a revisar
    N = 1024
    step = 64
    activation_percentage = None
    
    # Parámetros de envolvente
    append_audio = True
    append_envelopes = True
    homomorphic_dict = {'cutoff_freq': 10, 'delta_band': 5}
    hilbert_dict = {'analytic_env': True, 'analytic_env_mod': True, 
                    'inst_phase': False, 'inst_freq': False}
    simplicity_dict = None
    vfd_dict = {'N': N_env_vfd, 'noverlap': N_env_vfd - step_env_vfd, 'kmin': 4, 'kmax': 4, 
                'step_size_method': 'unit', 'inverse': True}
    multiscale_wavelet_dict = {'wavelet': 'db6', 'levels': [4,5], 'start_level': 0, 'end_level': 4}
    spec_track_dict =  {'freq_obj': [40, 60], 'N': N_env_spec, 
                        'noverlap': N_env_spec - step_env_spec, 
                        'padding': 0, 'repeat': 0, 'window': 'hann'}
    spec_energy_dict = {'band_limits': [30, 120], 'alpha': 1, 'N': N_env_energy, 
                        'noverlap': N_env_energy - step_env_energy, 'padding': 0, 
                        'repeat': 0 , 'window': 'hann'}
    wavelet_dict = {'wavelet': 'db6', 'levels': [4], 'start_level': 0, 'end_level': 4}
    append_fft = False


elif model_name in ['segnet_based_5_1', 'segnet_based_5_2', 'segnet_based_5_3',
                    'segnet_based_5_4', 'segnet_based_5_5', 'segnet_based_5_6',
                    'segnet_based_5_7']:
    
    if model_name == 'segnet_based_5_1':
        snr_list = [5]
        
    elif model_name == 'segnet_based_5_2':
        snr_list = [1, 5]
        
    elif model_name == 'segnet_based_5_3':
        snr_list = [0, 1, 5]
        
    elif model_name == 'segnet_based_5_4':
        snr_list = [-1, 0, 1, 5]
        
    elif model_name == 'segnet_based_5_5':
        snr_list = [-1, 0, 1]
        
    elif model_name == 'segnet_based_5_6':
        snr_list = [-1, 0]
        
    elif model_name == 'segnet_based_5_7':
        snr_list = [-1]
 
    
    # Definición de las ventanas a revisar
    N = 1024
    step = 64
    activation_percentage = None
    
    # Parámetros de envolvente
    append_audio = True
    append_envelopes = True
    homomorphic_dict = {'cutoff_freq': 10, 'delta_band': 5}
    hilbert_dict = {'analytic_env': True, 'analytic_env_mod': True, 
                    'inst_phase': False, 'inst_freq': False}
    simplicity_dict = None
    vfd_dict = {'N': N_env_vfd, 'noverlap': N_env_vfd - step_env_vfd, 'kmin': 4, 'kmax': 4, 
                'step_size_method': 'unit', 'inverse': True}
    multiscale_wavelet_dict = {'wavelet': 'db6', 'levels': [4,5], 'start_level': 0, 'end_level': 4}
    spec_track_dict =  {'freq_obj': [40, 60], 'N': N_env_spec, 
                        'noverlap': N_env_spec - step_env_spec, 
                        'padding': 0, 'repeat': 0, 'window': 'hann'}
    spec_energy_dict = {'band_limits': [30, 120], 'alpha': 1, 'N': N_env_energy, 
                        'noverlap': N_env_energy - step_env_energy, 'padding': 0, 
                        'repeat': 0 , 'window': 'hann'}
    wavelet_dict = {'wavelet': 'db6', 'levels': [4], 'start_level': 0, 'end_level': 4}
    append_fft = False


elif model_name in ['segnet_based_6_1', 'segnet_based_6_2', 'segnet_based_6_3',
                    'segnet_based_6_4', 'segnet_based_6_5', 'segnet_based_6_6',
                    'segnet_based_6_7', 'segnet_based_6_8', 'segnet_based_6_9',
                    'segnet_based_6_10']:
    # Definición de las ventanas a revisar
    N = 1024
    step = 64
    activation_percentage = None
    
    # Parámetros de envolvente
    append_audio = True
    append_envelopes = True
    homomorphic_dict = {'cutoff_freq': 10, 'delta_band': 5}
    hilbert_dict = {'analytic_env': True, 'analytic_env_mod': True, 
                    'inst_phase': False, 'inst_freq': False}
    simplicity_dict = None
    vfd_dict = {'N': N_env_vfd, 'noverlap': N_env_vfd - step_env_vfd, 'kmin': 4, 'kmax': 4, 
                'step_size_method': 'unit', 'inverse': True}
    multiscale_wavelet_dict = {'wavelet': 'db6', 'levels': [4,5], 'start_level': 0, 'end_level': 4}
    spec_track_dict =  {'freq_obj': [40, 60], 'N': N_env_spec, 
                        'noverlap': N_env_spec - step_env_spec, 
                        'padding': 0, 'repeat': 0, 'window': 'hann'}
    spec_energy_dict = {'band_limits': [30, 120], 'alpha': 1, 'N': N_env_energy, 
                        'noverlap': N_env_energy - step_env_energy, 'padding': 0, 
                        'repeat': 0 , 'window': 'hann'}
    wavelet_dict = {'wavelet': 'db6', 'levels': [4], 'start_level': 0, 'end_level': 4}
    append_fft = False


elif model_name in ['segnet_based_7_1', 'segnet_based_7_2', 'segnet_based_7_3',
                    'segnet_based_7_4', 'segnet_based_7_5', 'segnet_based_7_6',
                    'segnet_based_7_7', 'segnet_based_7_8', 'segnet_based_7_9',
                    'segnet_based_7_10']:
    # Definición de las ventanas a revisar
    N = 1024
    step = 64
    activation_percentage = None
    
    # Parámetros de envolvente
    append_audio = True
    append_envelopes = True
    homomorphic_dict = {'cutoff_freq': 10, 'delta_band': 5}
    hilbert_dict = {'analytic_env': True, 'analytic_env_mod': True, 
                    'inst_phase': False, 'inst_freq': False}
    simplicity_dict = None
    vfd_dict = {'N': N_env_vfd, 'noverlap': N_env_vfd - step_env_vfd, 'kmin': 4, 'kmax': 4, 
                'step_size_method': 'unit', 'inverse': True}
    multiscale_wavelet_dict = {'wavelet': 'db6', 'levels': [4,5], 'start_level': 0, 'end_level': 4}
    spec_track_dict =  {'freq_obj': [40, 60], 'N': N_env_spec, 
                        'noverlap': N_env_spec - step_env_spec, 
                        'padding': 0, 'repeat': 0, 'window': 'hann'}
    spec_energy_dict = {'band_limits': [30, 120], 'alpha': 1, 'N': N_env_energy, 
                        'noverlap': N_env_energy - step_env_energy, 'padding': 0, 
                        'repeat': 0 , 'window': 'hann'}
    wavelet_dict = {'wavelet': 'db6', 'levels': [4], 'start_level': 0, 'end_level': 4}
    append_fft = False


elif model_name in ['segnet_based_8_1', 'segnet_based_8_2', 'segnet_based_8_3',
                    'segnet_based_8_4', 'segnet_based_8_5', 'segnet_based_8_6',
                    'segnet_based_8_7', 'segnet_based_8_8', 'segnet_based_8_9',
                    'segnet_based_8_10', 'segnet_based_8_11', 'segnet_based_8_12',
                    'segnet_based_8_13']:
    # Definición de las ventanas a revisar
    activation_percentage = None
    
    if model_name == 'segnet_based_8_1':
        N = 256
        step = 64
    
    elif model_name == 'segnet_based_8_2':
        N = 512
        step = 64
    
    elif model_name == 'segnet_based_8_3':
        N = 2048
        step = 64
    
    elif model_name == 'segnet_based_8_4':
        N = 4096
        step = 64
        
    elif model_name == 'segnet_based_8_5':
        N = 8192
        step = 64
    
    elif model_name == 'segnet_based_8_6':
        N = 8192 * 2
        step = 256
    
    elif model_name == 'segnet_based_8_7':
        N = 1024
        step = 128
        
    elif model_name == 'segnet_based_8_8':
        N = 256
        step = 128
    
    elif model_name == 'segnet_based_8_9':
        N = 512
        step = 128
    
    elif model_name == 'segnet_based_8_10':
        N = 2048
        step = 128
    
    elif model_name == 'segnet_based_8_11':
        N = 4096
        step = 128
    
    elif model_name == 'segnet_based_8_12':
        N = 8192
        step = 128
    
    elif model_name == 'segnet_based_8_13':
        N = 8192 * 2
        step = 128
    
    
    
    # Parámetros de envolvente
    append_audio = True
    append_envelopes = True
    homomorphic_dict = {'cutoff_freq': 10, 'delta_band': 5}
    hilbert_dict = {'analytic_env': True, 'analytic_env_mod': True, 
                    'inst_phase': False, 'inst_freq': False}
    simplicity_dict = None
    vfd_dict = {'N': N_env_vfd, 'noverlap': N_env_vfd - step_env_vfd, 'kmin': 4, 'kmax': 4, 
                'step_size_method': 'unit', 'inverse': True}
    multiscale_wavelet_dict = {'wavelet': 'db6', 'levels': [4,5], 'start_level': 0, 'end_level': 4}
    spec_track_dict =  {'freq_obj': [40, 60], 'N': N_env_spec, 
                        'noverlap': N_env_spec - step_env_spec, 
                        'padding': 0, 'repeat': 0, 'window': 'hann'}
    spec_energy_dict = {'band_limits': [30, 120], 'alpha': 1, 'N': N_env_energy, 
                        'noverlap': N_env_energy - step_env_energy, 'padding': 0, 
                        'repeat': 0 , 'window': 'hann'}
    wavelet_dict = {'wavelet': 'db6', 'levels': [4], 'start_level': 0, 'end_level': 4}
    append_fft = False


elif model_name in ['segnet_based_9_1']:
    # Definición de las ventanas a revisar
    N = 8192 * 2
    step = 128
    activation_percentage = None
    
    # Parámetros de envolvente
    append_audio = True
    append_envelopes = True
    homomorphic_dict = {'cutoff_freq': 10, 'delta_band': 5}
    hilbert_dict = {'analytic_env': True, 'analytic_env_mod': True, 
                    'inst_phase': False, 'inst_freq': False}
    simplicity_dict = None
    vfd_dict = {'N': N_env_vfd, 'noverlap': N_env_vfd - step_env_vfd, 'kmin': 4, 'kmax': 4, 
                'step_size_method': 'unit', 'inverse': True}
    multiscale_wavelet_dict = {'wavelet': 'db6', 'levels': [4,5], 'start_level': 0, 'end_level': 4}
    spec_track_dict =  {'freq_obj': [40, 60], 'N': N_env_spec, 
                        'noverlap': N_env_spec - step_env_spec, 
                        'padding': 0, 'repeat': 0, 'window': 'hann'}
    spec_energy_dict = {'band_limits': [30, 120], 'alpha': 1, 'N': N_env_energy, 
                        'noverlap': N_env_energy - step_env_energy, 'padding': 0, 
                        'repeat': 0 , 'window': 'hann'}
    wavelet_dict = {'wavelet': 'db6', 'levels': [4], 'start_level': 0, 'end_level': 4}
    append_fft = False


elif model_name in ['segnet_based_10_1', 'segnet_based_10_2']:
    # Definición de las ventanas a revisar
    N = 8192 * 2
    step = 128
    activation_percentage = None
    
    # Parámetros de envolvente
    append_audio = True
    append_envelopes = True
    homomorphic_dict = {'cutoff_freq': 10, 'delta_band': 5}
    hilbert_dict = {'analytic_env': True, 'analytic_env_mod': True, 
                    'inst_phase': False, 'inst_freq': False}
    simplicity_dict = None
    vfd_dict = {'N': N_env_vfd, 'noverlap': N_env_vfd - step_env_vfd, 'kmin': 4, 'kmax': 4, 
                'step_size_method': 'unit', 'inverse': True}
    multiscale_wavelet_dict = {'wavelet': 'db6', 'levels': [4,5], 'start_level': 0, 'end_level': 4}
    spec_track_dict =  {'freq_obj': [40, 60], 'N': N_env_spec, 
                        'noverlap': N_env_spec - step_env_spec, 
                        'padding': 0, 'repeat': 0, 'window': 'hann'}
    spec_energy_dict = {'band_limits': [30, 120], 'alpha': 1, 'N': N_env_energy, 
                        'noverlap': N_env_energy - step_env_energy, 'padding': 0, 
                        'repeat': 0 , 'window': 'hann'}
    wavelet_dict = {'wavelet': 'db6', 'levels': [4], 'start_level': 0, 'end_level': 4}
    append_fft = False


elif model_name in ['segnet_based_11_1', 'segnet_based_11_2']:
    # Definición de las ventanas a revisar
    N = 8192 * 2
    step = 128
    activation_percentage = None
    
    # Parámetros de envolvente
    append_audio = True
    append_envelopes = True
    homomorphic_dict = {'cutoff_freq': 10, 'delta_band': 5}
    hilbert_dict = {'analytic_env': True, 'analytic_env_mod': True, 
                    'inst_phase': False, 'inst_freq': False}
    simplicity_dict = None
    vfd_dict = {'N': N_env_vfd, 'noverlap': N_env_vfd - step_env_vfd, 'kmin': 4, 'kmax': 4, 
                'step_size_method': 'unit', 'inverse': True}
    multiscale_wavelet_dict = {'wavelet': 'db6', 'levels': [4,5], 'start_level': 0, 'end_level': 4}
    spec_track_dict =  {'freq_obj': [40, 60], 'N': N_env_spec, 
                        'noverlap': N_env_spec - step_env_spec, 
                        'padding': 0, 'repeat': 0, 'window': 'hann'}
    spec_energy_dict = {'band_limits': [30, 120], 'alpha': 1, 'N': N_env_energy, 
                        'noverlap': N_env_energy - step_env_energy, 'padding': 0, 
                        'repeat': 0 , 'window': 'hann'}
    wavelet_dict = {'wavelet': 'db6', 'levels': [4], 'start_level': 0, 'end_level': 4}
    append_fft = False


elif model_name in ['segnet_based_12_1', 'segnet_based_12_2', 'segnet_based_12_3',
                    'segnet_based_12_4', 'segnet_based_12_5', 'segnet_based_12_6',
                    'segnet_based_12_7', 'segnet_based_12_8', 'segnet_based_12_9',
                    'segnet_based_12_10']:
    # Definición de las ventanas a revisar
    N = 8192 * 2
    step = 128
    activation_percentage = None
    
    # Data augmentation
    snr_list = [-1, 0, 1, 5]
    
    # Parámetros de envolvente
    append_audio = True
    append_envelopes = True
    homomorphic_dict = {'cutoff_freq': 10, 'delta_band': 5}
    hilbert_dict = {'analytic_env': True, 'analytic_env_mod': True, 
                    'inst_phase': False, 'inst_freq': False}
    simplicity_dict = None
    vfd_dict = {'N': N_env_vfd, 'noverlap': N_env_vfd - step_env_vfd, 'kmin': 4, 'kmax': 4, 
                'step_size_method': 'unit', 'inverse': True}
    multiscale_wavelet_dict = {'wavelet': 'db6', 'levels': [3,4], 'start_level': 0, 'end_level': 4}
    spec_track_dict =  {'freq_obj': [40, 60], 'N': N_env_spec, 
                        'noverlap': N_env_spec - step_env_spec, 
                        'padding': 0, 'repeat': 0, 'window': 'hann'}
    spec_energy_dict = {'band_limits': [30, 120], 'alpha': 1, 'N': N_env_energy, 
                        'noverlap': N_env_energy - step_env_energy, 'padding': 0, 
                        'repeat': 0 , 'window': 'hann'}
    wavelet_dict = {'wavelet': 'db6', 'levels': [4], 'start_level': 0, 'end_level': 4}
    append_fft = False


elif model_name in ['definitive_segnet_based']:
    # Definición de las ventanas a revisar
    N = 8192 * 2
    step = 128
    activation_percentage = None
    
    # Data augmentation
    snr_list = [-1, 0, 1, 5]
    
    # Parámetros de envolvente
    append_audio = True
    append_envelopes = True
    homomorphic_dict = {'cutoff_freq': 10, 'delta_band': 5}
    hilbert_dict = {'analytic_env': True, 'analytic_env_mod': True, 
                    'inst_phase': False, 'inst_freq': False}
    simplicity_dict = None
    vfd_dict = {'N': N_env_vfd, 'noverlap': N_env_vfd - step_env_vfd, 'kmin': 4, 'kmax': 4, 
                'step_size_method': 'unit', 'inverse': True}
    multiscale_wavelet_dict = {'wavelet': 'db6', 'levels': [3,4], 'start_level': 0, 'end_level': 4}
    spec_track_dict =  {'freq_obj': [40, 60], 'N': N_env_spec, 
                        'noverlap': N_env_spec - step_env_spec, 
                        'padding': 0, 'repeat': 0, 'window': 'hann'}
    spec_energy_dict = {'band_limits': [30, 120], 'alpha': 1, 'N': N_env_energy, 
                        'noverlap': N_env_energy - step_env_energy, 'padding': 0, 
                        'repeat': 0 , 'window': 'hann'}
    wavelet_dict = {'wavelet': 'db6', 'levels': [4], 'start_level': 0, 'end_level': 4}
    append_fft = False



###############       Definición de parámetros       ###############

# Definición de la lista de índices de entrenamiento, validación y testeo
if model_name in ['segnet_based_12_1', 'segnet_based_12_2', 'segnet_based_12_3',
                  'segnet_based_12_4', 'segnet_based_12_5', 'segnet_based_12_6',
                  'segnet_based_12_7', 'segnet_based_12_8', 'segnet_based_12_9',
                  'segnet_based_12_10']:
    # Definición del índice de la simulación
    index_sim = int(model_name.split('_')[-1])
    
    # Definición de las listas
    train_list = list()
    test_list = list()
    val_list = list()
    
    with open('Cross_validation_groups.txt', 'r', encoding='utf8') as file:
        # Lectura del diccionario
        dict_to_rev = literal_eval(file.readline())
        
        for i in range(1,11):
            if i == index_sim:
                test_list.extend(dict_to_rev[i])
            else:
                train_list.extend(dict_to_rev[i])
    
    # Definiendo los índices para cada conjunto de datos
    train_list = np.array(train_list)
    test_list = np.array(test_list)
    val_list = np.array(val_list)

elif model_name in ['definitive_segnet_based']:
    # Archivos en la carpeta de datos
    files = [i for i in os.listdir(db_folder) if i.endswith('.wav')]
    
    train_list = np.arange(len(files)) + 1
    test_list = None

else:
    with open('Database_separation.txt', 'r', encoding='utf8') as file:
        # Lectura del diccionario
        dict_to_rev = literal_eval(file.readline())
        
        # Definiendo los índices para cada conjunto de datos
        train_list = np.array(dict_to_rev['train_indexes'])
        val_list = np.array(dict_to_rev['val_indexes'])
        test_list = np.array(dict_to_rev['test_indexes'])
        

### Checkeo de versiones ###
for filename in os.listdir(f'{filepath_to_save}/'):
    if model_name in filename:
        check_pass = input(f'Ya existe una simulación de {model_name}. ¿Continuamos?\n' 
                           '[1] Si\n[2] No\nRespuesta: ')
        
        if check_pass == '1':
            break
        elif check_pass == '2':
            print('Simulación terminada.\n')
            exit()
        else:
            print('Opción no válida.')
            exit()


# Definiendo los parámetros especificados para get_model_data
get_model_data_info = {'test_size': 0.1, 'snr_list': snr_list, 
                       'big_batch_size': big_batch_size, 
                       'N': N, 'noverlap': N - step,
                       'padding_value': padding_value, 
                       'activation_percentage': activation_percentage,
                       'append_envelopes': append_envelopes,
                       'apply_bpfilter': apply_bpfilter,
                       'bp_parameters': bp_parameters, 
                       'homomorphic_dict': homomorphic_dict, 
                       'hilbert_dict': hilbert_dict, 
                       'simplicity_dict': simplicity_dict, 
                       'vfd_dict': vfd_dict, 
                       'multiscale_wavelet_dict': multiscale_wavelet_dict,
                       'spec_track_dict': spec_track_dict, 
                       'spec_energy_dict': spec_energy_dict, 
                       'wavelet_dict': wavelet_dict,
                       'append_fft': append_fft}

# Definiendo los parámetros especificados para la función de costo
loss_func_info = {'optimizer': optimizer, 'loss': loss_func, 'metrics': metrics,
                  'loss_weights': loss_weights}

# Finalmente guardando los datos
with open(f'{filepath_to_save}/{model_name}-get_model_data_params.txt', 
          'w', encoding='utf8') as file:
    file.write(f'{get_model_data_info}\n')
    file.write(f'{loss_func_info}')



###### Obtener los shapes #####

# Definición de los datos de entrenamiento y testeo
X_train, Y_train = \
    get_model_data_idxs(db_folder, snr_list=[], index_list=[0], N=N, 
                        noverlap=N-step, padding_value=padding_value, 
                        activation_percentage=activation_percentage, 
                        append_audio=append_audio, 
                        append_envelopes=append_envelopes, 
                        apply_bpfilter=apply_bpfilter, bp_parameters=bp_parameters, 
                        homomorphic_dict=homomorphic_dict, hilbert_dict=hilbert_dict,
                        simplicity_dict=simplicity_dict, 
                        vfd_dict=vfd_dict, 
                        multiscale_wavelet_dict=multiscale_wavelet_dict,
                        spec_track_dict=spec_track_dict, 
                        spec_energy_dict=spec_energy_dict,
                        wavelet_dict=wavelet_dict, 
                        append_fft=append_fft)


# Imprimiendo la dimensión de los archivos
print('Data shapes\n-----------')
print(X_train.shape)
print(Y_train.shape)


# Re Definición de los loss_weights
if model_name == 'segnet_based_9_1':
    print('\n\nGetting loss_weights\n---------------------')
    # Calculando los pesos
    loss_weights_info = \
        get_training_weights(db_folder, big_batch_size, train_list, 
                             N=N, noverlap=N-step, padding_value=padding_value, 
                             activation_percentage=activation_percentage, 
                             append_audio=True, freq_balancing='median')
    # loss_weights_info = {0: 0.22530332559708852, 1: 1.0, 2: 1.399790730671125}
    loss_weights = {'softmax_out': loss_weights_info}
    
    # Eliminando las variables registradas que no se referencian en memoria
    print("Recolectando registros de memoria sin uso...")
    n = gc.collect()
    print("Número de objetos inalcanzables recolectados por el GC:", n)
    print("Basura incoleccionable:", gc.garbage)
    


# Creación del modelo
if model_name in ['cnn_dnn_1_1']:
    model = cnn_dnn_1_1(input_shape=(X_train.shape[1], X_train.shape[2]),
                        padding_value=padding_value, name=model_name)

elif model_name in ['cnn_dnn_1_2']:
    model = cnn_dnn_1_2(input_shape=(X_train.shape[1], X_train.shape[2]),
                        padding_value=padding_value, name=model_name)

elif model_name in ['cnn_dnn_1_3']:
    model = cnn_dnn_1_3(input_shape=(X_train.shape[1], X_train.shape[2]),
                        padding_value=padding_value, name=model_name)

elif model_name in ['cnn_dnn_1_4']:
    model = cnn_dnn_1_4(input_shape=(X_train.shape[1], X_train.shape[2]),
                        padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_1_1']:
    model = segnet_based_1_1(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_1_1_all']:
    model = segnet_based_1_1_all(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_1_2']:
    model = segnet_based_1_2(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_1_2_all']:
    model = segnet_based_1_2_all(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name) 
    
elif model_name in ['segnet_based_1_3']:
    model = segnet_based_1_3(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_1_3_all']:
    model = segnet_based_1_3_all(input_shape=(X_train.shape[1], X_train.shape[2]),
                                 padding_value=padding_value, name=model_name) 

elif model_name in ['segnet_based_1_4']:
    model = segnet_based_1_4(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_1_4_all']:
    model = segnet_based_1_4_all(input_shape=(X_train.shape[1], X_train.shape[2]),
                                 padding_value=padding_value, name=model_name) 

elif model_name in ['segnet_based_2_1']:
    model = segnet_based_2_1(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_2_2']:
    model = segnet_based_2_2(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_2_3']:
    model = segnet_based_2_3(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_2_4']:
    model = segnet_based_2_4(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_2_5']:
    model = segnet_based_2_5(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_2_6']:
    model = segnet_based_2_6(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_2_7']:
    model = segnet_based_2_7(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_2_8']:
    model = segnet_based_2_8(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_2_9']:
    model = segnet_based_2_9(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_2_10']:
    model = segnet_based_2_10(input_shape=(X_train.shape[1], X_train.shape[2]),
                              padding_value=padding_value, name=model_name)
    
elif model_name in ['segnet_based_2_11']:
    model = segnet_based_2_11(input_shape=(X_train.shape[1], X_train.shape[2]),
                              padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_2_12']:
    model = segnet_based_2_12(input_shape=(X_train.shape[1], X_train.shape[2]),
                              padding_value=padding_value, name=model_name)
    
elif model_name in ['segnet_based_2_13']:
    model = segnet_based_2_13(input_shape=(X_train.shape[1], X_train.shape[2]),
                              padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_2_14']:
    model = segnet_based_2_14(input_shape=(X_train.shape[1], X_train.shape[2]),
                              padding_value=padding_value, name=model_name)
    
elif model_name in ['segnet_based_2_15']:
    model = segnet_based_2_15(input_shape=(X_train.shape[1], X_train.shape[2]),
                              padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_2_16']:
    model = segnet_based_2_16(input_shape=(X_train.shape[1], X_train.shape[2]),
                              padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_2_17']:
    model = segnet_based_2_17(input_shape=(X_train.shape[1], X_train.shape[2]),
                              padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_2_18']:
    model = segnet_based_2_18(input_shape=(X_train.shape[1], X_train.shape[2]),
                              padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_2_19']:
    model = segnet_based_2_19(input_shape=(X_train.shape[1], X_train.shape[2]),
                              padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_2_20']:
    model = segnet_based_2_20(input_shape=(X_train.shape[1], X_train.shape[2]),
                              padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_2_21']:
    model = segnet_based_2_21(input_shape=(X_train.shape[1], X_train.shape[2]),
                              padding_value=padding_value, name=model_name)
    
elif model_name in ['segnet_based_2_22']:
    model = segnet_based_2_22(input_shape=(X_train.shape[1], X_train.shape[2]),
                              padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_2_23']:
    model = segnet_based_2_23(input_shape=(X_train.shape[1], X_train.shape[2]),
                              padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_3_1']:
    model = segnet_based_3_1(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_3_2']:
    model = segnet_based_3_2(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_3_3']:
    model = segnet_based_3_3(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_3_4']:
    model = segnet_based_3_4(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_3_5']:
    model = segnet_based_3_5(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name)
    
elif model_name in ['segnet_based_3_6']:
    model = segnet_based_3_6(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_3_7']:
    model = segnet_based_3_7(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_3_8']:
    model = segnet_based_3_8(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_3_9']:
    model = segnet_based_3_9(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_3_10']:
    model = segnet_based_3_10(input_shape=(X_train.shape[1], X_train.shape[2]),
                              padding_value=padding_value, name=model_name)
    
elif model_name in ['segnet_based_3_11']:
    model = segnet_based_3_11(input_shape=(X_train.shape[1], X_train.shape[2]),
                              padding_value=padding_value, name=model_name)
    
elif model_name in ['segnet_based_3_12']:
    model = segnet_based_3_12(input_shape=(X_train.shape[1], X_train.shape[2]),
                              padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_3_13']:
    model = segnet_based_3_13(input_shape=(X_train.shape[1], X_train.shape[2]),
                              padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_3_14']:
    model = segnet_based_3_14(input_shape=(X_train.shape[1], X_train.shape[2]),
                              padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_3_15']:
    model = segnet_based_3_15(input_shape=(X_train.shape[1], X_train.shape[2]),
                              padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_4_1']:
    model = segnet_based_4_1(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_4_2']:
    model = segnet_based_4_2(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_4_3']:
    model = segnet_based_4_3(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_4_4']:
    model = segnet_based_4_4(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_5_1', 'segnet_based_5_2', 'segnet_based_5_3', 
                    'segnet_based_5_4', 'segnet_based_5_5', 'segnet_based_5_6', 
                    'segnet_based_5_7']:
    model = segnet_based_5_x(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_6_1']:
    model = segnet_based_6_1(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name)
    
elif model_name in ['segnet_based_6_2']:
    model = segnet_based_6_2(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name)
    
elif model_name in ['segnet_based_6_3']:
    model = segnet_based_6_3(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_6_4']:
    model = segnet_based_6_4(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name)
    
elif model_name in ['segnet_based_6_5']:
    model = segnet_based_6_5(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name)
    
elif model_name in ['segnet_based_6_6']:
    model = segnet_based_6_6(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name)
    
elif model_name in ['segnet_based_6_7']:
    model = segnet_based_6_7(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_6_8']:
    model = segnet_based_6_8(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_6_9']:
    model = segnet_based_6_9(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_6_10']:
    model = segnet_based_6_10(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_7_1']:
    model = segnet_based_7_1(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name)
    
elif model_name in ['segnet_based_7_2']:
    model = segnet_based_7_2(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name)
    
elif model_name in ['segnet_based_7_3']:
    model = segnet_based_7_3(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_7_4']:
    model = segnet_based_7_4(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_7_5']:
    model = segnet_based_7_5(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_7_6']:
    model = segnet_based_7_6(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_7_7']:
    model = segnet_based_7_7(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_7_8']:
    model = segnet_based_7_8(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_7_9']:
    model = segnet_based_7_9(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_7_10']:
    model = segnet_based_7_10(input_shape=(X_train.shape[1], X_train.shape[2]),
                              padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_8_1', 'segnet_based_8_2', 'segnet_based_8_3', 
                    'segnet_based_8_4', 'segnet_based_8_5', 'segnet_based_8_6',
                    'segnet_based_8_7', 'segnet_based_8_8', 'segnet_based_8_9', 
                    'segnet_based_8_10', 'segnet_based_8_11', 'segnet_based_8_12', 
                    'segnet_based_8_13']:
    model = segnet_based_8_x(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_9_1']:
    model = segnet_based_9_x(input_shape=(X_train.shape[1], X_train.shape[2]),
                             padding_value=padding_value, name=model_name)  

elif model_name in ['segnet_based_10_1']:
    model = segnet_based_10_1(input_shape=(X_train.shape[1], X_train.shape[2]),
                              padding_value=padding_value, name=model_name)
    
elif model_name in ['segnet_based_10_2']:
    model = segnet_based_10_2(input_shape=(X_train.shape[1], X_train.shape[2]),
                              padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_11_1']:
    model = segnet_based_11_1(input_shape=(X_train.shape[1], X_train.shape[2]),
                              padding_value=padding_value, name=model_name)
    
elif model_name in ['segnet_based_11_2']:
    model = segnet_based_11_2(input_shape=(X_train.shape[1], X_train.shape[2]),
                              padding_value=padding_value, name=model_name)

elif model_name in ['segnet_based_12_1', 'segnet_based_12_2', 'segnet_based_12_3',
                    'segnet_based_12_4', 'segnet_based_12_5', 'segnet_based_12_6',
                    'segnet_based_12_7', 'segnet_based_12_8', 'segnet_based_12_9',
                    'segnet_based_12_10']:
    model = segnet_based_12_x(input_shape=(X_train.shape[1], X_train.shape[2]),
                              padding_value=padding_value, name=model_name)  

elif model_name in ['definitive_segnet_based']:
    model = definitive_segnet_based(input_shape=(X_train.shape[1], X_train.shape[2]),
                                    padding_value=padding_value, name=model_name)


# Compilando las opciones del modelo
if model_name in ['Model_2_1', 'Model_2_1_2', 'Model_2_1_no-noise', 'Model_2_1_hyper-noise',
                  'Model_2_2', 'Model_2_3', 'Model_2_4', 'Model_2_5', 'Model_2_6', 'Model_2_7', 
                  'Model_2_7_2', 'Model_2_8', 'Model_4_1', 'Model_4_2', 'Model_4_3', 'Model_4_4', 
                  'Model_5_1', 'Model_5_1_1', 'Model_5_2_1', 'Model_5_2_2', 'Model_5_2_3', 
                  'Model_5_2_4', 'Model_5_2_4_again', 'Model_5_2_4_1', 'Model_5_2_5', 'Model_5_2_6', 
                  'Model_5_2_7', 'Model_5_2_8', 'Model_5_2_9', 'Model_5_2_9_alt', 'Model_5_2_9_alt_2',
                  'cnn_dnn_1_2']:
    loss_model = [loss_func, loss_func]


elif model_name in ['Model_2_9', 'Model_3', 'Model_6_1', 'Model_6_1_noised', 'Model_6_1_onechannel',
                    'Model_6_2', 'Model_6_3', 'Model_6_4_onechannel', 'Model_6_4_typicalchannels',
                    'Model_5_2_4_2', 'Model_5_2_4_2_pro', 'Model_5_2_4_2_pro_total','Model_5_2_4_3',
                    'Model_6_1_pro_total', 'Model_7_1_pro_total', 'Model_7_2_pro_total',
                    'Model_7_1_pro_total_epochs', 'Model_7_1_2_pro_total', 'Model_5_2_4_3_pro_total',
                    'Model_4_5_pro_total', 'Model_4_5_pro', 'Model_4_5_pro_total_epochs',
                    'Model_8_1_pro_total_augmented', 'Model_8_1_pro_total_augmented_2', 
                    'Model_6_1_pro_total_augmented', 'Model_8_2_pro_total_augmented',
                    'Model_8_3_pro_total_augmented',
                    'cnn_dnn_1_1', 'segnet_based_1_1', 'segnet_based_1_2', 'segnet_based_2_1',
                    'segnet_based_2_2', 'segnet_based_2_3', 'segnet_based_2_4', 'segnet_based_2_5',
                    'segnet_based_2_6', 'segnet_based_2_7', 'segnet_based_2_8', 'segnet_based_2_9',
                    'segnet_based_2_10', 'segnet_based_2_11', 'segnet_based_2_12', 'cnn_dnn_1_3',
                    'cnn_dnn_1_4', 'segnet_based_1_3', 'segnet_based_1_4', 'segnet_based_2_13', 
                    'segnet_based_2_14', 'segnet_based_2_15', 'segnet_based_2_16',
                    'segnet_based_2_17', 'segnet_based_2_18', 'segnet_based_2_19',
                    'segnet_based_2_20', 'segnet_based_2_21', 'segnet_based_1_1_all', 
                    'segnet_based_1_2_all', 'segnet_based_1_3_all', 'segnet_based_1_4_all',
                    'segnet_based_3_1', 'segnet_based_3_2', 'segnet_based_3_3',
                    'segnet_based_3_4', 'segnet_based_3_5', 'segnet_based_3_6',
                    'segnet_based_3_7', 'segnet_based_3_8', 'segnet_based_3_9',
                    'segnet_based_3_10', 'segnet_based_3_11', 'segnet_based_3_12',
                    'segnet_based_3_13', 'segnet_based_3_14', 'segnet_based_3_15',
                    'segnet_based_4_1', 'segnet_based_4_2', 'segnet_based_4_3',
                    'segnet_based_4_4', 'segnet_based_5_1', 'segnet_based_5_2', 
                    'segnet_based_5_3', 'segnet_based_5_4', 'segnet_based_5_5', 
                    'segnet_based_5_6', 'segnet_based_5_7', 'segnet_based_6_1',
                    'segnet_based_6_2', 'segnet_based_6_3', 'segnet_based_6_4',
                    'segnet_based_6_5', 'segnet_based_6_6', 'segnet_based_6_7',
                    'segnet_based_6_8', 'segnet_based_6_9', 'segnet_based_6_10',
                    'segnet_based_7_1', 'segnet_based_7_2', 'segnet_based_7_3',
                    'segnet_based_7_4', 'segnet_based_7_5', 'segnet_based_7_6',
                    'segnet_based_7_7', 'segnet_based_7_8', 'segnet_based_7_9',
                    'segnet_based_7_10', 'segnet_based_8_1', 'segnet_based_8_2', 
                    'segnet_based_8_3', 'segnet_based_8_4', 'segnet_based_8_5', 
                    'segnet_based_8_6', 'segnet_based_8_7', 'segnet_based_8_8', 
                    'segnet_based_8_9', 'segnet_based_8_10', 'segnet_based_8_11', 
                    'segnet_based_8_12', 'segnet_based_8_13', 'segnet_based_9_1',
                    'segnet_based_10_1', 'segnet_based_10_2', 'segnet_based_11_1', 
                    'segnet_based_11_2', 'segnet_based_12_1', 'segnet_based_12_2', 
                    'segnet_based_12_3', 'segnet_based_12_4', 'segnet_based_12_5', 
                    'segnet_based_12_6', 'segnet_based_12_7', 'segnet_based_12_8', 
                    'segnet_based_12_9', 'segnet_based_12_10', 'definitive_segnet_based',
                    'segnet_based_2_22', 'segnet_based_2_23']:
    loss_model = loss_func



# Compilando las opciones
model.compile(optimizer=optimizer, loss=loss_model,
              metrics=metrics, loss_weights=loss_weights)


# Mostrando el resumen
model.summary()

# Mostrando las dimensiones de la entrada
print(X_train.shape)
print(Y_train.shape)

# Y el gráfico del modelo
try:
    tf.keras.utils.plot_model(model, f'{filepath_to_save}/{model_name}.png', 
                              show_shapes=True, expand_nested=True)
    tf.keras.utils.plot_model(model, f'{filepath_to_save}/{model_name}_nested.png', 
                              show_shapes=True, expand_nested=False)
except:
    print('No se pudo graficar los modelos.')



############# Iteraciones por cada big batch #############

# Reseteando el archivo de historial
open(f'{filepath_to_save}/{model_name}.txt', 'w', encoding='utf8').close()
open(f'{filepath_to_save}/{model_name}_db.txt', 'w', encoding='utf8').close()
open(f'{filepath_to_save}/Last_model_reg.txt', 'w', encoding='utf8').close()

# Guardando los archivos de train y test asignados para esta simulación
with open(f'{filepath_to_save}/{model_name}_db.txt', 'a', encoding='utf8') as file:
    file.write(f'Train_indexes: {train_list}\n')
    file.write(f'Test_indexes: {test_list}\n')


# Retorna el modelo ya entrenado
for epoch in range(epochs):
    print(f'\nBig Epoch #{epoch+1}\n-------------\n')
    model = model_bigbatch_iteration(model, model_name, train_list_iter=train_list, 
                                     big_batch_size=big_batch_size, epoch=epoch)

    print('Guardando el modelo...\n\n')
    # Guardando el modelo en cada iteración
    model.save(f'{filepath_to_save}/{model_name}.h5')
    
    # Aplicando la validación
    if model_name not in ['segnet_based_12_1', 'segnet_based_12_2', 'segnet_based_12_3',
                          'segnet_based_12_4', 'segnet_based_12_5', 'segnet_based_12_6',
                          'segnet_based_12_7', 'segnet_based_12_8', 'segnet_based_12_9',
                          'segnet_based_12_10', 'definitive_segnet_based']:
        model_bigbatch_evaluation(model, model_name, index_list=val_list, epoch=epoch, 
                                  type_op='val')

    print(f'\n---------- Fin epoch {epoch+1} ----------\n\n')
    
    # Y guardando la información del entrenamiento con el testeo
    with open(f'{filepath_to_save}/{model_name}.txt', 'a', encoding='utf8') as file:
        file.write('---------------------------------------\n')


############# Testeando #############
if model_name not in ['definitive_segnet_based']:
    model_bigbatch_evaluation(model, model_name, index_list=test_list, epoch=epoch, 
                              type_op='test')
else:
    print("\n¡Felicidades! Su red está completa.\n")
