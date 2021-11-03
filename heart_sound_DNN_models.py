import tensorflow as tf


def model_2_1(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    
    Parameters
    ----------
    input_shape : tuple or list
        Dimensión de un ejemplo de entrada.
    padding_value : int
        Valor con el que se realiza el padding de los últimos segmentos de cada sonido.
    name : str or NoneType, optional
        Nombre del modelo. Por defecto es None.
        
    Returns
    -------
    model : tensorflow.keras.Model
        Modelo final de keras.
    '''
    # Definición de la entrada
    x_in = tf.keras.Input(shape=input_shape, dtype='float32')

    # Definición de la capa de máscara
    x_masked = tf.keras.layers.Masking(mask_value=padding_value, 
                                       input_shape=input_shape)(x_in)

    # Definición de las capas convolucionales
    ### Primer conv ###
    x_conv1 = tf.keras.layers.Conv1D(filters=5, kernel_size=200, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv1')(x_masked)
    x_conv1_norm = tf.keras.layers.BatchNormalization()(x_conv1)

    ### Segundo conv ###
    x_conv2 = tf.keras.layers.Conv1D(filters=10, kernel_size=200, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv2')(x_conv1_norm)
    x_conv2_norm = tf.keras.layers.BatchNormalization()(x_conv2)

    ### Tercer conv ###
    x_conv3 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv3')(x_conv2_norm)
    x_conv3_norm = tf.keras.layers.BatchNormalization()(x_conv3)

    ### Cuarto conv ###
    x_conv4 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv4')(x_conv3_norm)
    x_conv4_norm = tf.keras.layers.BatchNormalization()(x_conv4)

    # Definición de la capa de aplanamiento para conectar la CNN con la FCL 
    x_flat = tf.keras.layers.Flatten()(x_conv4_norm)
    
    
    # Definición de las capas fully connected
    x_full1 = tf.keras.layers.Dense(units=30, activation='relu',
                                    kernel_initializer='glorot_uniform')(x_flat)
    x_full1 = tf.keras.layers.BatchNormalization()(x_full1)

    x_full2 = tf.keras.layers.Dense(units=80, activation='relu',
                                    kernel_initializer='glorot_uniform')(x_full1)
    x_full2 = tf.keras.layers.BatchNormalization()(x_full2)

    x_full3 = tf.keras.layers.Dense(units=70, activation='relu',
                                    kernel_initializer='glorot_uniform')(x_full2)
    x_full3 = tf.keras.layers.BatchNormalization()(x_full3)

    # Definición de la última capa (activación)
    x_out_s1 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='glorot_uniform',
                                     name='S1_out')(x_full3)
    x_out_s2 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='glorot_uniform',
                                     name='S2_out')(x_full3)


    # Definición del modelo
    model = tf.keras.Model(inputs=x_in, outputs=[x_out_s1, x_out_s2], name=name)
    
    return model


def model_2_2(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    
    Parameters
    ----------
    input_shape : tuple or list
        Dimensión de un ejemplo de entrada.
    padding_value : int
        Valor con el que se realiza el padding de los últimos segmentos de cada sonido.
    name : str or NoneType, optional
        Nombre del modelo. Por defecto es None.
        
    Returns
    -------
    model : tensorflow.keras.Model
        Modelo final de keras.
    '''
    # Definición de la entrada
    x_in = tf.keras.Input(shape=input_shape, dtype='float32')

    # Definición de la capa de máscara
    x_masked = tf.keras.layers.Masking(mask_value=padding_value, 
                                       input_shape=input_shape)(x_in)

    # Definición de las capas convolucionales
    ### Primer conv ###
    x_conv1 = tf.keras.layers.Conv1D(filters=5, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv1')(x_masked)
    x_conv1_norm = tf.keras.layers.BatchNormalization()(x_conv1)

    ### Segundo conv ###
    x_conv2 = tf.keras.layers.Conv1D(filters=10, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv2')(x_conv1_norm)
    x_conv2_norm = tf.keras.layers.BatchNormalization()(x_conv2)

    ### Tercer conv ###
    x_conv3 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv3')(x_conv2_norm)
    x_conv3_norm = tf.keras.layers.BatchNormalization()(x_conv3)

    ### Cuarto conv ###
    x_conv4 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv4')(x_conv3_norm)
    x_conv4_norm = tf.keras.layers.BatchNormalization()(x_conv4)

    # Definición de la capa de aplanamiento para conectar la CNN con la FCL 
    x_flat = tf.keras.layers.Flatten()(x_conv4_norm)
    
    
    # Definición de las capas fully connected
    x_full1 = tf.keras.layers.Dense(units=30, activation='relu')(x_flat)
    x_full1 = tf.keras.layers.BatchNormalization()(x_full1)

    x_full2 = tf.keras.layers.Dense(units=80, activation='relu')(x_full1)
    x_full2 = tf.keras.layers.BatchNormalization()(x_full2)

    x_full3 = tf.keras.layers.Dense(units=70, activation='relu')(x_full2)
    x_full3 = tf.keras.layers.BatchNormalization()(x_full3)

    # Definición de la última capa (activación)
    x_out_s1 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='glorot_uniform',
                                     name='S1_out')(x_full3)
    x_out_s2 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='glorot_uniform',
                                     name='S2_out')(x_full3)


    # Definición del modelo
    model = tf.keras.Model(inputs=x_in, outputs=[x_out_s1, x_out_s2], name=name)
    
    return model


def model_2_3(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    Es una variación del model_2 en el que se quita la mitad de las CNN.
    
    Parameters
    ----------
    input_shape : tuple or list
        Dimensión de un ejemplo de entrada.
    padding_value : int
        Valor con el que se realiza el padding de los últimos segmentos de cada sonido.
    name : str or NoneType, optional
        Nombre del modelo. Por defecto es None.
        
    Returns
    -------
    model : tensorflow.keras.Model
        Modelo final de keras.
    '''
    # Definición de la entrada
    x_in = tf.keras.Input(shape=input_shape, dtype='float32')

    # Definición de la capa de máscara
    x_masked = tf.keras.layers.Masking(mask_value=padding_value, 
                                       input_shape=input_shape)(x_in)

    # Definición de las capas convolucionales
    ### Primer conv ###
    x_conv1 = tf.keras.layers.Conv1D(filters=5, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv1')(x_masked)
    x_conv1_norm = tf.keras.layers.BatchNormalization()(x_conv1)

    ### Segundo conv ###
    x_conv2 = tf.keras.layers.Conv1D(filters=10, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv2')(x_conv1_norm)
    x_conv2_norm = tf.keras.layers.BatchNormalization()(x_conv2)


    # Definición de la capa de aplanamiento para conectar la CNN con la FCL 
    x_flat = tf.keras.layers.Flatten()(x_conv2_norm)
    
    
    # Definición de las capas fully connected
    x_full1 = tf.keras.layers.Dense(units=30, activation='relu')(x_flat)
    x_full1 = tf.keras.layers.BatchNormalization()(x_full1)

    x_full2 = tf.keras.layers.Dense(units=80, activation='relu')(x_full1)
    x_full2 = tf.keras.layers.BatchNormalization()(x_full2)

    x_full3 = tf.keras.layers.Dense(units=70, activation='relu')(x_full2)
    x_full3 = tf.keras.layers.BatchNormalization()(x_full3)

    # Definición de la última capa (activación)
    x_out_s1 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='glorot_uniform',
                                     name='S1_out')(x_full3)
    x_out_s2 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='glorot_uniform',
                                     name='S2_out')(x_full3)


    # Definición del modelo
    model = tf.keras.Model(inputs=x_in, outputs=[x_out_s1, x_out_s2], name=name)
    
    return model


def model_2_4(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    Es una variación del model_2 en el que se agrega el doble de las CNN. 
    
    Parameters
    ----------
    input_shape : tuple or list
        Dimensión de un ejemplo de entrada.
    padding_value : int
        Valor con el que se realiza el padding de los últimos segmentos de cada sonido.
    name : str or NoneType, optional
        Nombre del modelo. Por defecto es None.
        
    Returns
    -------
    model : tensorflow.keras.Model
        Modelo final de keras.
    '''
    # Definición de la entrada
    x_in = tf.keras.Input(shape=input_shape, dtype='float32')

    # Definición de la capa de máscara
    x_masked = tf.keras.layers.Masking(mask_value=padding_value, 
                                       input_shape=input_shape)(x_in)

    # Definición de las capas convolucionales
    ### Primer conv ###
    x_conv1 = tf.keras.layers.Conv1D(filters=5, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv1')(x_masked)
    x_conv1_norm = tf.keras.layers.BatchNormalization()(x_conv1)

    ### Segundo conv ###
    x_conv2 = tf.keras.layers.Conv1D(filters=10, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv2')(x_conv1_norm)
    x_conv2_norm = tf.keras.layers.BatchNormalization()(x_conv2)

    ### Tercer conv ###
    x_conv3 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv3')(x_conv2_norm)
    x_conv3_norm = tf.keras.layers.BatchNormalization()(x_conv3)

    ### Cuarto conv ###
    x_conv4 = tf.keras.layers.Conv1D(filters=30, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv4')(x_conv3_norm)
    x_conv4_norm = tf.keras.layers.BatchNormalization()(x_conv4)
    
    ### Quinto conv ###
    x_conv5 = tf.keras.layers.Conv1D(filters=50, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv5')(x_conv4_norm)
    x_conv5_norm = tf.keras.layers.BatchNormalization()(x_conv5)
    
    ### Sexto conv ###
    x_conv6 = tf.keras.layers.Conv1D(filters=40, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv6')(x_conv5_norm)
    x_conv6_norm = tf.keras.layers.BatchNormalization()(x_conv6)
    
    ### Séptimo conv ###
    x_conv7 = tf.keras.layers.Conv1D(filters=30, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv7')(x_conv6_norm)
    x_conv7_norm = tf.keras.layers.BatchNormalization()(x_conv7)

    # Definición de la capa de aplanamiento para conectar la CNN con la FCL 
    x_flat = tf.keras.layers.Flatten()(x_conv7_norm)
    
    
    # Definición de las capas fully connected
    x_full1 = tf.keras.layers.Dense(units=30, activation='relu')(x_flat)
    x_full1 = tf.keras.layers.BatchNormalization()(x_full1)

    x_full2 = tf.keras.layers.Dense(units=80, activation='relu')(x_full1)
    x_full2 = tf.keras.layers.BatchNormalization()(x_full2)

    x_full3 = tf.keras.layers.Dense(units=70, activation='relu')(x_full2)
    x_full3 = tf.keras.layers.BatchNormalization()(x_full3)

    # Definición de la última capa (activación)
    x_out_s1 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='glorot_uniform',
                                     name='S1_out')(x_full3)
    x_out_s2 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='glorot_uniform',
                                     name='S2_out')(x_full3)


    # Definición del modelo
    model = tf.keras.Model(inputs=x_in, outputs=[x_out_s1, x_out_s2], name=name)
    
    return model


def model_2_5(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    Es una variación del model_2 en el que se deja la mitad de las FC.
    
    Parameters
    ----------
    input_shape : tuple or list
        Dimensión de un ejemplo de entrada.
    padding_value : int
        Valor con el que se realiza el padding de los últimos segmentos de cada sonido.
    name : str or NoneType, optional
        Nombre del modelo. Por defecto es None.
        
    Returns
    -------
    model : tensorflow.keras.Model
        Modelo final de keras.
    '''
    # Definición de la entrada
    x_in = tf.keras.Input(shape=input_shape, dtype='float32')

    # Definición de la capa de máscara
    x_masked = tf.keras.layers.Masking(mask_value=padding_value, 
                                       input_shape=input_shape)(x_in)

    # Definición de las capas convolucionales
    ### Primer conv ###
    x_conv1 = tf.keras.layers.Conv1D(filters=5, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv1')(x_masked)
    x_conv1_norm = tf.keras.layers.BatchNormalization()(x_conv1)

    ### Segundo conv ###
    x_conv2 = tf.keras.layers.Conv1D(filters=10, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv2')(x_conv1_norm)
    x_conv2_norm = tf.keras.layers.BatchNormalization()(x_conv2)

    ### Tercer conv ###
    x_conv3 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv3')(x_conv2_norm)
    x_conv3_norm = tf.keras.layers.BatchNormalization()(x_conv3)

    ### Cuarto conv ###
    x_conv4 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv4')(x_conv3_norm)
    x_conv4_norm = tf.keras.layers.BatchNormalization()(x_conv4)

    # Definición de la capa de aplanamiento para conectar la CNN con la FCL 
    x_flat = tf.keras.layers.Flatten()(x_conv4_norm)
    
    
    # Definición de las capas fully connected
    x_full1 = tf.keras.layers.Dense(units=30, activation='relu')(x_flat)
    x_full1 = tf.keras.layers.BatchNormalization()(x_full1)

    # Definición de la última capa (activación)
    x_out_s1 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='glorot_uniform',
                                     name='S1_out')(x_full1)
    x_out_s2 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='glorot_uniform',
                                     name='S2_out')(x_full1)


    # Definición del modelo
    model = tf.keras.Model(inputs=x_in, outputs=[x_out_s1, x_out_s2], name=name)
    
    return model


def model_2_6(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    En esta versión se usa el doble de capas FC.
    
    Parameters
    ----------
    input_shape : tuple or list
        Dimensión de un ejemplo de entrada.
    padding_value : int
        Valor con el que se realiza el padding de los últimos segmentos de cada sonido.
    name : str or NoneType, optional
        Nombre del modelo. Por defecto es None.
        
    Returns
    -------
    model : tensorflow.keras.Model
        Modelo final de keras.
    '''
    # Definición de la entrada
    x_in = tf.keras.Input(shape=input_shape, dtype='float32')

    # Definición de la capa de máscara
    x_masked = tf.keras.layers.Masking(mask_value=padding_value, 
                                       input_shape=input_shape)(x_in)

    # Definición de las capas convolucionales
    ### Primer conv ###
    x_conv1 = tf.keras.layers.Conv1D(filters=5, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv1')(x_masked)
    x_conv1_norm = tf.keras.layers.BatchNormalization()(x_conv1)

    ### Segundo conv ###
    x_conv2 = tf.keras.layers.Conv1D(filters=10, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv2')(x_conv1_norm)
    x_conv2_norm = tf.keras.layers.BatchNormalization()(x_conv2)

    ### Tercer conv ###
    x_conv3 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv3')(x_conv2_norm)
    x_conv3_norm = tf.keras.layers.BatchNormalization()(x_conv3)

    ### Cuarto conv ###
    x_conv4 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv4')(x_conv3_norm)
    x_conv4_norm = tf.keras.layers.BatchNormalization()(x_conv4)

    # Definición de la capa de aplanamiento para conectar la CNN con la FCL 
    x_flat = tf.keras.layers.Flatten()(x_conv4_norm)
    
    # Definición de las capas fully connected
    x_full1 = tf.keras.layers.Dense(units=30, activation='relu')(x_flat)
    x_full1 = tf.keras.layers.BatchNormalization()(x_full1)

    x_full2 = tf.keras.layers.Dense(units=80, activation='relu')(x_full1)
    x_full2 = tf.keras.layers.BatchNormalization()(x_full2)

    x_full3 = tf.keras.layers.Dense(units=70, activation='relu')(x_full2)
    x_full3 = tf.keras.layers.BatchNormalization()(x_full3)
    
    x_full4 = tf.keras.layers.Dense(units=70, activation='relu')(x_full3)
    x_full4 = tf.keras.layers.BatchNormalization()(x_full4)
    
    x_full5 = tf.keras.layers.Dense(units=70, activation='relu')(x_full4)
    x_full5 = tf.keras.layers.BatchNormalization()(x_full5)
    
    x_full6 = tf.keras.layers.Dense(units=70, activation='relu')(x_full5)
    x_full6 = tf.keras.layers.BatchNormalization()(x_full6)

    # Definición de la última capa (activación)
    x_out_s1 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='glorot_uniform',
                                     name='S1_out')(x_full6)
    x_out_s2 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='glorot_uniform',
                                     name='S2_out')(x_full6)


    # Definición del modelo
    model = tf.keras.Model(inputs=x_in, outputs=[x_out_s1, x_out_s2], name=name)
    
    return model


def model_2_7(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    En esta variación de usa la mitad de las capas convolucionales y la mitad de
    las capas FC.
    
    Parameters
    ----------
    input_shape : tuple or list
        Dimensión de un ejemplo de entrada.
    padding_value : int
        Valor con el que se realiza el padding de los últimos segmentos de cada sonido.
    name : str or NoneType, optional
        Nombre del modelo. Por defecto es None.
        
    Returns
    -------
    model : tensorflow.keras.Model
        Modelo final de keras.
    '''
    # Definición de la entrada
    x_in = tf.keras.Input(shape=input_shape, dtype='float32')

    # Definición de la capa de máscara
    x_masked = tf.keras.layers.Masking(mask_value=padding_value, 
                                       input_shape=input_shape)(x_in)

    # Definición de las capas convolucionales
    ### Primer conv ###
    x_conv1 = tf.keras.layers.Conv1D(filters=5, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv1')(x_masked)
    x_conv1_norm = tf.keras.layers.BatchNormalization()(x_conv1)

    ### Segundo conv ###
    x_conv2 = tf.keras.layers.Conv1D(filters=10, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='glorot_uniform', 
                                     name='Conv2')(x_conv1_norm)
    x_conv2_norm = tf.keras.layers.BatchNormalization()(x_conv2)


    
    
    # Definición de la capa de aplanamiento para conectar la CNN con la FCL 
    x_flat = tf.keras.layers.Flatten()(x_conv2_norm)
    
    
    
    # Definición de las capas fully connected
    x_full1 = tf.keras.layers.Dense(units=30, activation='relu')(x_flat)
    x_full1 = tf.keras.layers.BatchNormalization()(x_full1)

    # Definición de la última capa (activación)
    x_out_s1 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='glorot_uniform',
                                     name='S1_out')(x_full1)
    x_out_s2 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='glorot_uniform',
                                     name='S2_out')(x_full1)

    
    
    # Definición del modelo
    model = tf.keras.Model(inputs=x_in, outputs=[x_out_s1, x_out_s2], name=name)
    
    return model


def model_2_8(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    En esta versión se usa 10 de capas FC.
    
    Parameters
    ----------
    input_shape : tuple or list
        Dimensión de un ejemplo de entrada.
    padding_value : int
        Valor con el que se realiza el padding de los últimos segmentos de cada sonido.
    name : str or NoneType, optional
        Nombre del modelo. Por defecto es None.
        
    Returns
    -------
    model : tensorflow.keras.Model
        Modelo final de keras.
    '''
    # Definición de la entrada
    x_in = tf.keras.Input(shape=input_shape, dtype='float32')

    # Definición de la capa de máscara
    x_masked = tf.keras.layers.Masking(mask_value=padding_value, 
                                       input_shape=input_shape)(x_in)

    # Definición de las capas convolucionales
    ### Primer conv ###
    x_conv1 = tf.keras.layers.Conv1D(filters=5, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='he_normal', 
                                     name='Conv1')(x_masked)
    x_conv1_norm = tf.keras.layers.BatchNormalization()(x_conv1)

    ### Segundo conv ###
    x_conv2 = tf.keras.layers.Conv1D(filters=10, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='he_normal', 
                                     name='Conv2')(x_conv1_norm)
    x_conv2_norm = tf.keras.layers.BatchNormalization()(x_conv2)

    ### Tercer conv ###
    x_conv3 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='he_normal', 
                                     name='Conv3')(x_conv2_norm)
    x_conv3_norm = tf.keras.layers.BatchNormalization()(x_conv3)

    ### Cuarto conv ###
    x_conv4 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='he_normal', 
                                     name='Conv4')(x_conv3_norm)
    x_conv4_norm = tf.keras.layers.BatchNormalization()(x_conv4)

    # Definición de la capa de aplanamiento para conectar la CNN con la FCL 
    x_flat = tf.keras.layers.Flatten()(x_conv4_norm)
    
    # Definición de las capas fully connected
    x_full1 = tf.keras.layers.Dense(units=30, activation='relu', 
                                    kernel_initializer='he_normal')(x_flat)
    x_full1 = tf.keras.layers.BatchNormalization()(x_full1)

    x_full2 = tf.keras.layers.Dense(units=80, activation='relu', 
                                    kernel_initializer='he_normal')(x_full1)
    x_full2 = tf.keras.layers.BatchNormalization()(x_full2)

    x_full3 = tf.keras.layers.Dense(units=70, activation='relu', 
                                    kernel_initializer='he_normal')(x_full2)
    x_full3 = tf.keras.layers.BatchNormalization()(x_full3)
    
    x_full4 = tf.keras.layers.Dense(units=70, activation='relu', 
                                    kernel_initializer='he_normal')(x_full3)
    x_full4 = tf.keras.layers.BatchNormalization()(x_full4)
    
    x_full5 = tf.keras.layers.Dense(units=70, activation='relu', 
                                    kernel_initializer='he_normal')(x_full4)
    x_full5 = tf.keras.layers.BatchNormalization()(x_full5)
    
    x_full6 = tf.keras.layers.Dense(units=70, activation='relu', 
                                    kernel_initializer='he_normal')(x_full5)
    x_full6 = tf.keras.layers.BatchNormalization()(x_full6)
    
    x_full7 = tf.keras.layers.Dense(units=70, activation='relu', 
                                    kernel_initializer='he_normal')(x_full6)
    x_full7 = tf.keras.layers.BatchNormalization()(x_full7)
    
    x_full8 = tf.keras.layers.Dense(units=70, activation='relu', 
                                    kernel_initializer='he_normal')(x_full7)
    x_full8 = tf.keras.layers.BatchNormalization()(x_full8)
    
    x_full9 = tf.keras.layers.Dense(units=70, activation='relu', 
                                    kernel_initializer='he_normal')(x_full8)
    x_full9 = tf.keras.layers.BatchNormalization()(x_full9)
    
    x_full10 = tf.keras.layers.Dense(units=70, activation='relu', 
                                    kernel_initializer='he_normal')(x_full9)
    x_full10 = tf.keras.layers.BatchNormalization()(x_full10)

    # Definición de la última capa (activación)
    x_out_s1 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='he_normal',
                                     name='S1_out')(x_full10)
    x_out_s2 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='he_normal',
                                     name='S2_out')(x_full10)


    # Definición del modelo
    model = tf.keras.Model(inputs=x_in, outputs=[x_out_s1, x_out_s2], name=name)
    
    return model


def model_2_9(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    En el fondo es el mismo modelo que el 2_2 pero con salida softmax.
    
    Parameters
    ----------
    input_shape : tuple or list
        Dimensión de un ejemplo de entrada.
    padding_value : int
        Valor con el que se realiza el padding de los últimos segmentos de cada sonido.
    name : str or NoneType, optional
        Nombre del modelo. Por defecto es None.
        
    Returns
    -------
    model : tensorflow.keras.Model
        Modelo final de keras.
    '''
    # Definición de la entrada
    x_in = tf.keras.Input(shape=input_shape, dtype='float32')

    # Definición de la capa de máscara
    x_masked = tf.keras.layers.Masking(mask_value=padding_value, 
                                       input_shape=input_shape)(x_in)

    # Definición de las capas convolucionales
    ### Primer conv ###
    x_conv1 = tf.keras.layers.Conv1D(filters=5, kernel_size=200, padding='same', 
                                     activation='relu', kernel_initializer='he_normal', 
                                     name='Conv1')(x_masked)
    x_conv1_norm = tf.keras.layers.BatchNormalization()(x_conv1)

    ### Segundo conv ###
    x_conv2 = tf.keras.layers.Conv1D(filters=10, kernel_size=200, padding='same', 
                                     activation='relu', kernel_initializer='he_normal', 
                                     name='Conv2')(x_conv1_norm)
    x_conv2_norm = tf.keras.layers.BatchNormalization()(x_conv2)

    ### Tercer conv ###
    x_conv3 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='he_normal', 
                                     name='Conv3')(x_conv2_norm)
    x_conv3_norm = tf.keras.layers.BatchNormalization()(x_conv3)

    ### Cuarto conv ###
    x_conv4 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='he_normal', 
                                     name='Conv4')(x_conv3_norm)
    x_conv4_norm = tf.keras.layers.BatchNormalization()(x_conv4)

    # Definición de la capa de aplanamiento para conectar la CNN con la FCL 
    x_flat = tf.keras.layers.Flatten()(x_conv4_norm)
    
    
    # Definición de las capas fully connected
    x_full1 = tf.keras.layers.Dense(units=30, activation='relu',
                                    kernel_initializer='he_normal')(x_flat)
    x_full1 = tf.keras.layers.BatchNormalization()(x_full1)

    x_full2 = tf.keras.layers.Dense(units=80, activation='relu',
                                    kernel_initializer='he_normal')(x_full1)
    x_full2 = tf.keras.layers.BatchNormalization()(x_full2)

    x_full3 = tf.keras.layers.Dense(units=70, activation='relu',
                                    kernel_initializer='he_normal')(x_full2)
    x_full3 = tf.keras.layers.BatchNormalization()(x_full3)

    # Definición de la última capa (activación)
    x_out = tf.keras.layers.Dense(units=3, activation='softmax', 
                                  kernel_initializer='he_normal',
                                  name='Heart_out')(x_full3)

    # Definición del modelo
    model = tf.keras.Model(inputs=x_in, outputs=x_out, name=name)
    
    return model


def model_3(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    En este caso a diferencia del modelo 2, se hace que la salida sea solo un canal.
    Por lo tanto, la salida tendrá información tanto de S1 como de S2.
    
    Parameters
    ----------
    input_shape : tuple or list
        Dimensión de un ejemplo de entrada.
    padding_value : int
        Valor con el que se realiza el padding de los últimos segmentos de cada sonido.
    name : str or NoneType, optional
        Nombre del modelo. Por defecto es None.
        
    Returns
    -------
    model : tensorflow.keras.Model
        Modelo final de keras.
    '''
    # Definición de la entrada
    x_in = tf.keras.Input(shape=input_shape, dtype='float32')

    # Definición de la capa de máscara
    x_masked = tf.keras.layers.Masking(mask_value=padding_value, 
                                       input_shape=input_shape,
                                       name='Masking')(x_in)

    # Definición de las capas convolucionales
    ### Primer conv ###
    x_conv1 = tf.keras.layers.Conv1D(filters=5, kernel_size=200, padding='same', 
                                     activation='relu', 
                                     kernel_initializer='glorot_uniform', 
                                     name='Conv1')(x_masked)
    x_conv1_norm = tf.keras.layers.BatchNormalization(name='BatchConv1')(x_conv1)

    ### Segundo conv ###
    x_conv2 = tf.keras.layers.Conv1D(filters=10, kernel_size=200, padding='same', 
                                     activation='relu', 
                                     kernel_initializer='glorot_uniform', 
                                     name='Conv2')(x_conv1_norm)
    x_conv2_norm = tf.keras.layers.BatchNormalization(name='BatchConv2')(x_conv2)

    ### Tercer conv ###
    x_conv3 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                     activation='relu', 
                                     kernel_initializer='glorot_uniform', 
                                     name='Conv3')(x_conv2_norm)
    x_conv3_norm = tf.keras.layers.BatchNormalization(name='BatchConv3')(x_conv3)

    ### Cuarto conv ###
    x_conv4 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                     activation='relu', 
                                     kernel_initializer='glorot_uniform', 
                                     name='Conv4')(x_conv3_norm)
    x_conv4_norm = tf.keras.layers.BatchNormalization(name='BatchConv4')(x_conv4)

    # Definición de la capa de aplanamiento para conectar la CNN con la FCL 
    x_flat = tf.keras.layers.Flatten()(x_conv4_norm)
    
    
    # Definición de las capas fully connected
    x_full1 = tf.keras.layers.Dense(units=30, activation='relu', 
                                    name='FC1')(x_flat)
    x_full1 = tf.keras.layers.BatchNormalization(name='BatchFC1')(x_full1)
    
    
    x_full2 = tf.keras.layers.Dense(units=80, activation='relu', 
                                    name='FC2')(x_full1)
    x_full2 = tf.keras.layers.BatchNormalization(name='BatchFC2')(x_full2)
    
    
    x_full3 = tf.keras.layers.Dense(units=70, activation='relu', 
                                    name='FC3')(x_full2)
    x_full3 = tf.keras.layers.BatchNormalization(name='BatchFC3')(x_full3)

    # Definición de la última capa (activación)
    x_out = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                  kernel_initializer='glorot_uniform',
                                  name='Heart_out')(x_full3)


    # Definición del modelo
    model = tf.keras.Model(inputs=x_in, outputs=x_out, name=name)
    
    return model


def model_4_1(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    Básicamente es el modelo 2_1 pero aplicado a un mayor número de envolventes.
    Envolventes usadas:
    - Sonido raw
    - Filtros homomórficos
    - Envolventes de Hilbert
    - Wavelets multiescala
    
    Parameters
    ----------
    input_shape : tuple or list
        Dimensión de un ejemplo de entrada.
    padding_value : int
        Valor con el que se realiza el padding de los últimos segmentos de cada sonido.
    name : str or NoneType, optional
        Nombre del modelo. Por defecto es None.
        
    Returns
    -------
    model : tensorflow.keras.Model
        Modelo final de keras.
    '''
    return model_2_1(input_shape, padding_value, name=name)


def model_4_2(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    Básicamente es el modelo 2_1 pero aplicado a un mayor número de envolventes.
    Envolventes usadas:
    - Sonido raw
    - Filtros homomórficos
    - Envolventes de Hilbert
    - Wavelets multiescala
    - Spectral tracking
    - Simplicity envelope
    
    Parameters
    ----------
    input_shape : tuple or list
        Dimensión de un ejemplo de entrada.
    padding_value : int
        Valor con el que se realiza el padding de los últimos segmentos de cada sonido.
    name : str or NoneType, optional
        Nombre del modelo. Por defecto es None.
        
    Returns
    -------
    model : tensorflow.keras.Model
        Modelo final de keras.
    '''
    return model_2_1(input_shape, padding_value, name=name)


def model_4_3(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    Básicamente es el modelo 2_1 pero aplicado a un mayor número de envolventes.
    Envolventes usadas:
    - Sonido raw
    - Filtros homomórficos
    - Wavelets multiescala
    - Spectral tracking
    - Simplicity envelope
    
    Parameters
    ----------
    input_shape : tuple or list
        Dimensión de un ejemplo de entrada.
    padding_value : int
        Valor con el que se realiza el padding de los últimos segmentos de cada sonido.
    name : str or NoneType, optional
        Nombre del modelo. Por defecto es None.
        
    Returns
    -------
    model : tensorflow.keras.Model
        Modelo final de keras.
    '''
    return model_2_1(input_shape, padding_value, name=name)


def model_4_4(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    Básicamente es el modelo 2_1 pero aplicado a un mayor número de envolventes.
    Envolventes usadas:
    - Sonido raw
    - Filtros homomórficos
    - Envolventes de Hilbert
    - Wavelets multiescala
    - Spectral tracking
    - Simplicity envelope
    - Variance Fractal Dimension
    
    Parameters
    ----------
    input_shape : tuple or list
        Dimensión de un ejemplo de entrada.
    padding_value : int
        Valor con el que se realiza el padding de los últimos segmentos de cada sonido.
    name : str or NoneType, optional
        Nombre del modelo. Por defecto es None.
        
    Returns
    -------
    model : tensorflow.keras.Model
        Modelo final de keras.
    '''
    return model_2_1(input_shape, padding_value, name=name)


def model_4_5(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    Básicamente es el modelo 2_1 pero aplicado a un mayor número de envolventes.
    Envolventes usadas:
    - Sonido raw
    - Filtros homomórficos
    - Envolventes de magnitud de Hilbert
    - Wavelets multiescala
    - Spectral tracking
    - Energía por bandas
    - Variance Fractal Dimension
    
    Parameters
    ----------
    input_shape : tuple or list
        Dimensión de un ejemplo de entrada.
    padding_value : int
        Valor con el que se realiza el padding de los últimos segmentos de cada sonido.
    name : str or NoneType, optional
        Nombre del modelo. Por defecto es None.
        
    Returns
    -------
    model : tensorflow.keras.Model
        Modelo final de keras.
    '''
    # Definición de la entrada
    x_in = tf.keras.Input(shape=input_shape, dtype='float32')

    # Definición de la capa de máscara
    x_masked = tf.keras.layers.Masking(mask_value=padding_value, 
                                       input_shape=input_shape)(x_in)

    # Definición de las capas convolucionales
    ### Primer conv ###
    x_conv1 = tf.keras.layers.Conv1D(filters=5, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='he_normal', 
                                     name='Conv1')(x_masked)
    x_conv1_norm = tf.keras.layers.BatchNormalization()(x_conv1)

    ### Segundo conv ###
    x_conv2 = tf.keras.layers.Conv1D(filters=10, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='he_normal', 
                                     name='Conv2')(x_conv1_norm)
    x_conv2_norm = tf.keras.layers.BatchNormalization()(x_conv2)

    ### Tercer conv ###
    x_conv3 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='he_normal', 
                                     name='Conv3')(x_conv2_norm)
    x_conv3_norm = tf.keras.layers.BatchNormalization()(x_conv3)

    ### Cuarto conv ###
    x_conv4 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                     activation='relu', kernel_initializer='he_normal', 
                                     name='Conv4')(x_conv3_norm)
    x_conv4_norm = tf.keras.layers.BatchNormalization()(x_conv4)

    # Definición de la capa de aplanamiento para conectar la CNN con la FCL 
    x_flat = tf.keras.layers.Flatten()(x_conv4_norm)
    
    
    # Definición de las capas fully connected
    x_full1 = tf.keras.layers.Dense(units=30, activation='relu')(x_flat)
    x_full1 = tf.keras.layers.BatchNormalization()(x_full1)

    x_full2 = tf.keras.layers.Dense(units=80, activation='relu')(x_full1)
    x_full2 = tf.keras.layers.BatchNormalization()(x_full2)

    x_full3 = tf.keras.layers.Dense(units=70, activation='relu')(x_full2)
    x_full3 = tf.keras.layers.BatchNormalization()(x_full3)

    # Definición de la última capa (activación)
    x_out = tf.keras.layers.Dense(units=3, activation='softmax', 
                                  kernel_initializer='he_normal',
                                  name='S1_out')(x_full3)


    # Definición del modelo
    model = tf.keras.Model(inputs=x_in, outputs=x_out, name=name)
    
    return model


def model_5_1(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    Básicamente es el modelo 2_1 pero aplicado a un mayor número de envolventes.
    Además cada envolvente posee una red neuronal para cada envolvente.
    Envolventes usadas:
    - Sonido raw
    - Filtros homomórficos
    - Envolventes de Hilbert
    - Wavelets multiescala
    - Spectral tracking
    - Simplicity envelope
    - Variance Fractal Dimension
    
    Parameters
    ----------
    input_shape : tuple or list
        Dimensión de un ejemplo de entrada.
    padding_value : int
        Valor con el que se realiza el padding de los últimos segmentos de cada sonido.
    name : str or NoneType, optional
        Nombre del modelo. Por defecto es None.
        
    Returns
    -------
    model : tensorflow.keras.Model
        Modelo final de keras.
    '''
    # Definición de la entrada
    x_in = tf.keras.Input(shape=input_shape, dtype='float32')
    
    # Definición de la lista de salida s1 y s2
    s1_list = list()
    s2_list = list()
    
    # Aplicando una separación para cada canal
    for i in range(input_shape[1]):
        # Se obtiene la i-ésima envolvente de la matriz de entrada
        channel_i = tf.keras.layers.Lambda(lambda x : 
            tf.keras.backend.expand_dims(x[:,:,i], axis=-1), name=f'Ch_{i+1}')(x_in)
    
        # Se hace el path del modelo
        model_i = model_2_1((input_shape[0], 1), padding_value, name=f'model_2_1_ch{i+1}')
        
        # Conectando el canal a la entrada del modelo (salida es una lista)
        s1_out, s2_out = model_i(channel_i)
        
        # Agregando a la lista de parámetros
        s1_list.append(s1_out)
        s2_list.append(s2_out)
    
    # Definición de una capa de aplanamiento
    s1_concat = tf.keras.layers.concatenate(s1_list, name='s1_concat')
    s2_concat = tf.keras.layers.concatenate(s2_list, name='s2_concat')
    
    # Definición de las 2 capas de salida
    x_out_s1 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='he_normal',
                                     name='S1_out')(s1_concat)
    x_out_s2 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='he_normal',
                                     name='S2_out')(s2_concat)
        
    # Creación del modelo final
    model = tf.keras.Model(x_in, [x_out_s1, x_out_s2])
    
    return model


def model_5_1_1(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    Básicamente es el modelo 2_1 pero aplicado a un mayor número de envolventes.
    Además cada envolvente posee una red neuronal para cada envolvente. 
    
    La diferencia con el model_5_1 es que en esta ocasión se realizan las conexiones 
    manuales
    
    Envolventes usadas:
    - Sonido raw
    - Filtros homomórficos
    - Envolventes de Hilbert
    - Wavelets multiescala
    - Spectral tracking
    - Simplicity envelope
    - Variance Fractal Dimension
    
    Parameters
    ----------
    input_shape : tuple or list
        Dimensión de un ejemplo de entrada.
    padding_value : int
        Valor con el que se realiza el padding de los últimos segmentos de cada sonido.
    name : str or NoneType, optional
        Nombre del modelo. Por defecto es None.
        
    Returns
    -------
    model : tensorflow.keras.Model
        Modelo final de keras.
    '''
    # Definición de una lista auxiliar de capas FC de salida
    s1_list = list()
    s2_list = list()
    
    # Definición de la entrada
    x_in = tf.keras.Input(shape=input_shape, dtype='float32')

    # Definición de la capa de máscara
    x_masked = tf.keras.layers.Masking(mask_value=padding_value, 
                                       input_shape=input_shape)(x_in)
    
    # Aplicando una separación para cada canal
    for i in range(input_shape[1]):
        # Se obtiene la i-ésima envolvente de la matriz de entrada
        channel_i = tf.keras.layers.Lambda(lambda x : 
            tf.keras.backend.expand_dims(x[:,:,i], axis=-1), name=f'Ch_{i+1}')(x_masked)
    
        # Definición de las capas convolucionales
        ### Primer conv ###
        x_conv1 = tf.keras.layers.Conv1D(filters=5, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='he_normal', 
                                        name=f'Conv1_ch{i+1}')(channel_i)
        x_conv1_norm = tf.keras.layers.BatchNormalization()(x_conv1)

        ### Segundo conv ###
        x_conv2 = tf.keras.layers.Conv1D(filters=10, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='he_normal', 
                                        name=f'Conv2_ch{i+1}')(x_conv1_norm)
        x_conv2_norm = tf.keras.layers.BatchNormalization()(x_conv2)

        ### Tercer conv ###
        x_conv3 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='he_normal', 
                                        name=f'Conv3_ch{i+1}')(x_conv2_norm)
        x_conv3_norm = tf.keras.layers.BatchNormalization()(x_conv3)

        ### Cuarto conv ###
        x_conv4 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='he_normal', 
                                        name=f'Conv4_ch{i+1}')(x_conv3_norm)
        x_conv4_norm = tf.keras.layers.BatchNormalization()(x_conv4)
        
        
        # Definición de la capa de aplanamiento para conectar la CNN con la FCL 
        x_flat = tf.keras.layers.Flatten()(x_conv4_norm)
        
        
        # Definición de las capas fully connected
        ### Fully Connected 1 ###
        x_full1 = tf.keras.layers.Dense(units=30, activation='relu',
                                        kernel_initializer='he_normal',
                                        name=f'FC1_ch{i+1}')(x_flat)
        x_full1 = tf.keras.layers.BatchNormalization()(x_full1)

        ### Fully Connected 2 ###
        x_full2 = tf.keras.layers.Dense(units=80, activation='relu',
                                        kernel_initializer='he_normal',
                                        name=f'FC2_ch{i+1}')(x_full1)
        x_full2 = tf.keras.layers.BatchNormalization()(x_full2)

        ### Fully Connected 3 ###
        x_full3 = tf.keras.layers.Dense(units=70, activation='relu',
                                        kernel_initializer='he_normal',
                                        name=f'FC3_ch{i+1}')(x_full2)
        x_full3 = tf.keras.layers.BatchNormalization()(x_full3)

        # Definición de las 2 capas de salida
        s1_lay = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                       kernel_initializer='he_normal',
                                       name=f'S1_out_ch{i+1}')(x_full3)
        s2_lay = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                       kernel_initializer='he_normal',
                                       name=f'S2_out_ch{i+1}')(x_full3)
        
        # Agregando a las listas
        s1_list.append(s1_lay)
        s2_list.append(s2_lay)
    
    # Definición de una capa de aplanamiento
    s1_concat = tf.keras.layers.concatenate(s1_list, name='s1_concat')
    s2_concat = tf.keras.layers.concatenate(s2_list, name='s2_concat')
    
    # Definición de las 2 capas de salida
    x_out_s1 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='he_normal',
                                     name='S1_out')(s1_concat)
    x_out_s2 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='he_normal',
                                     name='S2_out')(s2_concat)
    
    # Creación del modelo final
    model = tf.keras.Model(x_in, [x_out_s1, x_out_s2])
    
    return model


def model_5_2_1(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    Básicamente es el modelo 5_1_1 pero con un mayor número de FC a la salida.
    
    Envolventes usadas:
    - Sonido raw
    - Filtros homomórficos
    - Envolventes de Hilbert
    - Wavelets multiescala
    - Spectral tracking
    - Simplicity envelope
    - Variance Fractal Dimension
    
    Parameters
    ----------
    input_shape : tuple or list
        Dimensión de un ejemplo de entrada.
    padding_value : int
        Valor con el que se realiza el padding de los últimos segmentos de cada sonido.
    name : str or NoneType, optional
        Nombre del modelo. Por defecto es None.
        
    Returns
    -------
    model : tensorflow.keras.Model
        Modelo final de keras.
    '''
    # Definición de una lista auxiliar de capas FC de salida
    full_list = list()
    
    # Definición de la entrada
    x_in = tf.keras.Input(shape=input_shape, dtype='float32')

    # Definición de la capa de máscara
    x_masked = tf.keras.layers.Masking(mask_value=padding_value, 
                                       input_shape=input_shape)(x_in)
    
    # Aplicando una separación para cada canal
    for i in range(input_shape[1]):
        # Se obtiene la i-ésima envolvente de la matriz de entrada
        channel_i = tf.keras.layers.Lambda(lambda x : 
            tf.keras.backend.expand_dims(x[:,:,i], axis=-1), name=f'Ch_{i+1}')(x_masked)
    
        # Definición de las capas convolucionales
        ### Primer conv ###
        x_conv1 = tf.keras.layers.Conv1D(filters=5, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='glorot_uniform', 
                                        name=f'Conv1_ch{i+1}')(channel_i)
        x_conv1_norm = tf.keras.layers.BatchNormalization()(x_conv1)

        ### Segundo conv ###
        x_conv2 = tf.keras.layers.Conv1D(filters=10, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='glorot_uniform', 
                                        name=f'Conv2_ch{i+1}')(x_conv1_norm)
        x_conv2_norm = tf.keras.layers.BatchNormalization()(x_conv2)

        ### Tercer conv ###
        x_conv3 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='glorot_uniform', 
                                        name=f'Conv3_ch{i+1}')(x_conv2_norm)
        x_conv3_norm = tf.keras.layers.BatchNormalization()(x_conv3)

        ### Cuarto conv ###
        x_conv4 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='glorot_uniform', 
                                        name=f'Conv4_ch{i+1}')(x_conv3_norm)
        x_conv4_norm = tf.keras.layers.BatchNormalization()(x_conv4)
        
        
        # Definición de la capa de aplanamiento para conectar la CNN con la FCL 
        x_flat = tf.keras.layers.Flatten()(x_conv4_norm)
        
        
        # Definición de las capas fully connected
        ### Fully Connected 1 ###
        x_full1 = tf.keras.layers.Dense(units=30, activation='relu',
                                        kernel_initializer='glorot_uniform',
                                        name=f'FC1_ch{i+1}')(x_flat)
        x_full1 = tf.keras.layers.BatchNormalization()(x_full1)

        ### Fully Connected 2 ###
        x_full2 = tf.keras.layers.Dense(units=80, activation='relu',
                                        kernel_initializer='glorot_uniform',
                                        name=f'FC2_ch{i+1}')(x_full1)
        x_full2 = tf.keras.layers.BatchNormalization()(x_full2)

        ### Fully Connected 3 ###
        x_full3 = tf.keras.layers.Dense(units=70, activation='relu',
                                        kernel_initializer='glorot_uniform',
                                        name=f'FC3_ch{i+1}')(x_full2)
        x_full3 = tf.keras.layers.BatchNormalization()(x_full3)

                
        # Agregando a las listas
        full_list.append(x_full3)
    
    # Definición de una capa de aplanamiento
    fully_concat = tf.keras.layers.concatenate(full_list, name='fully_concat')
    
    ### Fully Connected 1 out ###
    x_full1_out = tf.keras.layers.Dense(units=30, activation='relu',
                                   kernel_initializer='glorot_uniform',
                                   name=f'FC1_out')(fully_concat)
    x_full1_out = tf.keras.layers.BatchNormalization()(x_full1_out)
    
    
    ### Fully Connected 2 out ###
    x_full2_out = tf.keras.layers.Dense(units=30, activation='relu',
                                   kernel_initializer='glorot_uniform',
                                   name=f'FC2_out')(x_full1_out)
    x_full2_out = tf.keras.layers.BatchNormalization()(x_full2_out)
    
    
    ### Fully Connected 3 out ###
    x_full3_out = tf.keras.layers.Dense(units=30, activation='relu',
                                   kernel_initializer='glorot_uniform',
                                   name=f'FC3_out')(x_full2_out)
    x_full3_out = tf.keras.layers.BatchNormalization()(x_full3_out)
    
    
    # Definición de las 2 capas de salida
    x_out_s1 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='glorot_uniform',
                                     name='S1_out')(x_full3_out)
    x_out_s2 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='glorot_uniform',
                                     name='S2_out')(x_full3_out)
    
    # Creación del modelo final
    model = tf.keras.Model(x_in, [x_out_s1, x_out_s2])
    
    return model


def model_5_2_2(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    Básicamente es el modelo 5_2_1, pero con menos envolventes.
    
    Envolventes usadas:
    - Sonido raw
    - Filtros homomórficos
    - Wavelets multiescala
    - Spectral tracking
    - Simplicity envelope
    
    Parameters
    ----------
    input_shape : tuple or list
        Dimensión de un ejemplo de entrada.
    padding_value : int
        Valor con el que se realiza el padding de los últimos segmentos de cada sonido.
    name : str or NoneType, optional
        Nombre del modelo. Por defecto es None.
        
    Returns
    -------
    model : tensorflow.keras.Model
        Modelo final de keras.
    '''
    # Definición de una lista auxiliar de capas FC de salida
    full_list = list()
    
    # Definición de la entrada
    x_in = tf.keras.Input(shape=input_shape, dtype='float32')

    # Definición de la capa de máscara
    x_masked = tf.keras.layers.Masking(mask_value=padding_value, 
                                       input_shape=input_shape)(x_in)
    
    # Aplicando una separación para cada canal
    for i in range(input_shape[1]):
        # Se obtiene la i-ésima envolvente de la matriz de entrada
        channel_i = tf.keras.layers.Lambda(lambda x : 
            tf.keras.backend.expand_dims(x[:,:,i], axis=-1), name=f'Ch_{i+1}')(x_masked)
    
        # Definición de las capas convolucionales
        ### Primer conv ###
        x_conv1 = tf.keras.layers.Conv1D(filters=5, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='glorot_uniform', 
                                        name=f'Conv1_ch{i+1}')(channel_i)
        x_conv1_norm = tf.keras.layers.BatchNormalization()(x_conv1)

        ### Segundo conv ###
        x_conv2 = tf.keras.layers.Conv1D(filters=10, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='glorot_uniform', 
                                        name=f'Conv2_ch{i+1}')(x_conv1_norm)
        x_conv2_norm = tf.keras.layers.BatchNormalization()(x_conv2)

        ### Tercer conv ###
        x_conv3 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='glorot_uniform', 
                                        name=f'Conv3_ch{i+1}')(x_conv2_norm)
        x_conv3_norm = tf.keras.layers.BatchNormalization()(x_conv3)

        ### Cuarto conv ###
        x_conv4 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='glorot_uniform', 
                                        name=f'Conv4_ch{i+1}')(x_conv3_norm)
        x_conv4_norm = tf.keras.layers.BatchNormalization()(x_conv4)
        
        
        # Definición de la capa de aplanamiento para conectar la CNN con la FCL 
        x_flat = tf.keras.layers.Flatten()(x_conv4_norm)
        
        
        # Definición de las capas fully connected
        ### Fully Connected 1 ###
        x_full1 = tf.keras.layers.Dense(units=30, activation='relu',
                                        kernel_initializer='glorot_uniform',
                                        name=f'FC1_ch{i+1}')(x_flat)
        x_full1 = tf.keras.layers.BatchNormalization()(x_full1)

        ### Fully Connected 2 ###
        x_full2 = tf.keras.layers.Dense(units=80, activation='relu',
                                        kernel_initializer='glorot_uniform',
                                        name=f'FC2_ch{i+1}')(x_full1)
        x_full2 = tf.keras.layers.BatchNormalization()(x_full2)

        ### Fully Connected 3 ###
        x_full3 = tf.keras.layers.Dense(units=70, activation='relu',
                                        kernel_initializer='glorot_uniform',
                                        name=f'FC3_ch{i+1}')(x_full2)
        x_full3 = tf.keras.layers.BatchNormalization()(x_full3)

                
        # Agregando a las listas
        full_list.append(x_full3)
    
    # Definición de una capa de aplanamiento
    fully_concat = tf.keras.layers.concatenate(full_list, name='fully_concat')
    
    ### Fully Connected 1 out ###
    x_full1_out = tf.keras.layers.Dense(units=30, activation='relu',
                                   kernel_initializer='glorot_uniform',
                                   name=f'FC1_out')(fully_concat)
    x_full1_out = tf.keras.layers.BatchNormalization()(x_full1_out)
    
    
    ### Fully Connected 2 out ###
    x_full2_out = tf.keras.layers.Dense(units=30, activation='relu',
                                   kernel_initializer='glorot_uniform',
                                   name=f'FC2_out')(x_full1_out)
    x_full2_out = tf.keras.layers.BatchNormalization()(x_full2_out)
    
    
    ### Fully Connected 3 out ###
    x_full3_out = tf.keras.layers.Dense(units=30, activation='relu',
                                   kernel_initializer='glorot_uniform',
                                   name=f'FC3_out')(x_full2_out)
    x_full3_out = tf.keras.layers.BatchNormalization()(x_full3_out)
    
    
    # Definición de las 2 capas de salida
    x_out_s1 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='glorot_uniform',
                                     name='S1_out')(x_full3_out)
    x_out_s2 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='glorot_uniform',
                                     name='S2_out')(x_full3_out)
    
    # Creación del modelo final
    model = tf.keras.Model(x_in, [x_out_s1, x_out_s2])
    
    return model


def model_5_2_3(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    Básicamente es el modelo 5_2_1, pero sin la capa lambda de dispersión. En este
    caso se hace uso de una entrada distinta para cada envolvente
    
    Envolventes usadas:
    - Sonido raw
    - Filtros homomórficos
    - Envolventes de Hilbert
    - Wavelets multiescala
    - Spectral tracking
    - Simplicity envelope
    - Variance Fractal Dimension
    
    Parameters
    ----------
    input_shape : tuple or list
        Dimensión de un ejemplo de entrada.
    padding_value : int
        Valor con el que se realiza el padding de los últimos segmentos de cada sonido.
    name : str or NoneType, optional
        Nombre del modelo. Por defecto es None.
        
    Returns
    -------
    model : tensorflow.keras.Model
        Modelo final de keras.
    '''    
    # Definición de una lista auxiliar de las entradas y de capas FC de salida
    x_in_list = list()
    full_list = list()
    
    # Aplicando una separación para cada canal
    for i in range(input_shape[1]):        
        # Definición de la entrada
        x_in = tf.keras.Input(shape=(input_shape[0], 1), dtype='float32',
                              name=f'Input_ch{i+1}')
        
        # Agregando a la lista
        x_in_list.append(x_in)

        # Definición de la capa de máscara
        x_masked = tf.keras.layers.Masking(mask_value=padding_value, 
                                           input_shape=(input_shape[0], 1),
                                           name=f'Masking_ch{i+1}')(x_in)
    
        # Definición de las capas convolucionales
        ### Primer conv ###
        x_conv1 = tf.keras.layers.Conv1D(filters=5, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='glorot_uniform', 
                                        name=f'Conv1_ch{i+1}')(x_masked)
        x_conv1_norm = tf.keras.layers.BatchNormalization()(x_conv1)

        ### Segundo conv ###
        x_conv2 = tf.keras.layers.Conv1D(filters=10, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='glorot_uniform', 
                                        name=f'Conv2_ch{i+1}')(x_conv1_norm)
        x_conv2_norm = tf.keras.layers.BatchNormalization()(x_conv2)

        ### Tercer conv ###
        x_conv3 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='glorot_uniform', 
                                        name=f'Conv3_ch{i+1}')(x_conv2_norm)
        x_conv3_norm = tf.keras.layers.BatchNormalization()(x_conv3)

        ### Cuarto conv ###
        x_conv4 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='glorot_uniform', 
                                        name=f'Conv4_ch{i+1}')(x_conv3_norm)
        x_conv4_norm = tf.keras.layers.BatchNormalization()(x_conv4)
        
        
        # Definición de la capa de aplanamiento para conectar la CNN con la FCL 
        x_flat = tf.keras.layers.Flatten()(x_conv4_norm)
        
        
        # Definición de las capas fully connected
        ### Fully Connected 1 ###
        x_full1 = tf.keras.layers.Dense(units=30, activation='relu',
                                        kernel_initializer='glorot_uniform',
                                        name=f'FC1_ch{i+1}')(x_flat)
        x_full1 = tf.keras.layers.BatchNormalization()(x_full1)

        ### Fully Connected 2 ###
        x_full2 = tf.keras.layers.Dense(units=80, activation='relu',
                                        kernel_initializer='glorot_uniform',
                                        name=f'FC2_ch{i+1}')(x_full1)
        x_full2 = tf.keras.layers.BatchNormalization()(x_full2)

        ### Fully Connected 3 ###
        x_full3 = tf.keras.layers.Dense(units=70, activation='relu',
                                        kernel_initializer='glorot_uniform',
                                        name=f'FC3_ch{i+1}')(x_full2)
        x_full3 = tf.keras.layers.BatchNormalization()(x_full3)

                
        # Agregando a las listas
        full_list.append(x_full3)
    
    # Definición de una capa de aplanamiento
    fully_concat = tf.keras.layers.concatenate(full_list, name='fully_concat')
    
    ### Fully Connected 1 out ###
    x_full1_out = tf.keras.layers.Dense(units=30, activation='relu',
                                   kernel_initializer='glorot_uniform',
                                   name=f'FC1_out')(fully_concat)
    x_full1_out = tf.keras.layers.BatchNormalization()(x_full1_out)
    
    
    ### Fully Connected 2 out ###
    x_full2_out = tf.keras.layers.Dense(units=30, activation='relu',
                                   kernel_initializer='glorot_uniform',
                                   name=f'FC2_out')(x_full1_out)
    x_full2_out = tf.keras.layers.BatchNormalization()(x_full2_out)
    
    
    ### Fully Connected 3 out ###
    x_full3_out = tf.keras.layers.Dense(units=30, activation='relu',
                                   kernel_initializer='glorot_uniform',
                                   name=f'FC3_out')(x_full2_out)
    x_full3_out = tf.keras.layers.BatchNormalization()(x_full3_out)
    
    
    # Definición de las 2 capas de salida
    x_out_s1 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='glorot_uniform',
                                     name='S1_out')(x_full3_out)
    x_out_s2 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='glorot_uniform',
                                     name='S2_out')(x_full3_out)
    
    # Creación del modelo final
    model = tf.keras.Model(x_in_list, [x_out_s1, x_out_s2])
    
    return model


def model_5_2_4(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    Básicamente es el modelo 5_2_1, pero sin la capa lambda de dispersión. En este
    caso se hace uso de una entrada distinta para cada envolvente, y además se aplica
    una etapa sigmoide (como en el model_2_1) extra en cada canal.
    
    Envolventes usadas:
    - Sonido raw
    - Filtros homomórficos
    - Envolventes de Hilbert
    - Wavelets multiescala
    - Spectral tracking
    - Simplicity envelope
    - Variance Fractal Dimension
    
    Parameters
    ----------
    input_shape : tuple or list
        Dimensión de un ejemplo de entrada.
    padding_value : int
        Valor con el que se realiza el padding de los últimos segmentos de cada sonido.
    name : str or NoneType, optional
        Nombre del modelo. Por defecto es None.
        
    Returns
    -------
    model : tensorflow.keras.Model
        Modelo final de keras.
    '''    
    # Definición de una lista auxiliar de las entradas y de capas FC de salida
    x_in_list = list()
    s1_list = list()
    s2_list = list()
    
    # Aplicando una separación para cada canal
    for i in range(input_shape[1]):        
        # Definición de la entrada
        x_in = tf.keras.Input(shape=(input_shape[0], 1), dtype='float32',
                              name=f'Input_ch{i+1}')
        
        # Agregando a la lista
        x_in_list.append(x_in)

        # Definición de la capa de máscara
        x_masked = tf.keras.layers.Masking(mask_value=padding_value, 
                                           input_shape=(input_shape[0], 1),
                                           name=f'Masking_ch{i+1}')(x_in)
    
        # Definición de las capas convolucionales
        ### Primer conv ###
        x_conv1 = tf.keras.layers.Conv1D(filters=5, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='glorot_uniform', 
                                        name=f'Conv1_ch{i+1}')(x_masked)
        x_conv1_norm = tf.keras.layers.BatchNormalization()(x_conv1)

        ### Segundo conv ###
        x_conv2 = tf.keras.layers.Conv1D(filters=10, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='glorot_uniform', 
                                        name=f'Conv2_ch{i+1}')(x_conv1_norm)
        x_conv2_norm = tf.keras.layers.BatchNormalization()(x_conv2)

        ### Tercer conv ###
        x_conv3 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='glorot_uniform', 
                                        name=f'Conv3_ch{i+1}')(x_conv2_norm)
        x_conv3_norm = tf.keras.layers.BatchNormalization()(x_conv3)

        ### Cuarto conv ###
        x_conv4 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='glorot_uniform', 
                                        name=f'Conv4_ch{i+1}')(x_conv3_norm)
        x_conv4_norm = tf.keras.layers.BatchNormalization()(x_conv4)
        
        
        # Definición de la capa de aplanamiento para conectar la CNN con la FCL 
        x_flat = tf.keras.layers.Flatten()(x_conv4_norm)
        
        
        # Definición de las capas fully connected
        ### Fully Connected 1 ###
        x_full1 = tf.keras.layers.Dense(units=30, activation='relu',
                                        kernel_initializer='glorot_uniform',
                                        name=f'FC1_ch{i+1}')(x_flat)
        x_full1 = tf.keras.layers.BatchNormalization()(x_full1)

        ### Fully Connected 2 ###
        x_full2 = tf.keras.layers.Dense(units=80, activation='relu',
                                        kernel_initializer='glorot_uniform',
                                        name=f'FC2_ch{i+1}')(x_full1)
        x_full2 = tf.keras.layers.BatchNormalization()(x_full2)

        ### Fully Connected 3 ###
        x_full3 = tf.keras.layers.Dense(units=70, activation='relu',
                                        kernel_initializer='glorot_uniform',
                                        name=f'FC3_ch{i+1}')(x_full2)
        x_full3 = tf.keras.layers.BatchNormalization()(x_full3)

        # Definición de las 2 capas de salida
        s1_lay = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                       kernel_initializer='glorot_uniform',
                                       name=f'S1_out_ch{i+1}')(x_full3)
        s2_lay = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                       kernel_initializer='glorot_uniform',
                                       name=f'S2_out_ch{i+1}')(x_full3)
                
        # Agregando a las listas
        s1_list.append(s1_lay)
        s2_list.append(s2_lay)
    
    # Definición de una capa de aplanamiento
    s1_concat = tf.keras.layers.concatenate(s1_list, name='s1_concat')
    s2_concat = tf.keras.layers.concatenate(s2_list, name='s2_concat')
    
    
    # Definición de las 2 capas de salida
    x_out_s1 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='glorot_uniform',
                                     name='S1_out')(s1_concat)
    x_out_s2 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='glorot_uniform',
                                     name='S2_out')(s2_concat)
    
    # Creación del modelo final
    model = tf.keras.Model(x_in_list, [x_out_s1, x_out_s2])
    
    return model


def model_5_2_4_1(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    Básicamente es el modelo 5_2_4, pero usando la regla de oro planteada en [1].
    
    Envolventes usadas:
    - Sonido raw
    - Filtros homomórficos
    - Envolventes de Hilbert
    - Wavelets multiescala
    - Spectral tracking
    - Simplicity envelope
    - Variance Fractal Dimension
    
    Parameters
    ----------
    input_shape : tuple or list
        Dimensión de un ejemplo de entrada.
    padding_value : int
        Valor con el que se realiza el padding de los últimos segmentos de cada sonido.
    name : str or NoneType, optional
        Nombre del modelo. Por defecto es None.
        
    Returns
    -------
    model : tensorflow.keras.Model
        Modelo final de keras.
    
    References
    ----------
    [1] Gupta, C. N., Palaniappan, R., Rajan, S., Swaminathan, S., 
        & Krishnan, S. M. (2005, May). Segmentation and classification 
        of heart sounds. In Canadian Conference on Electrical and 
        Computer Engineering, 2005. (pp. 1674-1677). IEEE.
    '''    
    # Definición de una lista auxiliar de las entradas y de capas FC de salida
    x_in_list = list()
    s1_list = list()
    s2_list = list()
    
    # Aplicando una separación para cada canal
    for i in range(input_shape[1]):        
        # Definición de la entrada
        x_in = tf.keras.Input(shape=(input_shape[0], 1), dtype='float32',
                              name=f'Input_ch{i+1}')
        
        # Agregando a la lista
        x_in_list.append(x_in)

        # Definición de la capa de máscara
        x_masked = tf.keras.layers.Masking(mask_value=padding_value, 
                                           input_shape=(input_shape[0], 1),
                                           name=f'Masking_ch{i+1}')(x_in)
    
        # Definición de las capas convolucionales
        ### Primer conv ###
        x_conv1 = tf.keras.layers.Conv1D(filters=5, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='glorot_uniform', 
                                        name=f'Conv1_ch{i+1}')(x_masked)
        x_conv1_norm = tf.keras.layers.BatchNormalization()(x_conv1)

        ### Segundo conv ###
        x_conv2 = tf.keras.layers.Conv1D(filters=10, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='glorot_uniform', 
                                        name=f'Conv2_ch{i+1}')(x_conv1_norm)
        x_conv2_norm = tf.keras.layers.BatchNormalization()(x_conv2)

        ### Tercer conv ###
        x_conv3 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='glorot_uniform', 
                                        name=f'Conv3_ch{i+1}')(x_conv2_norm)
        x_conv3_norm = tf.keras.layers.BatchNormalization()(x_conv3)

        ### Cuarto conv ###
        x_conv4 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='glorot_uniform', 
                                        name=f'Conv4_ch{i+1}')(x_conv3_norm)
        x_conv4_norm = tf.keras.layers.BatchNormalization()(x_conv4)
        
        
        # Definición de la capa de aplanamiento para conectar la CNN con la FCL 
        x_flat = tf.keras.layers.Flatten()(x_conv4_norm)
        
        
        # Definición de las capas fully connected
        ### Fully Connected 1 ###
        x_full1 = tf.keras.layers.Dense(units=30, activation='relu',
                                        kernel_initializer='glorot_uniform',
                                        name=f'FC1_ch{i+1}')(x_flat)
        x_full1 = tf.keras.layers.BatchNormalization()(x_full1)

        ### Fully Connected 2 ###
        x_full2 = tf.keras.layers.Dense(units=80, activation='relu',
                                        kernel_initializer='glorot_uniform',
                                        name=f'FC2_ch{i+1}')(x_full1)
        x_full2 = tf.keras.layers.BatchNormalization()(x_full2)

        ### Fully Connected 3 ###
        x_full3 = tf.keras.layers.Dense(units=70, activation='relu',
                                        kernel_initializer='glorot_uniform',
                                        name=f'FC3_ch{i+1}')(x_full2)
        x_full3 = tf.keras.layers.BatchNormalization()(x_full3)

        # Definición de las 2 capas de salida
        s1_lay = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                       kernel_initializer='glorot_uniform',
                                       name=f'S1_out_ch{i+1}')(x_full3)
        s2_lay = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                       kernel_initializer='glorot_uniform',
                                       name=f'S2_out_ch{i+1}')(x_full3)
                
        # Agregando a las listas
        s1_list.append(s1_lay)
        s2_list.append(s2_lay)
    
    # Definición de una capa de aplanamiento
    s1_concat = tf.keras.layers.concatenate(s1_list, name='s1_concat')
    s2_concat = tf.keras.layers.concatenate(s2_list, name='s2_concat')
    
    
    # Definición de las 2 capas de salida
    x_out_s1 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='glorot_uniform',
                                     name='S1_out')(s1_concat)
    x_out_s2 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='glorot_uniform',
                                     name='S2_out')(s2_concat)
    
    # Creación del modelo final
    model = tf.keras.Model(x_in_list, [x_out_s1, x_out_s2])
    
    return model


def model_5_2_4_2(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    Básicamente es el modelo 5_2_4, pero usando como salida una capa softmax.
    
    Envolventes usadas:
    - Sonido raw
    - Filtros homomórficos
    - Envolvente de Hilbert de magnitud
    - Wavelets multiescala
    - Spectral tracking
    - Energía por bandas
    - Variance Fractal Dimension
    
    Parameters
    ----------
    input_shape : tuple or list
        Dimensión de un ejemplo de entrada.
    padding_value : int
        Valor con el que se realiza el padding de los últimos segmentos de cada sonido.
    name : str or NoneType, optional
        Nombre del modelo. Por defecto es None.
        
    Returns
    -------
    model : tensorflow.keras.Model
        Modelo final de keras.
    
    References
    ----------
    [1] Gupta, C. N., Palaniappan, R., Rajan, S., Swaminathan, S., 
        & Krishnan, S. M. (2005, May). Segmentation and classification 
        of heart sounds. In Canadian Conference on Electrical and 
        Computer Engineering, 2005. (pp. 1674-1677). IEEE.
    '''    
    # Definición de una lista auxiliar de las entradas y de capas FC de salida
    x_in_list = list()
    s12_list = list()
    
    # Aplicando una separación para cada canal
    for i in range(input_shape[1]):        
        # Definición de la entrada
        x_in = tf.keras.Input(shape=(input_shape[0], 1), dtype='float32',
                              name=f'Input_ch{i+1}')
        
        # Agregando a la lista
        x_in_list.append(x_in)

        # Definición de la capa de máscara
        x_masked = tf.keras.layers.Masking(mask_value=padding_value, 
                                           input_shape=(input_shape[0], 1),
                                           name=f'Masking_ch{i+1}')(x_in)
    
        # Definición de las capas convolucionales
        ### Primer conv ###
        x_conv1 = tf.keras.layers.Conv1D(filters=5, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='he_normal', 
                                        name=f'Conv1_ch{i+1}')(x_masked)
        x_conv1_norm = tf.keras.layers.BatchNormalization()(x_conv1)

        ### Segundo conv ###
        x_conv2 = tf.keras.layers.Conv1D(filters=10, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='he_normal', 
                                        name=f'Conv2_ch{i+1}')(x_conv1_norm)
        x_conv2_norm = tf.keras.layers.BatchNormalization()(x_conv2)

        ### Tercer conv ###
        x_conv3 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='he_normal', 
                                        name=f'Conv3_ch{i+1}')(x_conv2_norm)
        x_conv3_norm = tf.keras.layers.BatchNormalization()(x_conv3)

        ### Cuarto conv ###
        x_conv4 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='he_normal', 
                                        name=f'Conv4_ch{i+1}')(x_conv3_norm)
        x_conv4_norm = tf.keras.layers.BatchNormalization()(x_conv4)
        
        
        # Definición de la capa de aplanamiento para conectar la CNN con la FCL 
        x_flat = tf.keras.layers.Flatten()(x_conv4_norm)
        
        
        # Definición de las capas fully connected
        ### Fully Connected 1 ###
        x_full1 = tf.keras.layers.Dense(units=30, activation='relu',
                                        kernel_initializer='he_normal',
                                        name=f'FC1_ch{i+1}')(x_flat)
        x_full1 = tf.keras.layers.BatchNormalization()(x_full1)

        ### Fully Connected 2 ###
        x_full2 = tf.keras.layers.Dense(units=80, activation='relu',
                                        kernel_initializer='he_normal',
                                        name=f'FC2_ch{i+1}')(x_full1)
        x_full2 = tf.keras.layers.BatchNormalization()(x_full2)

        ### Fully Connected 3 ###
        x_full3 = tf.keras.layers.Dense(units=70, activation='relu',
                                        kernel_initializer='he_normal',
                                        name=f'FC3_ch{i+1}')(x_full2)
        x_full3 = tf.keras.layers.BatchNormalization()(x_full3)

        # Definición de las 2 capas de salida
        s1_lay = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                       kernel_initializer='he_normal',
                                       name=f'S1_out_ch{i+1}')(x_full3)
        s2_lay = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                       kernel_initializer='he_normal',
                                       name=f'S2_out_ch{i+1}')(x_full3)
                
        # Agregando a las listas
        s12_list.append(s1_lay)
        s12_list.append(s2_lay)
    
    # Definición de una capa de aplanamiento
    s12_concat = tf.keras.layers.concatenate(s12_list, name='s12_concat')
    
    # Definición de las 2 capas de salida
    x_out = tf.keras.layers.Dense(units=3, activation='softmax', 
                                  kernel_initializer='he_normal',
                                  name='Heart_out')(s12_concat)
    
    # Creación del modelo final
    model = tf.keras.Model(x_in_list, x_out)
    
    return model


def model_5_2_4_3(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    Básicamente es el modelo 5_2_4_2, pero usando como salida una capa softmax con
    solo 2 estados.
    
    Envolventes usadas:
    - Sonido raw
    - Filtros homomórficos
    - Envolvente de Hilbert de magnitud
    - Wavelets multiescala
    - Spectral tracking
    - Energía por bandas
    - Variance Fractal Dimension
    
    Parameters
    ----------
    input_shape : tuple or list
        Dimensión de un ejemplo de entrada.
    padding_value : int
        Valor con el que se realiza el padding de los últimos segmentos de cada sonido.
    name : str or NoneType, optional
        Nombre del modelo. Por defecto es None.
        
    Returns
    -------
    model : tensorflow.keras.Model
        Modelo final de keras.
    
    References
    ----------
    [1] Gupta, C. N., Palaniappan, R., Rajan, S., Swaminathan, S., 
        & Krishnan, S. M. (2005, May). Segmentation and classification 
        of heart sounds. In Canadian Conference on Electrical and 
        Computer Engineering, 2005. (pp. 1674-1677). IEEE.
    '''    
    # Definición de una lista auxiliar de las entradas y de capas FC de salida
    x_in_list = list()
    s12_list = list()
    
    # Aplicando una separación para cada canal
    for i in range(input_shape[1]):        
        # Definición de la entrada
        x_in = tf.keras.Input(shape=(input_shape[0], 1), dtype='float32',
                              name=f'Input_ch{i+1}')
        
        # Agregando a la lista
        x_in_list.append(x_in)

        # Definición de la capa de máscara
        x_masked = tf.keras.layers.Masking(mask_value=padding_value, 
                                           input_shape=(input_shape[0], 1),
                                           name=f'Masking_ch{i+1}')(x_in)
    
        # Definición de las capas convolucionales
        ### Primer conv ###
        x_conv1 = tf.keras.layers.Conv1D(filters=5, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='he_normal', 
                                        name=f'Conv1_ch{i+1}')(x_masked)
        x_conv1_norm = tf.keras.layers.BatchNormalization()(x_conv1)

        ### Segundo conv ###
        x_conv2 = tf.keras.layers.Conv1D(filters=10, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='he_normal', 
                                        name=f'Conv2_ch{i+1}')(x_conv1_norm)
        x_conv2_norm = tf.keras.layers.BatchNormalization()(x_conv2)

        ### Tercer conv ###
        x_conv3 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='he_normal', 
                                        name=f'Conv3_ch{i+1}')(x_conv2_norm)
        x_conv3_norm = tf.keras.layers.BatchNormalization()(x_conv3)

        ### Cuarto conv ###
        x_conv4 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='he_normal', 
                                        name=f'Conv4_ch{i+1}')(x_conv3_norm)
        x_conv4_norm = tf.keras.layers.BatchNormalization()(x_conv4)
        
        
        # Definición de la capa de aplanamiento para conectar la CNN con la FCL 
        x_flat = tf.keras.layers.Flatten()(x_conv4_norm)
        
        
        # Definición de las capas fully connected
        ### Fully Connected 1 ###
        x_full1 = tf.keras.layers.Dense(units=30, activation='relu',
                                        kernel_initializer='he_normal',
                                        name=f'FC1_ch{i+1}')(x_flat)
        x_full1 = tf.keras.layers.BatchNormalization()(x_full1)

        ### Fully Connected 2 ###
        x_full2 = tf.keras.layers.Dense(units=80, activation='relu',
                                        kernel_initializer='he_normal',
                                        name=f'FC2_ch{i+1}')(x_full1)
        x_full2 = tf.keras.layers.BatchNormalization()(x_full2)

        ### Fully Connected 3 ###
        x_full3 = tf.keras.layers.Dense(units=70, activation='relu',
                                        kernel_initializer='he_normal',
                                        name=f'FC3_ch{i+1}')(x_full2)
        x_full3 = tf.keras.layers.BatchNormalization()(x_full3)

        # Definición de las 2 capas de salida
        s12_lay = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                       kernel_initializer='he_normal',
                                       name=f'S12_out_ch{i+1}')(x_full3)
                
        # Agregando a las listas
        s12_list.append(s12_lay)
    
    # Definición de una capa de aplanamiento
    s12_concat = tf.keras.layers.concatenate(s12_list, name='s12_concat')
    
    # Definición de las 2 capas de salida
    x_out = tf.keras.layers.Dense(units=2, activation='softmax', 
                                  kernel_initializer='he_normal',
                                  name='Heart_out')(s12_concat)
    
    # Creación del modelo final
    model = tf.keras.Model(x_in_list, x_out)
    
    return model


def model_5_2_5(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    Básicamente es el modelo 5_2_1, pero sin la capa lambda de dispersión. En este
    caso se hace uso de una entrada distinta para cada envolvente, y además se aplica
    una etapa sigmoide (como en el model_2_1) extra en cada canal.
    
    Envolventes usadas:
    - Sonido raw
    - Filtros homomórficos
    - Envolventes de Hilbert
    - Wavelets multiescala
    - Spectral tracking
    - Simplicity envelope
    - Variance Fractal Dimension
    - FFT
    
    Parameters
    ----------
    input_shape : tuple or list
        Dimensión de un ejemplo de entrada.
    padding_value : int
        Valor con el que se realiza el padding de los últimos segmentos de cada sonido.
    name : str or NoneType, optional
        Nombre del modelo. Por defecto es None.
        
    Returns
    -------
    model : tensorflow.keras.Model
        Modelo final de keras.
    '''    
    # Definición de una lista auxiliar de las entradas y de capas FC de salida
    x_in_list = list()
    s1_list = list()
    s2_list = list()
    
    # Aplicando una separación para cada canal
    for i in range(input_shape[1]):        
        # Definición de la entrada
        x_in = tf.keras.Input(shape=(input_shape[0], 1), dtype='float32',
                              name=f'Input_ch{i+1}')
        
        # Agregando a la lista
        x_in_list.append(x_in)

        # Definición de la capa de máscara
        x_masked = tf.keras.layers.Masking(mask_value=padding_value, 
                                           input_shape=(input_shape[0], 1),
                                           name=f'Masking_ch{i+1}')(x_in)
    
        # Definición de las capas convolucionales
        ### Primer conv ###
        x_conv1 = tf.keras.layers.Conv1D(filters=5, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='glorot_uniform', 
                                        name=f'Conv1_ch{i+1}')(x_masked)
        x_conv1_norm = tf.keras.layers.BatchNormalization()(x_conv1)

        ### Segundo conv ###
        x_conv2 = tf.keras.layers.Conv1D(filters=10, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='glorot_uniform', 
                                        name=f'Conv2_ch{i+1}')(x_conv1_norm)
        x_conv2_norm = tf.keras.layers.BatchNormalization()(x_conv2)

        ### Tercer conv ###
        x_conv3 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='glorot_uniform', 
                                        name=f'Conv3_ch{i+1}')(x_conv2_norm)
        x_conv3_norm = tf.keras.layers.BatchNormalization()(x_conv3)

        ### Cuarto conv ###
        x_conv4 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                        activation='relu', kernel_initializer='glorot_uniform', 
                                        name=f'Conv4_ch{i+1}')(x_conv3_norm)
        x_conv4_norm = tf.keras.layers.BatchNormalization()(x_conv4)
        
        
        # Definición de la capa de aplanamiento para conectar la CNN con la FCL 
        x_flat = tf.keras.layers.Flatten()(x_conv4_norm)
        
        
        # Definición de las capas fully connected
        ### Fully Connected 1 ###
        x_full1 = tf.keras.layers.Dense(units=30, activation='relu',
                                        kernel_initializer='glorot_uniform',
                                        name=f'FC1_ch{i+1}')(x_flat)
        x_full1 = tf.keras.layers.BatchNormalization()(x_full1)

        ### Fully Connected 2 ###
        x_full2 = tf.keras.layers.Dense(units=80, activation='relu',
                                        kernel_initializer='glorot_uniform',
                                        name=f'FC2_ch{i+1}')(x_full1)
        x_full2 = tf.keras.layers.BatchNormalization()(x_full2)

        ### Fully Connected 3 ###
        x_full3 = tf.keras.layers.Dense(units=70, activation='relu',
                                        kernel_initializer='glorot_uniform',
                                        name=f'FC3_ch{i+1}')(x_full2)
        x_full3 = tf.keras.layers.BatchNormalization()(x_full3)

        # Definición de las 2 capas de salida
        s1_lay = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                       kernel_initializer='glorot_uniform',
                                       name=f'S1_out_ch{i+1}')(x_full3)
        s2_lay = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                       kernel_initializer='glorot_uniform',
                                       name=f'S2_out_ch{i+1}')(x_full3)
                
        # Agregando a las listas
        s1_list.append(s1_lay)
        s2_list.append(s2_lay)
    
    # Definición de una capa de aplanamiento
    s1_concat = tf.keras.layers.concatenate(s1_list, name='s1_concat')
    s2_concat = tf.keras.layers.concatenate(s2_list, name='s2_concat')
    
    
    # Definición de las 2 capas de salida
    x_out_s1 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='glorot_uniform',
                                     name='S1_out')(s1_concat)
    x_out_s2 = tf.keras.layers.Dense(units=1, activation='sigmoid', 
                                     kernel_initializer='glorot_uniform',
                                     name='S2_out')(s2_concat)
    
    # Creación del modelo final
    model = tf.keras.Model(x_in_list, [x_out_s1, x_out_s2])
    
    return model


def model_5_2_6(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    Básicamente es el modelo 5_2_5, pero con ciertas modificaciones del orden de cada
    capa. En primer lugar, se hace la capa CNN o FC, luego se aplica normalización, y
    luego la activación ReLu o sigmoide.
    Anterior: Capa -> Activación -> Normalización
    Ahora:    Capa -> Normalización -> Activación
    
    Envolventes usadas:
    - Sonido raw
    - Filtros homomórficos
    - Envolventes de Hilbert
    - Wavelets multiescala
    - Spectral tracking
    - Simplicity envelope
    - Variance Fractal Dimension
    - FFT
    
    Parameters
    ----------
    input_shape : tuple or list
        Dimensión de un ejemplo de entrada.
    padding_value : int
        Valor con el que se realiza el padding de los últimos segmentos de cada sonido.
    name : str or NoneType, optional
        Nombre del modelo. Por defecto es None.
        
    Returns
    -------
    model : tensorflow.keras.Model
        Modelo final de keras.
    '''    
    # Definición de una lista auxiliar de las entradas y de capas FC de salida
    x_in_list = list()
    s1_list = list()
    s2_list = list()
    
    # Aplicando una separación para cada canal
    for i in range(input_shape[1]):        
        # Definición de la entrada
        x_in = tf.keras.Input(shape=(input_shape[0], 1), dtype='float32',
                              name=f'Input_ch{i+1}')
        
        # Agregando a la lista
        x_in_list.append(x_in)

        # Definición de la capa de máscara
        x_masked = tf.keras.layers.Masking(mask_value=padding_value, 
                                           input_shape=(input_shape[0], 1),
                                           name=f'Masking_ch{i+1}')(x_in)

        # Definición de las capas convolucionales
        ### Primer conv ###
        x_conv1 = tf.keras.layers.Conv1D(filters=5, kernel_size=100, padding='same', 
                                         kernel_initializer='glorot_uniform', 
                                         name=f'Conv1_ch{i+1}')(x_masked)
        x_conv1 = tf.keras.layers.BatchNormalization(name=f'Batchnorm_Conv1_ch{i+1}')(x_conv1)
        x_conv1 = tf.keras.layers.Activation('relu', name=f'Act_Conv1_ch{i+1}')(x_conv1)

        ### Segundo conv ###
        x_conv2 = tf.keras.layers.Conv1D(filters=10, kernel_size=100, padding='same', 
                                         kernel_initializer='glorot_uniform', 
                                         name=f'Conv2_ch{i+1}')(x_conv1)
        x_conv2 = tf.keras.layers.BatchNormalization(name=f'Batchnorm_Conv2_ch{i+1}')(x_conv2)
        x_conv2 = tf.keras.layers.Activation('relu', name=f'Act_Conv2_ch{i+1}')(x_conv2)

        ### Tercer conv ###
        x_conv3 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                         kernel_initializer='glorot_uniform', 
                                         name=f'Conv3_ch{i+1}')(x_conv2)
        x_conv3 = tf.keras.layers.BatchNormalization(name=f'Batchnorm_Conv3_ch{i+1}')(x_conv3)
        x_conv3 = tf.keras.layers.Activation('relu', name=f'Act_Conv3_ch{i+1}')(x_conv3)

        ### Cuarto conv ###
        x_conv4 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                         kernel_initializer='glorot_uniform', 
                                         name=f'Conv4_ch{i+1}')(x_conv3)
        x_conv4 = tf.keras.layers.BatchNormalization(name=f'Batchnorm_Conv4_ch{i+1}')(x_conv4)
        x_conv4 = tf.keras.layers.Activation('relu', name=f'Act_Conv4_ch{i+1}')(x_conv4)
        
        
        # Definición de la capa de aplanamiento para conectar la CNN con la FCL 
        x_flat = tf.keras.layers.Flatten()(x_conv4)
        
        
        # Definición de las capas fully connected
        ### Fully Connected 1 ###
        x_full1 = tf.keras.layers.Dense(units=30, kernel_initializer='glorot_uniform',
                                        name=f'FC1_ch{i+1}')(x_flat)
        x_full1 = tf.keras.layers.BatchNormalization(name=f'Batchnorm_FC1_ch{i+1}')(x_full1)
        x_full1 = tf.keras.layers.Activation('relu', name=f'Act_FC1_ch{i+1}')(x_full1)

        ### Fully Connected 2 ###
        x_full2 = tf.keras.layers.Dense(units=80, kernel_initializer='glorot_uniform',
                                        name=f'FC2_ch{i+1}')(x_full1)
        x_full2 = tf.keras.layers.BatchNormalization(name=f'Batchnorm_FC2_ch{i+1}')(x_full2)
        x_full2 = tf.keras.layers.Activation('relu', name=f'Act_FC2_ch{i+1}')(x_full2)

        ### Fully Connected 3 ###
        x_full3 = tf.keras.layers.Dense(units=70, kernel_initializer='glorot_uniform',
                                        name=f'FC3_ch{i+1}')(x_full2)
        x_full3 = tf.keras.layers.BatchNormalization(name=f'Batchnorm_FC3_ch{i+1}')(x_full3)
        x_full3 = tf.keras.layers.Activation('relu', name=f'Act_FC3_ch{i+1}')(x_full3)

        
        ### Output layers por canal ###
        # Definición de la capa de salida S1 para el canal i
        s1_lay = tf.keras.layers.Dense(units=1, kernel_initializer='glorot_uniform',
                                       name=f'Dense_S1_ch{i+1}')(x_full3)
        s1_lay = tf.keras.layers.BatchNormalization(name=f'Batchnorm_S1_ch{i+1}')(s1_lay)
        s1_lay = tf.keras.layers.Activation('sigmoid', name=f'Act_S1_ch{i+1}')(s1_lay)
        
        # Definición de la capa de salida S2
        s2_lay = tf.keras.layers.Dense(units=1, kernel_initializer='glorot_uniform',
                                       name=f'Dense_S2_ch{i+1}')(x_full3)
        s2_lay = tf.keras.layers.BatchNormalization(name=f'Batchnorm_S2_ch{i+1}')(s2_lay)
        s2_lay = tf.keras.layers.Activation('sigmoid', name=f'Act_S2_ch{i+1}')(s2_lay)
        
        # Agregando a las listas
        s1_list.append(s1_lay)
        s2_list.append(s2_lay)

    ### Final layers ###
    
    # Definición de una capa de aplanamiento
    s1_concat = tf.keras.layers.concatenate(s1_list, name='s1_concat')
    s2_concat = tf.keras.layers.concatenate(s2_list, name='s2_concat')
    
    
    # Salida final para S1
    x_out_s1 = tf.keras.layers.Dense(units=1, kernel_initializer='glorot_uniform',
                                     name='S1_out_Dense')(s1_concat)
    x_out_s1 = tf.keras.layers.BatchNormalization(name='S1_out_Batchnorm')(x_out_s1)
    x_out_s1 = tf.keras.layers.Activation('sigmoid', name='S1_out_Act')(x_out_s1)
    
    # Salida final para S2
    x_out_s2 = tf.keras.layers.Dense(units=1, kernel_initializer='glorot_uniform',
                                     name='S2_out_Dense')(s2_concat)
    x_out_s2 = tf.keras.layers.BatchNormalization(name='S2_out_Batchnorm')(x_out_s2)
    x_out_s2 = tf.keras.layers.Activation('sigmoid', name='S2_out_Act')(x_out_s2)
    
    # Creación del modelo final
    model = tf.keras.Model(x_in_list, [x_out_s1, x_out_s2])
    
    return model


def model_5_2_7(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    Básicamente es el modelo 5_2_6, pero ahora con dropout a la salida de la capa 
    de activación.
    
    Envolventes usadas:
    - Sonido raw
    - Filtros homomórficos
    - Envolventes de Hilbert
    - Wavelets multiescala
    - Spectral tracking
    - Simplicity envelope
    - Variance Fractal Dimension
    - FFT
    
    Parameters
    ----------
    input_shape : tuple or list
        Dimensión de un ejemplo de entrada.
    padding_value : int
        Valor con el que se realiza el padding de los últimos segmentos de cada sonido.
    name : str or NoneType, optional
        Nombre del modelo. Por defecto es None.
        
    Returns
    -------
    model : tensorflow.keras.Model
        Modelo final de keras.
    '''    
    # Definición de una lista auxiliar de las entradas y de capas FC de salida
    x_in_list = list()
    s1_list = list()
    s2_list = list()
    
    # Aplicando una separación para cada canal
    for i in range(input_shape[1]):        
        # Definición de la entrada
        x_in = tf.keras.Input(shape=(input_shape[0], 1), dtype='float32',
                              name=f'Input_ch{i+1}')
        
        # Agregando a la lista
        x_in_list.append(x_in)

        # Definición de la capa de máscara
        x_masked = tf.keras.layers.Masking(mask_value=padding_value, 
                                           input_shape=(input_shape[0], 1),
                                           name=f'Masking_ch{i+1}')(x_in)

        # Definición de las capas convolucionales
        ### Primer conv ###
        x_conv1 = tf.keras.layers.Conv1D(filters=5, kernel_size=100, padding='same', 
                                         kernel_initializer='glorot_uniform', 
                                         name=f'Conv1_ch{i+1}')(x_masked)
        x_conv1 = tf.keras.layers.BatchNormalization(name=f'Batchnorm_Conv1_ch{i+1}')(x_conv1)
        x_conv1 = tf.keras.layers.Activation('relu', name=f'Act_Conv1_ch{i+1}')(x_conv1)
        x_conv1 = tf.keras.layers.Dropout(0.1)(x_conv1)

        
        ### Segundo conv ###
        x_conv2 = tf.keras.layers.Conv1D(filters=10, kernel_size=100, padding='same', 
                                         kernel_initializer='glorot_uniform', 
                                         name=f'Conv2_ch{i+1}')(x_conv1)
        x_conv2 = tf.keras.layers.BatchNormalization(name=f'Batchnorm_Conv2_ch{i+1}')(x_conv2)
        x_conv2 = tf.keras.layers.Activation('relu', name=f'Act_Conv2_ch{i+1}')(x_conv2)
        x_conv2 = tf.keras.layers.Dropout(0.1)(x_conv2)

        
        ### Tercer conv ###
        x_conv3 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                         kernel_initializer='glorot_uniform', 
                                         name=f'Conv3_ch{i+1}')(x_conv2)
        x_conv3 = tf.keras.layers.BatchNormalization(name=f'Batchnorm_Conv3_ch{i+1}')(x_conv3)
        x_conv3 = tf.keras.layers.Activation('relu', name=f'Act_Conv3_ch{i+1}')(x_conv3)
        x_conv3 = tf.keras.layers.Dropout(0.1)(x_conv3)

        
        ### Cuarto conv ###
        x_conv4 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                         kernel_initializer='glorot_uniform', 
                                         name=f'Conv4_ch{i+1}')(x_conv3)
        x_conv4 = tf.keras.layers.BatchNormalization(name=f'Batchnorm_Conv4_ch{i+1}')(x_conv4)
        x_conv4 = tf.keras.layers.Activation('relu', name=f'Act_Conv4_ch{i+1}')(x_conv4)
        x_conv4 = tf.keras.layers.Dropout(0.1)(x_conv4)
        
        # Definición de la capa de aplanamiento para conectar la CNN con la FCL 
        x_flat = tf.keras.layers.Flatten()(x_conv4)
        
        
        # Definición de las capas fully connected
        ### Fully Connected 1 ###
        x_full1 = tf.keras.layers.Dense(units=30, kernel_initializer='glorot_uniform',
                                        name=f'FC1_ch{i+1}')(x_flat)
        x_full1 = tf.keras.layers.BatchNormalization(name=f'Batchnorm_FC1_ch{i+1}')(x_full1)
        x_full1 = tf.keras.layers.Activation('relu', name=f'Act_FC1_ch{i+1}')(x_full1)
        x_full1 = tf.keras.layers.Dropout(0.1)(x_full1)

        ### Fully Connected 2 ###
        x_full2 = tf.keras.layers.Dense(units=80, kernel_initializer='glorot_uniform',
                                        name=f'FC2_ch{i+1}')(x_full1)
        x_full2 = tf.keras.layers.BatchNormalization(name=f'Batchnorm_FC2_ch{i+1}')(x_full2)
        x_full2 = tf.keras.layers.Activation('relu', name=f'Act_FC2_ch{i+1}')(x_full2)
        x_full2 = tf.keras.layers.Dropout(0.1)(x_full2)

        ### Fully Connected 3 ###
        x_full3 = tf.keras.layers.Dense(units=70, kernel_initializer='glorot_uniform',
                                        name=f'FC3_ch{i+1}')(x_full2)
        x_full3 = tf.keras.layers.BatchNormalization(name=f'Batchnorm_FC3_ch{i+1}')(x_full3)
        x_full3 = tf.keras.layers.Activation('relu', name=f'Act_FC3_ch{i+1}')(x_full3)
        x_full3 = tf.keras.layers.Dropout(0.1)(x_full3)

        
        ### Output layers por canal ###
        # Definición de la capa de salida S1 para el canal i
        s1_lay = tf.keras.layers.Dense(units=1, kernel_initializer='glorot_uniform',
                                       name=f'Dense_S1_ch{i+1}')(x_full3)
        s1_lay = tf.keras.layers.BatchNormalization(name=f'Batchnorm_S1_ch{i+1}')(s1_lay)
        s1_lay = tf.keras.layers.Activation('sigmoid', name=f'Act_S1_ch{i+1}')(s1_lay)
        
        # Definición de la capa de salida S2
        s2_lay = tf.keras.layers.Dense(units=1, kernel_initializer='glorot_uniform',
                                       name=f'Dense_S2_ch{i+1}')(x_full3)
        s2_lay = tf.keras.layers.BatchNormalization(name=f'Batchnorm_S2_ch{i+1}')(s2_lay)
        s2_lay = tf.keras.layers.Activation('sigmoid', name=f'Act_S2_ch{i+1}')(s2_lay)
        
        # Agregando a las listas
        s1_list.append(s1_lay)
        s2_list.append(s2_lay)

    ### Final layers ###
    
    # Definición de una capa de aplanamiento
    s1_concat = tf.keras.layers.concatenate(s1_list, name='s1_concat')
    s2_concat = tf.keras.layers.concatenate(s2_list, name='s2_concat')
    
    
    # Salida final para S1
    x_out_s1 = tf.keras.layers.Dense(units=1, kernel_initializer='glorot_uniform',
                                     name='S1_out_Dense')(s1_concat)
    x_out_s1 = tf.keras.layers.BatchNormalization(name='S1_out_Batchnorm')(x_out_s1)
    x_out_s1 = tf.keras.layers.Activation('sigmoid', name='S1_out_Act')(x_out_s1)
    
    # Salida final para S2
    x_out_s2 = tf.keras.layers.Dense(units=1, kernel_initializer='glorot_uniform',
                                     name='S2_out_Dense')(s2_concat)
    x_out_s2 = tf.keras.layers.BatchNormalization(name='S2_out_Batchnorm')(x_out_s2)
    x_out_s2 = tf.keras.layers.Activation('sigmoid', name='S2_out_Act')(x_out_s2)
    
    # Creación del modelo final
    model = tf.keras.Model(x_in_list, [x_out_s1, x_out_s2])
    
    return model


def model_5_2_8(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    Básicamente es el modelo 5_2_7, pero ahora con dropout 0.2.
    
    Envolventes usadas:
    - Sonido raw
    - Filtros homomórficos
    - Envolventes de Hilbert
    - Wavelets multiescala
    - Spectral tracking
    - Simplicity envelope
    - Variance Fractal Dimension
    - FFT
    
    Parameters
    ----------
    input_shape : tuple or list
        Dimensión de un ejemplo de entrada.
    padding_value : int
        Valor con el que se realiza el padding de los últimos segmentos de cada sonido.
    name : str or NoneType, optional
        Nombre del modelo. Por defecto es None.
        
    Returns
    -------
    model : tensorflow.keras.Model
        Modelo final de keras.
    '''    
    # Definición de una lista auxiliar de las entradas y de capas FC de salida
    x_in_list = list()
    s1_list = list()
    s2_list = list()
    
    # Aplicando una separación para cada canal
    for i in range(input_shape[1]):        
        # Definición de la entrada
        x_in = tf.keras.Input(shape=(input_shape[0], 1), dtype='float32',
                              name=f'Input_ch{i+1}')
        
        # Agregando a la lista
        x_in_list.append(x_in)

        # Definición de la capa de máscara
        x_masked = tf.keras.layers.Masking(mask_value=padding_value, 
                                           input_shape=(input_shape[0], 1),
                                           name=f'Masking_ch{i+1}')(x_in)

        # Definición de las capas convolucionales
        ### Primer conv ###
        x_conv1 = tf.keras.layers.Conv1D(filters=5, kernel_size=100, padding='same', 
                                         kernel_initializer='glorot_uniform', 
                                         name=f'Conv1_ch{i+1}')(x_masked)
        x_conv1 = tf.keras.layers.BatchNormalization(name=f'Batchnorm_Conv1_ch{i+1}')(x_conv1)
        x_conv1 = tf.keras.layers.Activation('relu', name=f'Act_Conv1_ch{i+1}')(x_conv1)
        x_conv1 = tf.keras.layers.Dropout(0.2)(x_conv1)

        
        ### Segundo conv ###
        x_conv2 = tf.keras.layers.Conv1D(filters=10, kernel_size=100, padding='same', 
                                         kernel_initializer='glorot_uniform', 
                                         name=f'Conv2_ch{i+1}')(x_conv1)
        x_conv2 = tf.keras.layers.BatchNormalization(name=f'Batchnorm_Conv2_ch{i+1}')(x_conv2)
        x_conv2 = tf.keras.layers.Activation('relu', name=f'Act_Conv2_ch{i+1}')(x_conv2)
        x_conv2 = tf.keras.layers.Dropout(0.2)(x_conv2)

        
        ### Tercer conv ###
        x_conv3 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                         kernel_initializer='glorot_uniform', 
                                         name=f'Conv3_ch{i+1}')(x_conv2)
        x_conv3 = tf.keras.layers.BatchNormalization(name=f'Batchnorm_Conv3_ch{i+1}')(x_conv3)
        x_conv3 = tf.keras.layers.Activation('relu', name=f'Act_Conv3_ch{i+1}')(x_conv3)
        x_conv3 = tf.keras.layers.Dropout(0.2)(x_conv3)

        
        ### Cuarto conv ###
        x_conv4 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                         kernel_initializer='glorot_uniform', 
                                         name=f'Conv4_ch{i+1}')(x_conv3)
        x_conv4 = tf.keras.layers.BatchNormalization(name=f'Batchnorm_Conv4_ch{i+1}')(x_conv4)
        x_conv4 = tf.keras.layers.Activation('relu', name=f'Act_Conv4_ch{i+1}')(x_conv4)
        x_conv4 = tf.keras.layers.Dropout(0.2)(x_conv4)
        
        # Definición de la capa de aplanamiento para conectar la CNN con la FCL 
        x_flat = tf.keras.layers.Flatten()(x_conv4)
        
        
        # Definición de las capas fully connected
        ### Fully Connected 1 ###
        x_full1 = tf.keras.layers.Dense(units=30, kernel_initializer='glorot_uniform',
                                        name=f'FC1_ch{i+1}')(x_flat)
        x_full1 = tf.keras.layers.BatchNormalization(name=f'Batchnorm_FC1_ch{i+1}')(x_full1)
        x_full1 = tf.keras.layers.Activation('relu', name=f'Act_FC1_ch{i+1}')(x_full1)
        x_full1 = tf.keras.layers.Dropout(0.2)(x_full1)

        ### Fully Connected 2 ###
        x_full2 = tf.keras.layers.Dense(units=80, kernel_initializer='glorot_uniform',
                                        name=f'FC2_ch{i+1}')(x_full1)
        x_full2 = tf.keras.layers.BatchNormalization(name=f'Batchnorm_FC2_ch{i+1}')(x_full2)
        x_full2 = tf.keras.layers.Activation('relu', name=f'Act_FC2_ch{i+1}')(x_full2)
        x_full2 = tf.keras.layers.Dropout(0.2)(x_full2)

        ### Fully Connected 3 ###
        x_full3 = tf.keras.layers.Dense(units=70, kernel_initializer='glorot_uniform',
                                        name=f'FC3_ch{i+1}')(x_full2)
        x_full3 = tf.keras.layers.BatchNormalization(name=f'Batchnorm_FC3_ch{i+1}')(x_full3)
        x_full3 = tf.keras.layers.Activation('relu', name=f'Act_FC3_ch{i+1}')(x_full3)
        x_full3 = tf.keras.layers.Dropout(0.2)(x_full3)

        
        ### Output layers por canal ###
        # Definición de la capa de salida S1 para el canal i
        s1_lay = tf.keras.layers.Dense(units=1, kernel_initializer='glorot_uniform',
                                       name=f'Dense_S1_ch{i+1}')(x_full3)
        s1_lay = tf.keras.layers.BatchNormalization(name=f'Batchnorm_S1_ch{i+1}')(s1_lay)
        s1_lay = tf.keras.layers.Activation('sigmoid', name=f'Act_S1_ch{i+1}')(s1_lay)
        
        # Definición de la capa de salida S2
        s2_lay = tf.keras.layers.Dense(units=1, kernel_initializer='glorot_uniform',
                                       name=f'Dense_S2_ch{i+1}')(x_full3)
        s2_lay = tf.keras.layers.BatchNormalization(name=f'Batchnorm_S2_ch{i+1}')(s2_lay)
        s2_lay = tf.keras.layers.Activation('sigmoid', name=f'Act_S2_ch{i+1}')(s2_lay)
        
        # Agregando a las listas
        s1_list.append(s1_lay)
        s2_list.append(s2_lay)

    ### Final layers ###
    
    # Definición de una capa de aplanamiento
    s1_concat = tf.keras.layers.concatenate(s1_list, name='s1_concat')
    s2_concat = tf.keras.layers.concatenate(s2_list, name='s2_concat')
    
    
    # Salida final para S1
    x_out_s1 = tf.keras.layers.Dense(units=1, kernel_initializer='glorot_uniform',
                                     name='S1_out_Dense')(s1_concat)
    x_out_s1 = tf.keras.layers.BatchNormalization(name='S1_out_Batchnorm')(x_out_s1)
    x_out_s1 = tf.keras.layers.Activation('sigmoid', name='S1_out_Act')(x_out_s1)
    
    # Salida final para S2
    x_out_s2 = tf.keras.layers.Dense(units=1, kernel_initializer='glorot_uniform',
                                     name='S2_out_Dense')(s2_concat)
    x_out_s2 = tf.keras.layers.BatchNormalization(name='S2_out_Batchnorm')(x_out_s2)
    x_out_s2 = tf.keras.layers.Activation('sigmoid', name='S2_out_Act')(x_out_s2)
    
    # Creación del modelo final
    model = tf.keras.Model(x_in_list, [x_out_s1, x_out_s2])
    
    return model


def model_5_2_9(input_shape, padding_value, name=None):
    '''Intento de modelo Convolutional Neural Networks + Fully Connected Layers 1.
    Básicamente es el modelo 5_2_8, pero ahora con dropout 0.2 y maxpooling
    
    Envolventes usadas:
    - Sonido raw
    - Filtros homomórficos
    - Envolventes de Hilbert
    - Wavelets multiescala
    - Spectral tracking
    - Simplicity envelope
    - Variance Fractal Dimension
    - FFT
    
    Parameters
    ----------
    input_shape : tuple or list
        Dimensión de un ejemplo de entrada.
    padding_value : int
        Valor con el que se realiza el padding de los últimos segmentos de cada sonido.
    name : str or NoneType, optional
        Nombre del modelo. Por defecto es None.
        
    Returns
    -------
    model : tensorflow.keras.Model
        Modelo final de keras.
    '''    
    # Definición de una lista auxiliar de las entradas y de capas FC de salida
    x_in_list = list()
    s1_list = list()
    s2_list = list()
    
    # Aplicando una separación para cada canal
    for i in range(input_shape[1]):        
        # Definición de la entrada
        x_in = tf.keras.Input(shape=(input_shape[0], 1), dtype='float32',
                              name=f'Input_ch{i+1}')
        
        # Agregando a la lista
        x_in_list.append(x_in)

        # Definición de la capa de máscara
        x_masked = tf.keras.layers.Masking(mask_value=padding_value, 
                                           input_shape=(input_shape[0], 1),
                                           name=f'Masking_ch{i+1}')(x_in)

        # Definición de las capas convolucionales
        ### Primer conv ###
        x_conv1 = tf.keras.layers.Conv1D(filters=5, kernel_size=100, padding='same', 
                                         kernel_initializer='glorot_uniform', 
                                         name=f'Conv1_ch{i+1}')(x_masked)
        x_conv1 = tf.keras.layers.BatchNormalization(name=f'Batchnorm_Conv1_ch{i+1}')(x_conv1)
        x_conv1 = tf.keras.layers.Activation('relu', name=f'Act_Conv1_ch{i+1}')(x_conv1)
        x_conv1 = tf.keras.layers.Dropout(0.2, name=f'Drop_Conv1_ch{i+1}')(x_conv1)
        x_conv1 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='valid',
                                               name=f'MaxPool_Conv1_ch{i+1}')(x_conv1)

        
        ### Segundo conv ###
        x_conv2 = tf.keras.layers.Conv1D(filters=10, kernel_size=100, padding='same', 
                                         kernel_initializer='glorot_uniform', 
                                         name=f'Conv2_ch{i+1}')(x_conv1)
        x_conv2 = tf.keras.layers.BatchNormalization(name=f'Batchnorm_Conv2_ch{i+1}')(x_conv2)
        x_conv2 = tf.keras.layers.Activation('relu', name=f'Act_Conv2_ch{i+1}')(x_conv2)
        x_conv2 = tf.keras.layers.Dropout(0.2, name=f'Drop_Conv2_ch{i+1}')(x_conv2)
        x_conv2 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='valid',
                                               name=f'MaxPool_Conv2_ch{i+1}')(x_conv2)

        
        ### Tercer conv ###
        x_conv3 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                         kernel_initializer='glorot_uniform', 
                                         name=f'Conv3_ch{i+1}')(x_conv2)
        x_conv3 = tf.keras.layers.BatchNormalization(name=f'Batchnorm_Conv3_ch{i+1}')(x_conv3)
        x_conv3 = tf.keras.layers.Activation('relu', name=f'Act_Conv3_ch{i+1}')(x_conv3)
        x_conv3 = tf.keras.layers.Dropout(0.2, name=f'Drop_Conv3_ch{i+1}')(x_conv3)
        x_conv3 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='valid',
                                               name=f'MaxPool_Conv3_ch{i+1}')(x_conv3)

        
        ### Cuarto conv ###
        x_conv4 = tf.keras.layers.Conv1D(filters=20, kernel_size=100, padding='same', 
                                         kernel_initializer='glorot_uniform', 
                                         name=f'Conv4_ch{i+1}')(x_conv3)
        x_conv4 = tf.keras.layers.BatchNormalization(name=f'Batchnorm_Conv4_ch{i+1}')(x_conv4)
        x_conv4 = tf.keras.layers.Activation('relu', name=f'Act_Conv4_ch{i+1}')(x_conv4)
        x_conv4 = tf.keras.layers.Dropout(0.2, name=f'Drop_Conv4_ch{i+1}')(x_conv4)
        x_conv4 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='valid',
                                               name=f'MaxPool_Conv4_ch{i+1}')(x_conv4)
        
        # Definición de la capa de aplanamiento para conectar la CNN con la FCL 
        x_flat = tf.keras.layers.Flatten()(x_conv4)
        
        
        # Definición de las capas fully connected
        ### Fully Connected 1 ###
        x_full1 = tf.keras.layers.Dense(units=30, kernel_initializer='glorot_uniform',
                                        name=f'FC1_ch{i+1}')(x_flat)
        x_full1 = tf.keras.layers.BatchNormalization(name=f'Batchnorm_FC1_ch{i+1}')(x_full1)
        x_full1 = tf.keras.layers.Activation('relu', name=f'Act_FC1_ch{i+1}')(x_full1)
        x_full1 = tf.keras.layers.Dropout(0.2)(x_full1)

        ### Fully Connected 2 ###
        x_full2 = tf.keras.layers.Dense(units=80, kernel_initializer='glorot_uniform',
                                        name=f'FC2_ch{i+1}')(x_full1)
        x_full2 = tf.keras.layers.BatchNormalization(name=f'Batchnorm_FC2_ch{i+1}')(x_full2)
        x_full2 = tf.keras.layers.Activation('relu', name=f'Act_FC2_ch{i+1}')(x_full2)
        x_full2 = tf.keras.layers.Dropout(0.2)(x_full2)

        ### Fully Connected 3 ###
        x_full3 = tf.keras.layers.Dense(units=70, kernel_initializer='glorot_uniform',
                                        name=f'FC3_ch{i+1}')(x_full2)
        x_full3 = tf.keras.layers.BatchNormalization(name=f'Batchnorm_FC3_ch{i+1}')(x_full3)
        x_full3 = tf.keras.layers.Activation('relu', name=f'Act_FC3_ch{i+1}')(x_full3)
        x_full3 = tf.keras.layers.Dropout(0.2)(x_full3)

        
        ### Output layers por canal ###
        # Definición de la capa de salida S1 para el canal i
        s1_lay = tf.keras.layers.Dense(units=1, kernel_initializer='glorot_uniform',
                                       name=f'Dense_S1_ch{i+1}')(x_full3)
        s1_lay = tf.keras.layers.BatchNormalization(name=f'Batchnorm_S1_ch{i+1}')(s1_lay)
        s1_lay = tf.keras.layers.Activation('sigmoid', name=f'Act_S1_ch{i+1}')(s1_lay)
        
        # Definición de la capa de salida S2
        s2_lay = tf.keras.layers.Dense(units=1, kernel_initializer='glorot_uniform',
                                       name=f'Dense_S2_ch{i+1}')(x_full3)
        s2_lay = tf.keras.layers.BatchNormalization(name=f'Batchnorm_S2_ch{i+1}')(s2_lay)
        s2_lay = tf.keras.layers.Activation('sigmoid', name=f'Act_S2_ch{i+1}')(s2_lay)
        
        # Agregando a las listas
        s1_list.append(s1_lay)
        s2_list.append(s2_lay)

    ### Final layers ###
    
    # Definición de una capa de aplanamiento
    s1_concat = tf.keras.layers.concatenate(s1_list, name='s1_concat')
    s2_concat = tf.keras.layers.concatenate(s2_list, name='s2_concat')
    
    
    # Salida final para S1
    x_out_s1 = tf.keras.layers.Dense(units=1, kernel_initializer='glorot_uniform',
                                     name='S1_out_Dense')(s1_concat)
    x_out_s1 = tf.keras.layers.BatchNormalization(name='S1_out_Batchnorm')(x_out_s1)
    x_out_s1 = tf.keras.layers.Activation('sigmoid', name='S1_out_Act')(x_out_s1)
    
    # Salida final para S2
    x_out_s2 = tf.keras.layers.Dense(units=1, kernel_initializer='glorot_uniform',
                                     name='S2_out_Dense')(s2_concat)
    x_out_s2 = tf.keras.layers.BatchNormalization(name='S2_out_Batchnorm')(x_out_s2)
    x_out_s2 = tf.keras.layers.Activation('sigmoid', name='S2_out_Act')(x_out_s2)
    
    # Creación del modelo final
    model = tf.keras.Model(x_in_list, [x_out_s1, x_out_s2])
    
    return model


def model_6_1(input_shape, padding_value, name=None):
    '''CNN basada en arquitectura encoder-decoder basada en SegNet.
    
    References
    ----------
    [1] Badrinarayanan, V., Kendall, A., & Cipolla, R. (2017). 
        Segnet: A deep convolutional encoder-decoder architecture for 
        image segmentation. IEEE transactions on pattern analysis and 
        machine intelligence, 39(12), 2481-2495.
    [2] Ye, J. C., & Sung, W. K. (2019). Understanding geometry of 
        encoder-decoder CNNs. arXiv preprint arXiv:1901.07647.
    '''
    def _conv_bn_act_layer(input_layer, filters, kernel_size, padding,
                          kernel_initializer, name):
        '''Función auxiliar que modela las capas azules conv + batchnorm +
        Activation ReLU para realizar el ENCODING.'''
        # Aplicando la concatenación de capas
        x_conv = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, 
                                        kernel_initializer=kernel_initializer,
                                        padding=padding, 
                                        name=f'Conv_{name}')(input_layer)
        x_conv = \
            tf.keras.layers.BatchNormalization(name=f'BatchNorm_{name}')(x_conv)
        x_conv = \
            tf.keras.layers.Activation('relu', name=f'Activation_{name}')(x_conv)

        return x_conv
    
    
    def _encoding_layer(input_layer, n_layers_conv, layer_params):
        '''Función auxiliar que permite modelar "n_layers_conv" capas CNN seguida de 
        una capa de Maxpooling, tal como se puede ver en la figura 2 de [1].  
        '''
        # Definición de la salida de este bloque
        x_enc = input_layer
        
        # Aplicando "n_layers_conv" capas convolucionales de codificación
        for i in range(n_layers_conv):
            x_enc = _conv_bn_act_layer(x_enc, filters=layer_params['filters'], 
                                       kernel_size=layer_params['kernel_size'], 
                                       padding=layer_params['padding'],
                                       kernel_initializer=layer_params['kernel_initializer'], 
                                       name=f"{layer_params['name']}_{i}")

        # Finalmente la capa de MaxPooling
        x_enc = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, 
                                             padding='valid',
                                             name=f"MaxPool_Conv_{layer_params['name']}")(x_enc)
        return x_enc
    
    
    def _decoding_layer(input_layer, n_layers_conv, layer_params):
        '''Función auxiliar que permite modelar una capa de upsampling seguido de 
        "n_layers_conv" capas CNN, tal como se puede ver en la figura 2 de [1].  
        '''
        # Capa de upsampling
        x_dec = tf.keras.layers.UpSampling1D(size=2, name=f"Upsampling_"\
                                                          f"{layer_params['name']}")(input_layer)
        
        # Aplicando "n_layers_conv" capas convolucionales de decodificación
        for i in range(n_layers_conv):
            x_dec = _conv_bn_act_layer(x_dec, filters=layer_params['filters'], 
                                       kernel_size=layer_params['kernel_size'], 
                                       padding=layer_params['padding'],
                                       kernel_initializer=layer_params['kernel_initializer'], 
                                       name=f"{layer_params['name']}_{i}")

        return x_dec
    
    
    # Definición de la entrada
    x_in = tf.keras.Input(shape=input_shape, dtype='float32')

    # Definición de la capa de máscara
    x_masked = tf.keras.layers.Masking(mask_value=padding_value)(x_in)

    ############        Definición de las capas convolucionales        ############
    
    ### Encoding ###
    
    # Primera capa de encoding
    layer_params_1 = {'filters': input_shape[1] * 2, 'kernel_size': 50, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 'name': 'enc1'}
    x_enc1 = _encoding_layer(x_masked, n_layers_conv=2, layer_params=layer_params_1)
    
    # Segunda capa de encoding
    layer_params_2 = {'filters': input_shape[1] * 4, 'kernel_size': 50, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 'name': 'enc2'}
    x_enc2 = _encoding_layer(x_enc1, n_layers_conv=2, layer_params=layer_params_2)
    
    # Tercera capa de encoding
    layer_params_3 = {'filters': input_shape[1] * 8, 'kernel_size': 50, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 'name': 'enc3'}
    x_enc3 = _encoding_layer(x_enc2, n_layers_conv=3, layer_params=layer_params_3)
    
    # Cuarta capa de encoding
    layer_params_4 = {'filters': input_shape[1] * 16, 'kernel_size': 50, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 'name': 'enc4'}
    x_enc4 = _encoding_layer(x_enc3, n_layers_conv=3, layer_params=layer_params_4)
    
    
    ### Decoding ###
    
    # Cuarta capa de salida del decoding
    layer_params_4 = {'filters': input_shape[1] * 16, 'kernel_size': 50, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 'name': 'dec4'}
    x_dec4 = _decoding_layer(x_enc4, n_layers_conv=3, layer_params=layer_params_4)
    
    # Tercera capa de salida del decoding
    layer_params_3 = {'filters': input_shape[1] * 8, 'kernel_size': 50, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 'name': 'dec3'}
    x_dec3 = _decoding_layer(x_dec4, n_layers_conv=3, layer_params=layer_params_3)
    
    # Segunda capa de salida del decoding
    layer_params_2 = {'filters': input_shape[1] * 4, 'kernel_size': 50, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 'name': 'dec2'}
    x_dec2 = _decoding_layer(x_dec3, n_layers_conv=2, layer_params=layer_params_2)
    
    # Primera capa de salida del decoding
    layer_params_1 = {'filters': input_shape[1] * 2, 'kernel_size': 50, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 'name': 'dec1'}
    x_dec1 = _decoding_layer(x_dec2, n_layers_conv=2, layer_params=layer_params_1)
                                       
    
    # Aplicando reshape
    # x_reshaped = tf.keras.layers.Reshape((input_shape[0], input_shape[1] * 2))(x_dec1)
    
    # Definición de la capa de salida
    x_out = tf.keras.layers.Dense(3, activation='softmax', kernel_initializer='he_normal',
                                  name='softmax_out')(x_dec1)
    
    # Definición del modelo
    model = tf.keras.Model(inputs=x_in, outputs=x_out, name=name)
    
    return model


def model_6_2(input_shape, padding_value, name=None):
    '''CNN basada en arquitectura encoder-decoder basada en SegNet.
    En este caso se hace la misma red que 6_1, pero aplicando una
    regularización L2 con lambda = 0.01.
    
    References
    ----------
    [1] Badrinarayanan, V., Kendall, A., & Cipolla, R. (2017). 
        Segnet: A deep convolutional encoder-decoder architecture for 
        image segmentation. IEEE transactions on pattern analysis and 
        machine intelligence, 39(12), 2481-2495.
    [2] Ye, J. C., & Sung, W. K. (2019). Understanding geometry of 
        encoder-decoder CNNs. arXiv preprint arXiv:1901.07647.
    '''
    def _conv_bn_act_layer(input_layer, filters, kernel_size, padding,
                          kernel_initializer, kernel_regularizer, name):
        '''Función auxiliar que modela las capas azules conv + batchnorm +
        Activation ReLU para realizar el ENCODING.'''
        # Aplicando la concatenación de capas
        x_conv = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, 
                                        kernel_initializer=kernel_initializer,
                                        kernel_regularizer=kernel_regularizer,
                                        padding=padding, 
                                        name=f'Conv_{name}')(input_layer)
        x_conv = \
            tf.keras.layers.BatchNormalization(name=f'BatchNorm_{name}')(x_conv)
        x_conv = \
            tf.keras.layers.Activation('relu', name=f'Activation_{name}')(x_conv)

        return x_conv
    
    
    def _encoding_layer(input_layer, n_layers_conv, layer_params):
        '''Función auxiliar que permite modelar "n_layers_conv" capas CNN seguida de 
        una capa de Maxpooling, tal como se puede ver en la figura 2 de [1].  
        '''
        # Definición de la salida de este bloque
        x_enc = input_layer
        
        # Aplicando "n_layers_conv" capas convolucionales de codificación
        for i in range(n_layers_conv):
            x_enc = _conv_bn_act_layer(x_enc, filters=layer_params['filters'], 
                                       kernel_size=layer_params['kernel_size'], 
                                       padding=layer_params['padding'],
                                       kernel_initializer=layer_params['kernel_initializer'],
                                       kernel_regularizer=layer_params['kernel_regularizer'], 
                                       name=f"{layer_params['name']}_{i}")

        # Finalmente la capa de MaxPooling
        x_enc = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, 
                                             padding='valid',
                                             name=f"MaxPool_Conv_{layer_params['name']}")(x_enc)
        return x_enc
    
    
    def _decoding_layer(input_layer, n_layers_conv, layer_params):
        '''Función auxiliar que permite modelar una capa de upsampling seguido de 
        "n_layers_conv" capas CNN, tal como se puede ver en la figura 2 de [1].  
        '''
        # Capa de upsampling
        x_dec = tf.keras.layers.UpSampling1D(size=2, name=f"Upsampling_"\
                                                          f"{layer_params['name']}")(input_layer)
        
        # Aplicando "n_layers_conv" capas convolucionales de decodificación
        for i in range(n_layers_conv):
            x_dec = _conv_bn_act_layer(x_dec, filters=layer_params['filters'], 
                                       kernel_size=layer_params['kernel_size'], 
                                       padding=layer_params['padding'],
                                       kernel_initializer=layer_params['kernel_initializer'],
                                       kernel_regularizer=layer_params['kernel_regularizer'], 
                                       name=f"{layer_params['name']}_{i}")

        return x_dec
    
    
    # Definición de la entrada
    x_in = tf.keras.Input(shape=input_shape, dtype='float32')

    # Definición de la capa de máscara
    x_masked = tf.keras.layers.Masking(mask_value=padding_value)(x_in)

    ############        Definición de las capas convolucionales        ############
    
    ### Encoding ###
    
    # Primera capa de encoding
    layer_params_1 = {'filters': input_shape[1] * 2, 'kernel_size': 50, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 'kernel_regularizer': 'l2',
                      'name': 'enc1'}
    x_enc1 = _encoding_layer(x_masked, n_layers_conv=2, layer_params=layer_params_1)
    
    # Segunda capa de encoding
    layer_params_2 = {'filters': input_shape[1] * 4, 'kernel_size': 50, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 'kernel_regularizer': 'l2',
                      'name': 'enc2'}
    x_enc2 = _encoding_layer(x_enc1, n_layers_conv=2, layer_params=layer_params_2)
    
    # Tercera capa de encoding
    layer_params_3 = {'filters': input_shape[1] * 8, 'kernel_size': 50, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 'kernel_regularizer': 'l2',
                      'name': 'enc3'}
    x_enc3 = _encoding_layer(x_enc2, n_layers_conv=3, layer_params=layer_params_3)
    
    # Cuarta capa de encoding
    layer_params_4 = {'filters': input_shape[1] * 16, 'kernel_size': 50, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 'kernel_regularizer': 'l2',
                      'name': 'enc4'}
    x_enc4 = _encoding_layer(x_enc3, n_layers_conv=3, layer_params=layer_params_4)
    
    
    ### Decoding ###
    
    # Cuarta capa de salida del decoding
    layer_params_4 = {'filters': input_shape[1] * 16, 'kernel_size': 50, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 'kernel_regularizer': 'l2',
                      'name': 'dec4'}
    x_dec4 = _decoding_layer(x_enc4, n_layers_conv=3, layer_params=layer_params_4)
    
    # Tercera capa de salida del decoding
    layer_params_3 = {'filters': input_shape[1] * 8, 'kernel_size': 50, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 'kernel_regularizer': 'l2',
                      'name': 'dec3'}
    x_dec3 = _decoding_layer(x_dec4, n_layers_conv=3, layer_params=layer_params_3)
    
    # Segunda capa de salida del decoding
    layer_params_2 = {'filters': input_shape[1] * 4, 'kernel_size': 50, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 'kernel_regularizer': 'l2',
                      'name': 'dec2'}
    x_dec2 = _decoding_layer(x_dec3, n_layers_conv=2, layer_params=layer_params_2)
    
    # Primera capa de salida del decoding
    layer_params_1 = {'filters': input_shape[1] * 2, 'kernel_size': 50, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 'kernel_regularizer': 'l2',
                      'name': 'dec1'}
    x_dec1 = _decoding_layer(x_dec2, n_layers_conv=2, layer_params=layer_params_1)
                                       
    
    # Aplicando reshape
    # x_reshaped = tf.keras.layers.Reshape((input_shape[0], input_shape[1] * 2))(x_dec1)
    
    # Definición de la capa de salida
    x_out = tf.keras.layers.Dense(3, activation='softmax', kernel_initializer='he_normal',
                                  name='softmax_out')(x_dec1)
    
    # Definición del modelo
    model = tf.keras.Model(inputs=x_in, outputs=x_out, name=name)
    
    return model


def model_6_3(input_shape, padding_value, name=None):
    '''CNN basada en arquitectura encoder-decoder basada en SegNet.
    En este caso se hace la misma red que 6_1, pero aplicando una
    regularización L2 con lambda = 0.001.
    
    References
    ----------
    [1] Badrinarayanan, V., Kendall, A., & Cipolla, R. (2017). 
        Segnet: A deep convolutional encoder-decoder architecture for 
        image segmentation. IEEE transactions on pattern analysis and 
        machine intelligence, 39(12), 2481-2495.
    [2] Ye, J. C., & Sung, W. K. (2019). Understanding geometry of 
        encoder-decoder CNNs. arXiv preprint arXiv:1901.07647.
    '''
    def _conv_bn_act_layer(input_layer, filters, kernel_size, padding,
                          kernel_initializer, kernel_regularizer, name):
        '''Función auxiliar que modela las capas azules conv + batchnorm +
        Activation ReLU para realizar el ENCODING.'''
        # Aplicando la concatenación de capas
        x_conv = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, 
                                        kernel_initializer=kernel_initializer,
                                        kernel_regularizer=kernel_regularizer,
                                        padding=padding, 
                                        name=f'Conv_{name}')(input_layer)
        x_conv = \
            tf.keras.layers.BatchNormalization(name=f'BatchNorm_{name}')(x_conv)
        x_conv = \
            tf.keras.layers.Activation('relu', name=f'Activation_{name}')(x_conv)

        return x_conv
    
    
    def _encoding_layer(input_layer, n_layers_conv, layer_params):
        '''Función auxiliar que permite modelar "n_layers_conv" capas CNN seguida de 
        una capa de Maxpooling, tal como se puede ver en la figura 2 de [1].  
        '''
        # Definición de la salida de este bloque
        x_enc = input_layer
        
        # Aplicando "n_layers_conv" capas convolucionales de codificación
        for i in range(n_layers_conv):
            x_enc = _conv_bn_act_layer(x_enc, filters=layer_params['filters'], 
                                       kernel_size=layer_params['kernel_size'], 
                                       padding=layer_params['padding'],
                                       kernel_initializer=layer_params['kernel_initializer'],
                                       kernel_regularizer=layer_params['kernel_regularizer'], 
                                       name=f"{layer_params['name']}_{i}")

        # Finalmente la capa de MaxPooling
        x_enc = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, 
                                             padding='valid',
                                             name=f"MaxPool_Conv_{layer_params['name']}")(x_enc)
        return x_enc
    
    
    def _decoding_layer(input_layer, n_layers_conv, layer_params):
        '''Función auxiliar que permite modelar una capa de upsampling seguido de 
        "n_layers_conv" capas CNN, tal como se puede ver en la figura 2 de [1].  
        '''
        # Capa de upsampling
        x_dec = tf.keras.layers.UpSampling1D(size=2, name=f"Upsampling_"\
                                                          f"{layer_params['name']}")(input_layer)
        
        # Aplicando "n_layers_conv" capas convolucionales de decodificación
        for i in range(n_layers_conv):
            x_dec = _conv_bn_act_layer(x_dec, filters=layer_params['filters'], 
                                       kernel_size=layer_params['kernel_size'], 
                                       padding=layer_params['padding'],
                                       kernel_initializer=layer_params['kernel_initializer'],
                                       kernel_regularizer=layer_params['kernel_regularizer'], 
                                       name=f"{layer_params['name']}_{i}")

        return x_dec
    
    
    # Definición de la entrada
    x_in = tf.keras.Input(shape=input_shape, dtype='float32')

    # Definición de la capa de máscara
    x_masked = tf.keras.layers.Masking(mask_value=padding_value)(x_in)

    ############        Definición de las capas convolucionales        ############
    
    ### Encoding ###
    
    # Primera capa de encoding
    layer_params_1 = {'filters': input_shape[1] * 2, 'kernel_size': 50, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 
                      'kernel_regularizer': tf.keras.regularizers.l2(l2=0.001),
                      'name': 'enc1'}
    x_enc1 = _encoding_layer(x_masked, n_layers_conv=2, layer_params=layer_params_1)
    
    # Segunda capa de encoding
    layer_params_2 = {'filters': input_shape[1] * 4, 'kernel_size': 50, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 
                      'kernel_regularizer': tf.keras.regularizers.l2(l2=0.001),
                      'name': 'enc2'}
    x_enc2 = _encoding_layer(x_enc1, n_layers_conv=2, layer_params=layer_params_2)
    
    # Tercera capa de encoding
    layer_params_3 = {'filters': input_shape[1] * 8, 'kernel_size': 50, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 
                      'kernel_regularizer': tf.keras.regularizers.l2(l2=0.001),
                      'name': 'enc3'}
    x_enc3 = _encoding_layer(x_enc2, n_layers_conv=3, layer_params=layer_params_3)
    
    # Cuarta capa de encoding
    layer_params_4 = {'filters': input_shape[1] * 16, 'kernel_size': 50, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 
                      'kernel_regularizer': tf.keras.regularizers.l2(l2=0.001),
                      'name': 'enc4'}
    x_enc4 = _encoding_layer(x_enc3, n_layers_conv=3, layer_params=layer_params_4)
    
    
    ### Decoding ###
    
    # Cuarta capa de salida del decoding
    layer_params_4 = {'filters': input_shape[1] * 16, 'kernel_size': 50, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 
                      'kernel_regularizer': tf.keras.regularizers.l2(l2=0.001),
                      'name': 'dec4'}
    x_dec4 = _decoding_layer(x_enc4, n_layers_conv=3, layer_params=layer_params_4)
    
    # Tercera capa de salida del decoding
    layer_params_3 = {'filters': input_shape[1] * 8, 'kernel_size': 50, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 
                      'kernel_regularizer': tf.keras.regularizers.l2(l2=0.001),
                      'name': 'dec3'}
    x_dec3 = _decoding_layer(x_dec4, n_layers_conv=3, layer_params=layer_params_3)
    
    # Segunda capa de salida del decoding
    layer_params_2 = {'filters': input_shape[1] * 4, 'kernel_size': 50, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 
                      'kernel_regularizer': tf.keras.regularizers.l2(l2=0.001),
                      'name': 'dec2'}
    x_dec2 = _decoding_layer(x_dec3, n_layers_conv=2, layer_params=layer_params_2)
    
    # Primera capa de salida del decoding
    layer_params_1 = {'filters': input_shape[1] * 2, 'kernel_size': 50, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 
                      'kernel_regularizer': tf.keras.regularizers.l2(l2=0.001),
                      'name': 'dec1'}
    x_dec1 = _decoding_layer(x_dec2, n_layers_conv=2, layer_params=layer_params_1)
                                       
    
    # Aplicando reshape
    # x_reshaped = tf.keras.layers.Reshape((input_shape[0], input_shape[1] * 2))(x_dec1)
    
    # Definición de la capa de salida
    x_out = tf.keras.layers.Dense(3, activation='softmax', kernel_initializer='he_normal',
                                  name='softmax_out')(x_dec1)
    
    # Definición del modelo
    model = tf.keras.Model(inputs=x_in, outputs=x_out, name=name)
    
    return model


def model_6_4(input_shape, padding_value, name=None):
    '''CNN basada en arquitectura encoder-decoder basada en SegNet.
    En este caso se hace la misma red que 6_3, pero con una capa
    adicional de encoding/decoding.
    
    References
    ----------
    [1] Badrinarayanan, V., Kendall, A., & Cipolla, R. (2017). 
        Segnet: A deep convolutional encoder-decoder architecture for 
        image segmentation. IEEE transactions on pattern analysis and 
        machine intelligence, 39(12), 2481-2495.
    [2] Ye, J. C., & Sung, W. K. (2019). Understanding geometry of 
        encoder-decoder CNNs. arXiv preprint arXiv:1901.07647.
    '''
    def _conv_bn_act_layer(input_layer, filters, kernel_size, padding,
                          kernel_initializer, kernel_regularizer, name):
        '''Función auxiliar que modela las capas azules conv + batchnorm +
        Activation ReLU para realizar el ENCODING.'''
        # Aplicando la concatenación de capas
        x_conv = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, 
                                        kernel_initializer=kernel_initializer,
                                        kernel_regularizer=kernel_regularizer,
                                        padding=padding, 
                                        name=f'Conv_{name}')(input_layer)
        x_conv = \
            tf.keras.layers.BatchNormalization(name=f'BatchNorm_{name}')(x_conv)
        x_conv = \
            tf.keras.layers.Activation('relu', name=f'Activation_{name}')(x_conv)

        return x_conv
    
    
    def _encoding_layer(input_layer, n_layers_conv, layer_params):
        '''Función auxiliar que permite modelar "n_layers_conv" capas CNN seguida de 
        una capa de Maxpooling, tal como se puede ver en la figura 2 de [1].  
        '''
        # Definición de la salida de este bloque
        x_enc = input_layer
        
        # Aplicando "n_layers_conv" capas convolucionales de codificación
        for i in range(n_layers_conv):
            x_enc = _conv_bn_act_layer(x_enc, filters=layer_params['filters'], 
                                       kernel_size=layer_params['kernel_size'], 
                                       padding=layer_params['padding'],
                                       kernel_initializer=layer_params['kernel_initializer'],
                                       kernel_regularizer=layer_params['kernel_regularizer'], 
                                       name=f"{layer_params['name']}_{i}")

        # Finalmente la capa de MaxPooling
        x_enc = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, 
                                             padding='valid',
                                             name=f"MaxPool_Conv_{layer_params['name']}")(x_enc)
        return x_enc
    
    
    def _decoding_layer(input_layer, n_layers_conv, layer_params):
        '''Función auxiliar que permite modelar una capa de upsampling seguido de 
        "n_layers_conv" capas CNN, tal como se puede ver en la figura 2 de [1].  
        '''
        # Capa de upsampling
        x_dec = tf.keras.layers.UpSampling1D(size=2, name=f"Upsampling_"\
                                                          f"{layer_params['name']}")(input_layer)
        
        # Aplicando "n_layers_conv" capas convolucionales de decodificación
        for i in range(n_layers_conv):
            x_dec = _conv_bn_act_layer(x_dec, filters=layer_params['filters'], 
                                       kernel_size=layer_params['kernel_size'], 
                                       padding=layer_params['padding'],
                                       kernel_initializer=layer_params['kernel_initializer'],
                                       kernel_regularizer=layer_params['kernel_regularizer'], 
                                       name=f"{layer_params['name']}_{i}")

        return x_dec
    
    
    # Definición de la entrada
    x_in = tf.keras.Input(shape=input_shape, dtype='float32')

    # Definición de la capa de máscara
    x_masked = tf.keras.layers.Masking(mask_value=padding_value)(x_in)

    ############        Definición de las capas convolucionales        ############
    
    ### Encoding ###
    
    # Primera capa de encoding
    layer_params_1 = {'filters': input_shape[1] * 2, 'kernel_size': 50, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 
                      'kernel_regularizer': tf.keras.regularizers.l2(l2=0.001),
                      'name': 'enc1'}
    x_enc1 = _encoding_layer(x_masked, n_layers_conv=2, layer_params=layer_params_1)
    
    # Segunda capa de encoding
    layer_params_2 = {'filters': input_shape[1] * 4, 'kernel_size': 50, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 
                      'kernel_regularizer': tf.keras.regularizers.l2(l2=0.001),
                      'name': 'enc2'}
    x_enc2 = _encoding_layer(x_enc1, n_layers_conv=2, layer_params=layer_params_2)
    
    # Tercera capa de encoding
    layer_params_3 = {'filters': input_shape[1] * 8, 'kernel_size': 50, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 
                      'kernel_regularizer': tf.keras.regularizers.l2(l2=0.001),
                      'name': 'enc3'}
    x_enc3 = _encoding_layer(x_enc2, n_layers_conv=3, layer_params=layer_params_3)
    
    # Cuarta capa de encoding
    layer_params_4 = {'filters': input_shape[1] * 16, 'kernel_size': 50, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 
                      'kernel_regularizer': tf.keras.regularizers.l2(l2=0.001),
                      'name': 'enc4'}
    x_enc4 = _encoding_layer(x_enc3, n_layers_conv=3, layer_params=layer_params_4)
    
    # Quinta capa de encoding
    layer_params_5 = {'filters': input_shape[1] * 32, 'kernel_size': 50, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 
                      'kernel_regularizer': tf.keras.regularizers.l2(l2=0.001),
                      'name': 'enc5'}
    x_enc5 = _encoding_layer(x_enc4, n_layers_conv=4, layer_params=layer_params_5)
    
    
    ### Decoding ###
    
    # Quinta capa de salida del decoding
    layer_params_5 = {'filters': input_shape[1] * 32, 'kernel_size': 50, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 
                      'kernel_regularizer': tf.keras.regularizers.l2(l2=0.001),
                      'name': 'dec5'}
    x_dec5 = _decoding_layer(x_enc5, n_layers_conv=3, layer_params=layer_params_5)
    
    # Cuarta capa de salida del decoding
    layer_params_4 = {'filters': input_shape[1] * 16, 'kernel_size': 50, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 
                      'kernel_regularizer': tf.keras.regularizers.l2(l2=0.001),
                      'name': 'dec4'}
    x_dec4 = _decoding_layer(x_dec5, n_layers_conv=3, layer_params=layer_params_4)
    
    # Tercera capa de salida del decoding
    layer_params_3 = {'filters': input_shape[1] * 8, 'kernel_size': 50, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 
                      'kernel_regularizer': tf.keras.regularizers.l2(l2=0.001),
                      'name': 'dec3'}
    x_dec3 = _decoding_layer(x_dec4, n_layers_conv=3, layer_params=layer_params_3)
    
    # Segunda capa de salida del decoding
    layer_params_2 = {'filters': input_shape[1] * 4, 'kernel_size': 50, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 
                      'kernel_regularizer': tf.keras.regularizers.l2(l2=0.001),
                      'name': 'dec2'}
    x_dec2 = _decoding_layer(x_dec3, n_layers_conv=2, layer_params=layer_params_2)
    
    # Primera capa de salida del decoding
    layer_params_1 = {'filters': input_shape[1] * 2, 'kernel_size': 50, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 
                      'kernel_regularizer': tf.keras.regularizers.l2(l2=0.001),
                      'name': 'dec1'}
    x_dec1 = _decoding_layer(x_dec2, n_layers_conv=2, layer_params=layer_params_1)
                                       
    
    # Aplicando reshape
    # x_reshaped = tf.keras.layers.Reshape((input_shape[0], input_shape[1] * 2))(x_dec1)
    
    # Definición de la capa de salida
    x_out = tf.keras.layers.Dense(3, activation='softmax', kernel_initializer='he_normal',
                                  name='softmax_out')(x_dec1)
    
    # Definición del modelo
    model = tf.keras.Model(inputs=x_in, outputs=x_out, name=name)
    
    return model


def model_7_1(input_shape, padding_value, name=None):
    '''CNN basada en arquitectura encoder-decoder basada en SegNet.
    Consiste en la misma implementación que el modelo 6_1, pero aplicando
    una SegNet a cada una de las envolventes de forma independiente.
    
    References
    ----------
    [1] Badrinarayanan, V., Kendall, A., & Cipolla, R. (2017). 
        Segnet: A deep convolutional encoder-decoder architecture for 
        image segmentation. IEEE transactions on pattern analysis and 
        machine intelligence, 39(12), 2481-2495.
    [2] Ye, J. C., & Sung, W. K. (2019). Understanding geometry of 
        encoder-decoder CNNs. arXiv preprint arXiv:1901.07647.
    '''
    def _conv_bn_act_layer(input_layer, filters, kernel_size, padding,
                          kernel_initializer, name):
        '''Función auxiliar que modela las capas azules conv + batchnorm +
        Activation ReLU para realizar el ENCODING.'''
        # Aplicando la concatenación de capas
        x_conv = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, 
                                        kernel_initializer=kernel_initializer,
                                        padding=padding, 
                                        name=f'Conv_{name}')(input_layer)
        x_conv = \
            tf.keras.layers.BatchNormalization(name=f'BatchNorm_{name}')(x_conv)
        x_conv = \
            tf.keras.layers.Activation('relu', name=f'Activation_{name}')(x_conv)

        return x_conv
    
    
    def _encoding_layer(input_layer, n_layers_conv, layer_params):
        '''Función auxiliar que permite modelar "n_layers_conv" capas CNN seguida de 
        una capa de Maxpooling, tal como se puede ver en la figura 2 de [1].  
        '''
        # Definición de la salida de este bloque
        x_enc = input_layer
        
        # Aplicando "n_layers_conv" capas convolucionales de codificación
        for i in range(n_layers_conv):
            x_enc = _conv_bn_act_layer(x_enc, filters=layer_params['filters'], 
                                       kernel_size=layer_params['kernel_size'], 
                                       padding=layer_params['padding'],
                                       kernel_initializer=layer_params['kernel_initializer'], 
                                       name=f"{layer_params['name']}_{i}")

        # Finalmente la capa de MaxPooling
        x_enc = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, 
                                             padding='valid',
                                             name=f"MaxPool_Conv_{layer_params['name']}")(x_enc)
        return x_enc
    
    
    def _decoding_layer(input_layer, n_layers_conv, layer_params):
        '''Función auxiliar que permite modelar una capa de upsampling seguido de 
        "n_layers_conv" capas CNN, tal como se puede ver en la figura 2 de [1].  
        '''
        # Capa de upsampling
        x_dec = tf.keras.layers.UpSampling1D(size=2, name=f"Upsampling_"\
                                                          f"{layer_params['name']}")(input_layer)
        
        # Aplicando "n_layers_conv" capas convolucionales de decodificación
        for i in range(n_layers_conv):
            x_dec = _conv_bn_act_layer(x_dec, filters=layer_params['filters'], 
                                       kernel_size=layer_params['kernel_size'], 
                                       padding=layer_params['padding'],
                                       kernel_initializer=layer_params['kernel_initializer'], 
                                       name=f"{layer_params['name']}_{i}")

        return x_dec
    
    # Definición de una lista auxiliar de entradas y salidas
    x_in_list = list()
    list_decs = list()
    
    for i in range(input_shape[1]):
        # Definición de la entrada
        x_in = tf.keras.Input(shape=(input_shape[0], 1), dtype='float32',
                              name=f'Input_ch{i+1}')

        # Agregando a la lista
        x_in_list.append(x_in)
        
        # Definición de la capa de máscara
        x_masked = tf.keras.layers.Masking(mask_value=padding_value, name=f'Masking_ch{i+1}')(x_in)

        ############        Definición de las capas convolucionales        ############
        
        ### Encoding ###
        
        # Primera capa de encoding
        layer_params_1 = {'filters': 2, 'kernel_size': 50, 'padding': 'same',
                        'kernel_initializer': 'he_normal', 'name': f'enc1_ch{i+1}'}
        x_enc1 = _encoding_layer(x_masked, n_layers_conv=2, layer_params=layer_params_1)
        
        # Segunda capa de encoding
        layer_params_2 = {'filters': 4, 'kernel_size': 50, 'padding': 'same',
                        'kernel_initializer': 'he_normal', 'name': f'enc2_ch{i+1}'}
        x_enc2 = _encoding_layer(x_enc1, n_layers_conv=2, layer_params=layer_params_2)
        
        # Tercera capa de encoding
        layer_params_3 = {'filters': 8, 'kernel_size': 50, 'padding': 'same',
                        'kernel_initializer': 'he_normal', 'name': f'enc3_ch{i+1}'}
        x_enc3 = _encoding_layer(x_enc2, n_layers_conv=3, layer_params=layer_params_3)
        
        # Cuarta capa de encoding
        layer_params_4 = {'filters': 16, 'kernel_size': 50, 'padding': 'same',
                        'kernel_initializer': 'he_normal', 'name': f'enc4_ch{i+1}'}
        x_enc4 = _encoding_layer(x_enc3, n_layers_conv=3, layer_params=layer_params_4)
        
        
        ### Decoding ###
        
        # Cuarta capa de salida del decoding
        layer_params_4 = {'filters': 16, 'kernel_size': 50, 'padding': 'same',
                        'kernel_initializer': 'he_normal', 'name': f'dec4_ch{i+1}'}
        x_dec4 = _decoding_layer(x_enc4, n_layers_conv=3, layer_params=layer_params_4)
        
        # Tercera capa de salida del decoding
        layer_params_3 = {'filters': 8, 'kernel_size': 50, 'padding': 'same',
                        'kernel_initializer': 'he_normal', 'name': f'dec3_ch{i+1}'}
        x_dec3 = _decoding_layer(x_dec4, n_layers_conv=3, layer_params=layer_params_3)
        
        # Segunda capa de salida del decoding
        layer_params_2 = {'filters': 4, 'kernel_size': 50, 'padding': 'same',
                        'kernel_initializer': 'he_normal', 'name': f'dec2_ch{i+1}'}
        x_dec2 = _decoding_layer(x_dec3, n_layers_conv=2, layer_params=layer_params_2)
        
        # Primera capa de salida del decoding
        layer_params_1 = {'filters': 2, 'kernel_size': 50, 'padding': 'same',
                        'kernel_initializer': 'he_normal', 'name': f'dec1_ch{i+1}'}
        x_dec1 = _decoding_layer(x_dec2, n_layers_conv=2, layer_params=layer_params_1)
                                        
            
        # Definición de la capa de salida
        # x_out = tf.keras.layers.Dense(3, activation='softmax', kernel_initializer='he_normal',
        #                             name=f'softmax_out_ch{i}')(x_dec1)
        
        # Agregando a la lista de salida
        list_decs.append(x_dec1)
    
    # Concatenando la lista de envolventes para generar una matriz
    x_conc = tf.keras.layers.concatenate(list_decs, axis=-1, name='Concatenate_layer')
    
    # Finalmente, definiendo la capa de salida
    x_out = tf.keras.layers.Dense(3, activation='softmax', kernel_initializer='he_normal',
                                  name=f'Heart_sound')(x_conc)
    
    # Definición del modelo
    model = tf.keras.Model(inputs=x_in_list, outputs=x_out, name=name)
    
    return model


def model_7_1_2(input_shape, padding_value, name=None):
    '''CNN basada en arquitectura encoder-decoder basada en SegNet.
    Consiste en la misma implementación que el modelo 7_1, pero con una sola
    etiqueta.
    
    References
    ----------
    [1] Badrinarayanan, V., Kendall, A., & Cipolla, R. (2017). 
        Segnet: A deep convolutional encoder-decoder architecture for 
        image segmentation. IEEE transactions on pattern analysis and 
        machine intelligence, 39(12), 2481-2495.
    [2] Ye, J. C., & Sung, W. K. (2019). Understanding geometry of 
        encoder-decoder CNNs. arXiv preprint arXiv:1901.07647.
    '''
    def _conv_bn_act_layer(input_layer, filters, kernel_size, padding,
                          kernel_initializer, name):
        '''Función auxiliar que modela las capas azules conv + batchnorm +
        Activation ReLU para realizar el ENCODING.'''
        # Aplicando la concatenación de capas
        x_conv = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, 
                                        kernel_initializer=kernel_initializer,
                                        padding=padding, 
                                        name=f'Conv_{name}')(input_layer)
        x_conv = \
            tf.keras.layers.BatchNormalization(name=f'BatchNorm_{name}')(x_conv)
        x_conv = \
            tf.keras.layers.Activation('relu', name=f'Activation_{name}')(x_conv)

        return x_conv
    
    
    def _encoding_layer(input_layer, n_layers_conv, layer_params):
        '''Función auxiliar que permite modelar "n_layers_conv" capas CNN seguida de 
        una capa de Maxpooling, tal como se puede ver en la figura 2 de [1].  
        '''
        # Definición de la salida de este bloque
        x_enc = input_layer
        
        # Aplicando "n_layers_conv" capas convolucionales de codificación
        for i in range(n_layers_conv):
            x_enc = _conv_bn_act_layer(x_enc, filters=layer_params['filters'], 
                                       kernel_size=layer_params['kernel_size'], 
                                       padding=layer_params['padding'],
                                       kernel_initializer=layer_params['kernel_initializer'], 
                                       name=f"{layer_params['name']}_{i}")

        # Finalmente la capa de MaxPooling
        x_enc = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, 
                                             padding='valid',
                                             name=f"MaxPool_Conv_{layer_params['name']}")(x_enc)
        return x_enc
    
    
    def _decoding_layer(input_layer, n_layers_conv, layer_params):
        '''Función auxiliar que permite modelar una capa de upsampling seguido de 
        "n_layers_conv" capas CNN, tal como se puede ver en la figura 2 de [1].  
        '''
        # Capa de upsampling
        x_dec = tf.keras.layers.UpSampling1D(size=2, name=f"Upsampling_"\
                                                          f"{layer_params['name']}")(input_layer)
        
        # Aplicando "n_layers_conv" capas convolucionales de decodificación
        for i in range(n_layers_conv):
            x_dec = _conv_bn_act_layer(x_dec, filters=layer_params['filters'], 
                                       kernel_size=layer_params['kernel_size'], 
                                       padding=layer_params['padding'],
                                       kernel_initializer=layer_params['kernel_initializer'], 
                                       name=f"{layer_params['name']}_{i}")

        return x_dec
    
    # Definición de una lista auxiliar de entradas y salidas
    x_in_list = list()
    list_decs = list()
    
    for i in range(input_shape[1]):
        # Definición de la entrada
        x_in = tf.keras.Input(shape=(input_shape[0], 1), dtype='float32',
                              name=f'Input_ch{i+1}')

        # Agregando a la lista
        x_in_list.append(x_in)
        
        # Definición de la capa de máscara
        x_masked = tf.keras.layers.Masking(mask_value=padding_value, name=f'Masking_ch{i+1}')(x_in)

        ############        Definición de las capas convolucionales        ############
        
        ### Encoding ###
        
        # Primera capa de encoding
        layer_params_1 = {'filters': 2, 'kernel_size': 50, 'padding': 'same',
                        'kernel_initializer': 'he_normal', 'name': f'enc1_ch{i+1}'}
        x_enc1 = _encoding_layer(x_masked, n_layers_conv=2, layer_params=layer_params_1)
        
        # Segunda capa de encoding
        layer_params_2 = {'filters': 4, 'kernel_size': 50, 'padding': 'same',
                        'kernel_initializer': 'he_normal', 'name': f'enc2_ch{i+1}'}
        x_enc2 = _encoding_layer(x_enc1, n_layers_conv=2, layer_params=layer_params_2)
        
        # Tercera capa de encoding
        layer_params_3 = {'filters': 8, 'kernel_size': 50, 'padding': 'same',
                        'kernel_initializer': 'he_normal', 'name': f'enc3_ch{i+1}'}
        x_enc3 = _encoding_layer(x_enc2, n_layers_conv=3, layer_params=layer_params_3)
        
        # Cuarta capa de encoding
        layer_params_4 = {'filters': 16, 'kernel_size': 50, 'padding': 'same',
                        'kernel_initializer': 'he_normal', 'name': f'enc4_ch{i+1}'}
        x_enc4 = _encoding_layer(x_enc3, n_layers_conv=3, layer_params=layer_params_4)
        
        
        ### Decoding ###
        
        # Cuarta capa de salida del decoding
        layer_params_4 = {'filters': 16, 'kernel_size': 50, 'padding': 'same',
                        'kernel_initializer': 'he_normal', 'name': f'dec4_ch{i+1}'}
        x_dec4 = _decoding_layer(x_enc4, n_layers_conv=3, layer_params=layer_params_4)
        
        # Tercera capa de salida del decoding
        layer_params_3 = {'filters': 8, 'kernel_size': 50, 'padding': 'same',
                        'kernel_initializer': 'he_normal', 'name': f'dec3_ch{i+1}'}
        x_dec3 = _decoding_layer(x_dec4, n_layers_conv=3, layer_params=layer_params_3)
        
        # Segunda capa de salida del decoding
        layer_params_2 = {'filters': 4, 'kernel_size': 50, 'padding': 'same',
                        'kernel_initializer': 'he_normal', 'name': f'dec2_ch{i+1}'}
        x_dec2 = _decoding_layer(x_dec3, n_layers_conv=2, layer_params=layer_params_2)
        
        # Primera capa de salida del decoding
        layer_params_1 = {'filters': 2, 'kernel_size': 50, 'padding': 'same',
                        'kernel_initializer': 'he_normal', 'name': f'dec1_ch{i+1}'}
        x_dec1 = _decoding_layer(x_dec2, n_layers_conv=2, layer_params=layer_params_1)
                                        
            
        # Definición de la capa de salida
        # x_out = tf.keras.layers.Dense(3, activation='softmax', kernel_initializer='he_normal',
        #                             name=f'softmax_out_ch{i}')(x_dec1)
        
        # Agregando a la lista de salida
        list_decs.append(x_dec1)
    
    # Concatenando la lista de envolventes para generar una matriz
    x_conc = tf.keras.layers.concatenate(list_decs, axis=-1, name='Concatenate_layer')
    
    # Finalmente, definiendo la capa de salida
    x_out = tf.keras.layers.Dense(2, activation='softmax', kernel_initializer='he_normal',
                                  name=f'Heart_sound')(x_conc)
    
    # Definición del modelo
    model = tf.keras.Model(inputs=x_in_list, outputs=x_out, name=name)
    
    return model


def model_7_2(input_shape, padding_value, name=None):
    '''CNN basada en arquitectura encoder-decoder basada en SegNet.
    Consiste en la misma implementación que el modelo 6_1, pero aplicando
    una SegNet a cada una de las envolventes de forma independiente.
    En esta aproximación, a diferencia del model_7_1, se utiliza una 
    etapa softmax por cada canal.
    
    References
    ----------
    [1] Badrinarayanan, V., Kendall, A., & Cipolla, R. (2017). 
        Segnet: A deep convolutional encoder-decoder architecture for 
        image segmentation. IEEE transactions on pattern analysis and 
        machine intelligence, 39(12), 2481-2495.
    [2] Ye, J. C., & Sung, W. K. (2019). Understanding geometry of 
        encoder-decoder CNNs. arXiv preprint arXiv:1901.07647.
    '''
    def _conv_bn_act_layer(input_layer, filters, kernel_size, padding,
                          kernel_initializer, name):
        '''Función auxiliar que modela las capas azules conv + batchnorm +
        Activation ReLU para realizar el ENCODING.'''
        # Aplicando la concatenación de capas
        x_conv = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, 
                                        kernel_initializer=kernel_initializer,
                                        padding=padding, 
                                        name=f'Conv_{name}')(input_layer)
        x_conv = \
            tf.keras.layers.BatchNormalization(name=f'BatchNorm_{name}')(x_conv)
        x_conv = \
            tf.keras.layers.Activation('relu', name=f'Activation_{name}')(x_conv)

        return x_conv
    
    
    def _encoding_layer(input_layer, n_layers_conv, layer_params):
        '''Función auxiliar que permite modelar "n_layers_conv" capas CNN seguida de 
        una capa de Maxpooling, tal como se puede ver en la figura 2 de [1].  
        '''
        # Definición de la salida de este bloque
        x_enc = input_layer
        
        # Aplicando "n_layers_conv" capas convolucionales de codificación
        for i in range(n_layers_conv):
            x_enc = _conv_bn_act_layer(x_enc, filters=layer_params['filters'], 
                                       kernel_size=layer_params['kernel_size'], 
                                       padding=layer_params['padding'],
                                       kernel_initializer=layer_params['kernel_initializer'], 
                                       name=f"{layer_params['name']}_{i}")

        # Finalmente la capa de MaxPooling
        x_enc = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, 
                                             padding='valid',
                                             name=f"MaxPool_Conv_{layer_params['name']}")(x_enc)
        return x_enc
    
    
    def _decoding_layer(input_layer, n_layers_conv, layer_params):
        '''Función auxiliar que permite modelar una capa de upsampling seguido de 
        "n_layers_conv" capas CNN, tal como se puede ver en la figura 2 de [1].  
        '''
        # Capa de upsampling
        x_dec = tf.keras.layers.UpSampling1D(size=2, name=f"Upsampling_"\
                                                          f"{layer_params['name']}")(input_layer)
        
        # Aplicando "n_layers_conv" capas convolucionales de decodificación
        for i in range(n_layers_conv):
            x_dec = _conv_bn_act_layer(x_dec, filters=layer_params['filters'], 
                                       kernel_size=layer_params['kernel_size'], 
                                       padding=layer_params['padding'],
                                       kernel_initializer=layer_params['kernel_initializer'], 
                                       name=f"{layer_params['name']}_{i}")

        return x_dec
    
    # Definición de una lista auxiliar de entradas y salidas
    x_in_list = list()
    list_decs = list()
    
    for i in range(input_shape[1]):
        # Definición de la entrada
        x_in = tf.keras.Input(shape=(input_shape[0], 1), dtype='float32',
                              name=f'Input_ch{i+1}')

        # Agregando a la lista
        x_in_list.append(x_in)
        
        # Definición de la capa de máscara
        x_masked = tf.keras.layers.Masking(mask_value=padding_value, name=f'Masking_ch{i+1}')(x_in)

        ############        Definición de las capas convolucionales        ############
        
        ### Encoding ###
        
        # Primera capa de encoding
        layer_params_1 = {'filters': 2, 'kernel_size': 50, 'padding': 'same',
                        'kernel_initializer': 'he_normal', 'name': f'enc1_ch{i+1}'}
        x_enc1 = _encoding_layer(x_masked, n_layers_conv=2, layer_params=layer_params_1)
        
        # Segunda capa de encoding
        layer_params_2 = {'filters': 4, 'kernel_size': 50, 'padding': 'same',
                        'kernel_initializer': 'he_normal', 'name': f'enc2_ch{i+1}'}
        x_enc2 = _encoding_layer(x_enc1, n_layers_conv=2, layer_params=layer_params_2)
        
        # Tercera capa de encoding
        layer_params_3 = {'filters': 8, 'kernel_size': 50, 'padding': 'same',
                        'kernel_initializer': 'he_normal', 'name': f'enc3_ch{i+1}'}
        x_enc3 = _encoding_layer(x_enc2, n_layers_conv=3, layer_params=layer_params_3)
        
        # Cuarta capa de encoding
        layer_params_4 = {'filters': 16, 'kernel_size': 50, 'padding': 'same',
                        'kernel_initializer': 'he_normal', 'name': f'enc4_ch{i+1}'}
        x_enc4 = _encoding_layer(x_enc3, n_layers_conv=3, layer_params=layer_params_4)
        
        
        ### Decoding ###
        
        # Cuarta capa de salida del decoding
        layer_params_4 = {'filters': 16, 'kernel_size': 50, 'padding': 'same',
                        'kernel_initializer': 'he_normal', 'name': f'dec4_ch{i+1}'}
        x_dec4 = _decoding_layer(x_enc4, n_layers_conv=3, layer_params=layer_params_4)
        
        # Tercera capa de salida del decoding
        layer_params_3 = {'filters': 8, 'kernel_size': 50, 'padding': 'same',
                        'kernel_initializer': 'he_normal', 'name': f'dec3_ch{i+1}'}
        x_dec3 = _decoding_layer(x_dec4, n_layers_conv=3, layer_params=layer_params_3)
        
        # Segunda capa de salida del decoding
        layer_params_2 = {'filters': 4, 'kernel_size': 50, 'padding': 'same',
                        'kernel_initializer': 'he_normal', 'name': f'dec2_ch{i+1}'}
        x_dec2 = _decoding_layer(x_dec3, n_layers_conv=2, layer_params=layer_params_2)
        
        # Primera capa de salida del decoding
        layer_params_1 = {'filters': 2, 'kernel_size': 50, 'padding': 'same',
                        'kernel_initializer': 'he_normal', 'name': f'dec1_ch{i+1}'}
        x_dec1 = _decoding_layer(x_dec2, n_layers_conv=2, layer_params=layer_params_1)
                                        
            
        # Definición de la capa de salida
        x_out_chan = tf.keras.layers.Dense(3, activation='softmax', kernel_initializer='he_normal',
                                           name=f'softmax_out_ch{i}')(x_dec1)
        
        # Agregando a la lista de salida
        list_decs.append(x_out_chan)
    
    # Concatenando la lista de envolventes para generar una matriz
    x_conc = tf.keras.layers.concatenate(list_decs, axis=-1, name='Concatenate_layer')
    
    # Finalmente, definiendo la capa de salida
    x_out = tf.keras.layers.Dense(3, activation='softmax', kernel_initializer='he_normal',
                                  name=f'Heart_sound')(x_conc)
    
    # Definición del modelo
    model = tf.keras.Model(inputs=x_in_list, outputs=x_out, name=name)
    
    return model


def model_8_1(input_shape, padding_value, name=None):
    '''CNN basada en arquitectura encoder-decoder basada en SegNet.
    Consiste en la misma red planteada en el model_6_1 pero usando
    una cantidad de filtros fija en cada capa.
    
    References
    ----------
    [1] Badrinarayanan, V., Kendall, A., & Cipolla, R. (2017). 
        Segnet: A deep convolutional encoder-decoder architecture for 
        image segmentation. IEEE transactions on pattern analysis and 
        machine intelligence, 39(12), 2481-2495.
    [2] Ye, J. C., & Sung, W. K. (2019). Understanding geometry of 
        encoder-decoder CNNs. arXiv preprint arXiv:1901.07647.
    '''
    def _conv_bn_act_layer(input_layer, filters, kernel_size, padding,
                          kernel_initializer, name):
        '''Función auxiliar que modela las capas azules conv + batchnorm +
        Activation ReLU para realizar el ENCODING.'''
        # Aplicando la concatenación de capas
        x_conv = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, 
                                        kernel_initializer=kernel_initializer,
                                        padding=padding, 
                                        name=f'Conv_{name}')(input_layer)
        x_conv = \
            tf.keras.layers.BatchNormalization(name=f'BatchNorm_{name}')(x_conv)
        x_conv = \
            tf.keras.layers.Activation('relu', name=f'Activation_{name}')(x_conv)

        return x_conv
    
    
    def _encoding_layer(input_layer, n_layers_conv, layer_params):
        '''Función auxiliar que permite modelar "n_layers_conv" capas CNN seguida de 
        una capa de Maxpooling, tal como se puede ver en la figura 2 de [1].  
        '''
        # Definición de la salida de este bloque
        x_enc = input_layer
        
        # Aplicando "n_layers_conv" capas convolucionales de codificación
        for i in range(n_layers_conv):
            x_enc = _conv_bn_act_layer(x_enc, filters=layer_params['filters'], 
                                       kernel_size=layer_params['kernel_size'], 
                                       padding=layer_params['padding'],
                                       kernel_initializer=layer_params['kernel_initializer'], 
                                       name=f"{layer_params['name']}_{i}")

        # Finalmente la capa de MaxPooling
        x_enc = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, 
                                             padding='valid',
                                             name=f"MaxPool_Conv_{layer_params['name']}")(x_enc)
        return x_enc
    
    
    def _decoding_layer(input_layer, n_layers_conv, layer_params):
        '''Función auxiliar que permite modelar una capa de upsampling seguido de 
        "n_layers_conv" capas CNN, tal como se puede ver en la figura 2 de [1].  
        '''
        # Capa de upsampling
        x_dec = tf.keras.layers.UpSampling1D(size=2, name=f"Upsampling_"\
                                                          f"{layer_params['name']}")(input_layer)
        
        # Aplicando "n_layers_conv" capas convolucionales de decodificación
        for i in range(n_layers_conv):
            x_dec = _conv_bn_act_layer(x_dec, filters=layer_params['filters'], 
                                       kernel_size=layer_params['kernel_size'], 
                                       padding=layer_params['padding'],
                                       kernel_initializer=layer_params['kernel_initializer'], 
                                       name=f"{layer_params['name']}_{i}")

        return x_dec
    
    
    # Definición de la entrada
    x_in = tf.keras.Input(shape=input_shape, dtype='float32')

    # Definición de la capa de máscara
    x_masked = tf.keras.layers.Masking(mask_value=padding_value)(x_in)

    ############        Definición de las capas convolucionales        ############
    
    ### Encoding ###
    
    # Primera capa de encoding
    layer_params_1 = {'filters': 13, 'kernel_size': 50, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 'name': 'enc1'}
    x_enc1 = _encoding_layer(x_masked, n_layers_conv=2, layer_params=layer_params_1)
    
    # Segunda capa de encoding
    layer_params_2 = {'filters': 13, 'kernel_size': 50, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 'name': 'enc2'}
    x_enc2 = _encoding_layer(x_enc1, n_layers_conv=2, layer_params=layer_params_2)
    
    # Tercera capa de encoding
    layer_params_3 = {'filters': 13, 'kernel_size': 50, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 'name': 'enc3'}
    x_enc3 = _encoding_layer(x_enc2, n_layers_conv=3, layer_params=layer_params_3)
    
    # Cuarta capa de encoding
    layer_params_4 = {'filters': 13, 'kernel_size': 50, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 'name': 'enc4'}
    x_enc4 = _encoding_layer(x_enc3, n_layers_conv=3, layer_params=layer_params_4)
    
    
    ### Decoding ###
    
    # Cuarta capa de salida del decoding
    layer_params_4 = {'filters': 13, 'kernel_size': 50, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 'name': 'dec4'}
    x_dec4 = _decoding_layer(x_enc4, n_layers_conv=3, layer_params=layer_params_4)
    
    # Tercera capa de salida del decoding
    layer_params_3 = {'filters': 13, 'kernel_size': 50, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 'name': 'dec3'}
    x_dec3 = _decoding_layer(x_dec4, n_layers_conv=3, layer_params=layer_params_3)
    
    # Segunda capa de salida del decoding
    layer_params_2 = {'filters': 13, 'kernel_size': 50, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 'name': 'dec2'}
    x_dec2 = _decoding_layer(x_dec3, n_layers_conv=2, layer_params=layer_params_2)
    
    # Primera capa de salida del decoding
    layer_params_1 = {'filters': 13, 'kernel_size': 50, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 'name': 'dec1'}
    x_dec1 = _decoding_layer(x_dec2, n_layers_conv=2, layer_params=layer_params_1)
                                       
    
    # Aplicando reshape
    # x_reshaped = tf.keras.layers.Reshape((input_shape[0], input_shape[1] * 2))(x_dec1)
    
    # Definición de la capa de salida
    x_out = tf.keras.layers.Dense(3, activation='softmax', kernel_initializer='he_normal',
                                  name='softmax_out')(x_dec1)
    
    # Definición del modelo
    model = tf.keras.Model(inputs=x_in, outputs=x_out, name=name)
    
    return model


def model_8_2(input_shape, padding_value, name=None):
    '''CNN basada en arquitectura encoder-decoder basada en SegNet.
    Consiste en la misma red planteada en el model_6_1 pero usando
    una cantidad de filtros fija en cada capa, y el largo del filtro
    es de 200.
    
    References
    ----------
    [1] Badrinarayanan, V., Kendall, A., & Cipolla, R. (2017). 
        Segnet: A deep convolutional encoder-decoder architecture for 
        image segmentation. IEEE transactions on pattern analysis and 
        machine intelligence, 39(12), 2481-2495.
    [2] Ye, J. C., & Sung, W. K. (2019). Understanding geometry of 
        encoder-decoder CNNs. arXiv preprint arXiv:1901.07647.
    '''
    def _conv_bn_act_layer(input_layer, filters, kernel_size, padding,
                          kernel_initializer, name):
        '''Función auxiliar que modela las capas azules conv + batchnorm +
        Activation ReLU para realizar el ENCODING.'''
        # Aplicando la concatenación de capas
        x_conv = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, 
                                        kernel_initializer=kernel_initializer,
                                        padding=padding, 
                                        name=f'Conv_{name}')(input_layer)
        x_conv = \
            tf.keras.layers.BatchNormalization(name=f'BatchNorm_{name}')(x_conv)
        x_conv = \
            tf.keras.layers.Activation('relu', name=f'Activation_{name}')(x_conv)

        return x_conv
    
    
    def _encoding_layer(input_layer, n_layers_conv, layer_params):
        '''Función auxiliar que permite modelar "n_layers_conv" capas CNN seguida de 
        una capa de Maxpooling, tal como se puede ver en la figura 2 de [1].  
        '''
        # Definición de la salida de este bloque
        x_enc = input_layer
        
        # Aplicando "n_layers_conv" capas convolucionales de codificación
        for i in range(n_layers_conv):
            x_enc = _conv_bn_act_layer(x_enc, filters=layer_params['filters'], 
                                       kernel_size=layer_params['kernel_size'], 
                                       padding=layer_params['padding'],
                                       kernel_initializer=layer_params['kernel_initializer'], 
                                       name=f"{layer_params['name']}_{i}")

        # Finalmente la capa de MaxPooling
        x_enc = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, 
                                             padding='valid',
                                             name=f"MaxPool_Conv_{layer_params['name']}")(x_enc)
        return x_enc
    
    
    def _decoding_layer(input_layer, n_layers_conv, layer_params):
        '''Función auxiliar que permite modelar una capa de upsampling seguido de 
        "n_layers_conv" capas CNN, tal como se puede ver en la figura 2 de [1].  
        '''
        # Capa de upsampling
        x_dec = tf.keras.layers.UpSampling1D(size=2, name=f"Upsampling_"\
                                                          f"{layer_params['name']}")(input_layer)
        
        # Aplicando "n_layers_conv" capas convolucionales de decodificación
        for i in range(n_layers_conv):
            x_dec = _conv_bn_act_layer(x_dec, filters=layer_params['filters'], 
                                       kernel_size=layer_params['kernel_size'], 
                                       padding=layer_params['padding'],
                                       kernel_initializer=layer_params['kernel_initializer'], 
                                       name=f"{layer_params['name']}_{i}")

        return x_dec
    
    
    # Definición de la entrada
    x_in = tf.keras.Input(shape=input_shape, dtype='float32')

    # Definición de la capa de máscara
    x_masked = tf.keras.layers.Masking(mask_value=padding_value)(x_in)

    ############        Definición de las capas convolucionales        ############
    
    ### Encoding ###
    
    # Primera capa de encoding
    layer_params_1 = {'filters': 13, 'kernel_size': 200, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 'name': 'enc1'}
    x_enc1 = _encoding_layer(x_masked, n_layers_conv=2, layer_params=layer_params_1)
    
    # Segunda capa de encoding
    layer_params_2 = {'filters': 13, 'kernel_size': 200, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 'name': 'enc2'}
    x_enc2 = _encoding_layer(x_enc1, n_layers_conv=2, layer_params=layer_params_2)
    
    # Tercera capa de encoding
    layer_params_3 = {'filters': 13, 'kernel_size': 200, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 'name': 'enc3'}
    x_enc3 = _encoding_layer(x_enc2, n_layers_conv=3, layer_params=layer_params_3)
    
    # Cuarta capa de encoding
    layer_params_4 = {'filters': 13, 'kernel_size': 200, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 'name': 'enc4'}
    x_enc4 = _encoding_layer(x_enc3, n_layers_conv=3, layer_params=layer_params_4)
    
    
    ### Decoding ###
    
    # Cuarta capa de salida del decoding
    layer_params_4 = {'filters': 13, 'kernel_size': 200, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 'name': 'dec4'}
    x_dec4 = _decoding_layer(x_enc4, n_layers_conv=3, layer_params=layer_params_4)
    
    # Tercera capa de salida del decoding
    layer_params_3 = {'filters': 13, 'kernel_size': 200, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 'name': 'dec3'}
    x_dec3 = _decoding_layer(x_dec4, n_layers_conv=3, layer_params=layer_params_3)
    
    # Segunda capa de salida del decoding
    layer_params_2 = {'filters': 13, 'kernel_size': 200, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 'name': 'dec2'}
    x_dec2 = _decoding_layer(x_dec3, n_layers_conv=2, layer_params=layer_params_2)
    
    # Primera capa de salida del decoding
    layer_params_1 = {'filters': 13, 'kernel_size': 200, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 'name': 'dec1'}
    x_dec1 = _decoding_layer(x_dec2, n_layers_conv=2, layer_params=layer_params_1)
                                       
    
    # Aplicando reshape
    # x_reshaped = tf.keras.layers.Reshape((input_shape[0], input_shape[1] * 2))(x_dec1)
    
    # Definición de la capa de salida
    x_out = tf.keras.layers.Dense(3, activation='softmax', kernel_initializer='he_normal',
                                  name='softmax_out')(x_dec1)
    
    # Definición del modelo
    model = tf.keras.Model(inputs=x_in, outputs=x_out, name=name)
    
    return model


def model_8_3(input_shape, padding_value, name=None):
    '''CNN basada en arquitectura encoder-decoder basada en SegNet.
    Consiste en la misma red planteada en el model_6_1 pero usando
    una cantidad de filtros fija en cada capa, y el largo del filtro
    es de 100.
    
    References
    ----------
    [1] Badrinarayanan, V., Kendall, A., & Cipolla, R. (2017). 
        Segnet: A deep convolutional encoder-decoder architecture for 
        image segmentation. IEEE transactions on pattern analysis and 
        machine intelligence, 39(12), 2481-2495.
    [2] Ye, J. C., & Sung, W. K. (2019). Understanding geometry of 
        encoder-decoder CNNs. arXiv preprint arXiv:1901.07647.
    '''
    def _conv_bn_act_layer(input_layer, filters, kernel_size, padding,
                          kernel_initializer, name):
        '''Función auxiliar que modela las capas azules conv + batchnorm +
        Activation ReLU para realizar el ENCODING.'''
        # Aplicando la concatenación de capas
        x_conv = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, 
                                        kernel_initializer=kernel_initializer,
                                        padding=padding, 
                                        name=f'Conv_{name}')(input_layer)
        x_conv = \
            tf.keras.layers.BatchNormalization(name=f'BatchNorm_{name}')(x_conv)
        x_conv = \
            tf.keras.layers.Activation('relu', name=f'Activation_{name}')(x_conv)

        return x_conv
    
    
    def _encoding_layer(input_layer, n_layers_conv, layer_params):
        '''Función auxiliar que permite modelar "n_layers_conv" capas CNN seguida de 
        una capa de Maxpooling, tal como se puede ver en la figura 2 de [1].  
        '''
        # Definición de la salida de este bloque
        x_enc = input_layer
        
        # Aplicando "n_layers_conv" capas convolucionales de codificación
        for i in range(n_layers_conv):
            x_enc = _conv_bn_act_layer(x_enc, filters=layer_params['filters'], 
                                       kernel_size=layer_params['kernel_size'], 
                                       padding=layer_params['padding'],
                                       kernel_initializer=layer_params['kernel_initializer'], 
                                       name=f"{layer_params['name']}_{i}")

        # Finalmente la capa de MaxPooling
        x_enc = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, 
                                             padding='valid',
                                             name=f"MaxPool_Conv_{layer_params['name']}")(x_enc)
        return x_enc
    
    
    def _decoding_layer(input_layer, n_layers_conv, layer_params):
        '''Función auxiliar que permite modelar una capa de upsampling seguido de 
        "n_layers_conv" capas CNN, tal como se puede ver en la figura 2 de [1].  
        '''
        # Capa de upsampling
        x_dec = tf.keras.layers.UpSampling1D(size=2, name=f"Upsampling_"\
                                                          f"{layer_params['name']}")(input_layer)
        
        # Aplicando "n_layers_conv" capas convolucionales de decodificación
        for i in range(n_layers_conv):
            x_dec = _conv_bn_act_layer(x_dec, filters=layer_params['filters'], 
                                       kernel_size=layer_params['kernel_size'], 
                                       padding=layer_params['padding'],
                                       kernel_initializer=layer_params['kernel_initializer'], 
                                       name=f"{layer_params['name']}_{i}")

        return x_dec
    
    
    # Definición de la entrada
    x_in = tf.keras.Input(shape=input_shape, dtype='float32')

    # Definición de la capa de máscara
    x_masked = tf.keras.layers.Masking(mask_value=padding_value)(x_in)

    ############        Definición de las capas convolucionales        ############
    
    ### Encoding ###
    
    # Primera capa de encoding
    layer_params_1 = {'filters': 13, 'kernel_size': 100, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 'name': 'enc1'}
    x_enc1 = _encoding_layer(x_masked, n_layers_conv=2, layer_params=layer_params_1)
    
    # Segunda capa de encoding
    layer_params_2 = {'filters': 13, 'kernel_size': 100, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 'name': 'enc2'}
    x_enc2 = _encoding_layer(x_enc1, n_layers_conv=2, layer_params=layer_params_2)
    
    # Tercera capa de encoding
    layer_params_3 = {'filters': 13, 'kernel_size': 100, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 'name': 'enc3'}
    x_enc3 = _encoding_layer(x_enc2, n_layers_conv=3, layer_params=layer_params_3)
    
    # Cuarta capa de encoding
    layer_params_4 = {'filters': 13, 'kernel_size': 100, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 'name': 'enc4'}
    x_enc4 = _encoding_layer(x_enc3, n_layers_conv=3, layer_params=layer_params_4)
    
    
    ### Decoding ###
    
    # Cuarta capa de salida del decoding
    layer_params_4 = {'filters': 13, 'kernel_size': 100, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 'name': 'dec4'}
    x_dec4 = _decoding_layer(x_enc4, n_layers_conv=3, layer_params=layer_params_4)
    
    # Tercera capa de salida del decoding
    layer_params_3 = {'filters': 13, 'kernel_size': 100, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 'name': 'dec3'}
    x_dec3 = _decoding_layer(x_dec4, n_layers_conv=3, layer_params=layer_params_3)
    
    # Segunda capa de salida del decoding
    layer_params_2 = {'filters': 13, 'kernel_size': 100, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 'name': 'dec2'}
    x_dec2 = _decoding_layer(x_dec3, n_layers_conv=2, layer_params=layer_params_2)
    
    # Primera capa de salida del decoding
    layer_params_1 = {'filters': 13, 'kernel_size': 100, 'padding': 'same',
                      'kernel_initializer': 'he_normal', 'name': 'dec1'}
    x_dec1 = _decoding_layer(x_dec2, n_layers_conv=2, layer_params=layer_params_1)
                                       
    
    # Aplicando reshape
    # x_reshaped = tf.keras.layers.Reshape((input_shape[0], input_shape[1] * 2))(x_dec1)
    
    # Definición de la capa de salida
    x_out = tf.keras.layers.Dense(3, activation='softmax', kernel_initializer='he_normal',
                                  name='softmax_out')(x_dec1)
    
    # Definición del modelo
    model = tf.keras.Model(inputs=x_in, outputs=x_out, name=name)
    
    return model


# Módulo de testeo
if __name__ == '__main__':
    # model = model_5_2_9((128,3), padding_value=2, name='Testeo')
    model = model_5_2_4_3((128,3), padding_value=2, name='Testeo')
    # model = model_7_2((1024,3), padding_value=2, name='Testeo')
    print(model.summary())
    tf.keras.utils.plot_model(model, to_file='Testeo.png', show_shapes=True, 
                              expand_nested=True)
