import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.io import loadmat
from prediction_functions import hss_segmentation


# Módulo de testeo de repositorio #
if __name__ == '__main__':
    # Parámetros
    lowpass_params = {'freq_pass': 140, 'freq_stop': 150}
    model_name = 'definitive_segnet_based'
    db_folder = 'samples_test'
    
    # Abriendo audio de ejemplo
    filename = 'samples_test/435_Patient081_Dis1'
    audio, samplerate = sf.read(f'{filename}.wav')
    labels = loadmat(f'{filename}.mat')['PCG_states']
    
    # Obteniendo la salida de la red
    _, y_hat_to, (y_out2, y_out3, y_out4) = \
            hss_segmentation(audio, samplerate, model_name,
                            length_desired=len(audio),
                            lowpass_params=lowpass_params,
                            plot_outputs=False)
    
    # Creación de la figura
    fig, axs = plt.subplots(3, 1, figsize=(15,10), sharex=True, frameon=True)

    audio_data_plot = 0.5 * audio / max(abs(audio))
    axs[0].plot(audio_data_plot + 0.5, label=r'$s(n)$', color='silver', zorder=0)
    axs[0].plot(y_hat_to[0,:,0], label=r'$S_0$', color='limegreen', zorder=2)
    axs[0].plot(y_hat_to[0,:,1], label=r'$S_1$', color='red', zorder=1)
    axs[0].plot(y_hat_to[0,:,2], label= r'$S_2$', color='blue', zorder=1)
    axs[0].legend(loc='lower right')
    axs[0].set_title('Predicción de sonidos cardiacos')
    axs[0].set_yticks([0, 0.5, 1])
    axs[0].set_ylabel(r'$P(y(n) = k | X)$')

    axs[1].plot(y_out3)
    axs[1].set_ylabel(r'$y(n)$ (3 labels)')
    axs[1].set_yticks([0,1,2])
    axs[1].set_yticklabels([r'$S_0$', r'$S_1$', r'$S_2$'])
    axs[1].set_ylim([-0.3, 2.3])

    axs[2].plot(np.repeat(labels, 20), label='Real', color='C0')
    axs[2].plot(y_out4, label='Pred', color='C1', linestyle='--', linewidth=3)
    axs[2].set_xlabel('Tiempo [ms]')
    axs[2].set_yticks([1,2,3,4])
    axs[2].set_yticklabels([r'$S_1$', 'Sys', r'$S_2$', 'Dia'])
    axs[2].set_ylabel(r"$y'(n)$ (4 labels)")
    axs[2].set_ylim([0.7, 4.3])
    axs[2].legend(loc='lower right')

    # Alineando los labels del eje y
    fig.align_ylabels(axs[:])

    # Remover espacio horizontal entre plot_outputs
    fig.subplots_adjust(wspace=0.1, hspace=0)

    plt.show()
    