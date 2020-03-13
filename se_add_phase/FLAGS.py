class StaticKey(object):
  MODEL_TRAIN_KEY = 'train'
  MODEL_VALIDATE_KEY = 'validation'
  MODEL_INFER_KEY = 'infer'

  def config_name(self): # config_name
    return self.__class__.__name__

class BaseConfig(StaticKey):
  VISIBLE_GPU = "0"
  root_dir = '/home/lhf/worklhf/seAddPhase/'
  # datasets_name = 'vctk_musan_datasets'
  datasets_name = 'noisy_datasets_16k'
  '''
  # dir to store log, model and results files:
  $root_dir/$datasets_name: datasets dir
  $root_dir/exp/$config_name/log: logs(include tensorboard log)
  $root_dir/exp/$config_name/ckpt: ckpt
  $root_dir/exp/$config_name/enhanced_testsets: enhanced results
  $root_dir/exp/$config_name/hparams
  '''

  # min_TF_version = "1.14.0"
  min_Torch_version = "1.0.0"


  train_noisy_set = 'noisy_trainset_wav'
  train_clean_set = 'clean_trainset_wav'
  validation_noisy_set = 'noisy_testset_wav'
  validation_clean_set = 'clean_testset_wav'
  test_noisy_sets = ['noisy_testset_wav']
  test_clean_sets = ['clean_testset_wav']

  n_train_set_records = 11572
  n_val_set_records = 824
  n_test_set_records = 824

  train_val_wav_seconds = 3.0

  batch_size = 12

  relative_loss_epsilon = 0.1
  RL_idx = 2.0
  st_frame_length_for_loss = 512
  st_frame_step_for_loss = 256
  sampling_rate = 16000
  frame_length = 400
  frame_step = 160
  fft_length = 512
  optimizer = "Adam" # "Adam" | "RMSProp"
  learning_rate = 0.001
  max_gradient_norm = 5.0

  GPU_RAM_ALLOW_GROWTH = True
  GPU_PARTION = 0.97

  max_epoch = 40
  batches_to_logging = 200000

  use_lr_warmup = True # true: lr warmup; false: lr halving
  warmup_steps = 6000. # for (use_lr_warmup == true)

  # melMat: tf.contrib.signal.linear_to_mel_weight_matrix(129,129,8000,125,3900)
  # plt.pcolormesh
  # import matplotlib.pyplot as plt

  """
  @param losses:
  see model.py : MODEL.get_losses()
  """
  sum_losses = ["loss_compressedMag_mse", "loss_compressedStft_mse"]
  sum_losses_w = []
  show_losses = ["loss_compressedMag_mse", "loss_compressedStft_mse"]
  show_losses_w = []
  stop_criterion_losses = ["loss_compressedMag_mse", "loss_compressedStft_mse"]
  stop_criterion_losses_w = []

  blstm_layers_M = 3
  blstm_layers_P = 3
  rnn_units_M = [512, 512, 512]
  rnn_units_P = [512, 512, 512]
  frequency_dim = 257
  loss_compressedMag_idx = 0.3

  stream_A_feature_type = "stft" # "stft" | "mag"
  stream_P_feature_type = "stft" # "stft" | "normed_stft"
  stft_norm_method = "atan2" # atan2 | div
  stft_div_norm_eps = 1e-5 # for stft_norm_method=div

  clip_grads = False

  fixU = 50.0


class p40(BaseConfig):
  # GPU_PARTION = 0.27
  root_dir = '/home/zhangwenbo5/lihongfeng/seAddPhase'


class pse_cprMSE(p40):
  sum_losses = ["loss_compressedMag_mse", "loss_compressedStft_mse"]
  show_losses = ["loss_compressedMag_mse", "loss_compressedStft_mse",
                 "loss_mag_mse", "loss_stft_mse", "loss_CosSim"]
  stop_criterion_losses = ["loss_compressedMag_mse", "loss_compressedStft_mse"]

class pse_FTloss_CleanMagForPhase(p40):
  sum_losses = ["FTloss_mag_mse", "FTcleanmag_EstP_stftMSE"]
  show_losses = ["FTloss_mag_mse", "FTcleanmag_EstP_stftMSE",
                 "loss_mag_mse", "loss_stft_mse", "loss_CosSim"]
  stop_criterion_losses = ["FTloss_mag_mse", "FTcleanmag_EstP_stftMSE"]
  fixU = 50.0

class pse_FTloss_EstMagForPhase(p40):
  sum_losses = ["FTloss_mag_mse", "FTestmag_EstP_stftMSE", "FTcleanmag_EstP_stftMSE"]
  show_losses = ["FTloss_mag_mse", "FTestmag_EstP_stftMSE", "FTcleanmag_EstP_stftMSE",
                 "loss_mag_mse", "loss_stft_mse", "loss_CosSim"]
  stop_criterion_losses = ["FTloss_mag_mse", "FTestmag_EstP_stftMSE"]
  fixU = 50.0


PARAM = pse_cprMSE

# CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 python -m pse_FTloss_EstMagForPhase._2_train
