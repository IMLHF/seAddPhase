from torch import nn
import torch
import collections
import numpy as np
from torch.nn.utils import clip_grad_norm_

from ..FLAGS import PARAM
from ..utils import losses
from ..utils import misc_utils
from ..models import conv_stft


class SelfConv2d(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size,
               stride=1, use_bias=True, padding='same', activation=None):
    super(SelfConv2d, self).__init__()
    assert padding.lower() in ['same', 'valid'], 'padding must be same or valid.'
    if padding.lower() == 'same':
      if type(kernel_size) is int:
        padding_nn = kernel_size // 2
      else:
        padding_nn = []
        for kernel_s in kernel_size:
          padding_nn.append(kernel_s // 2)
    self.conv2d_fn = nn.Conv2d(in_channels, out_channels, kernel_size,
                               stride=stride, bias=use_bias, padding=padding_nn)
    self.act = activation

  def forward(self, feature_in):
    """
    feature_in : [N, C, F, T]
    """
    out = feature_in
    out = self.conv2d_fn(out)
    if self.act is not None:
      out = self.act(out)
    return out


class BatchNormAndActivate(nn.Module):
  def __init__(self, channel, activation=nn.ReLU(inplace=True)):
    super(BatchNormAndActivate, self).__init__()
    self.bn_layer = nn.BatchNorm2d(channel)
    self.activate_fn = activation

  def forward(self, fea_in):
    """
    fea_in: [N, C, F, T]
    """
    out = self.bn_layer(fea_in)
    if self.activate_fn is not None:
      out = self.activate_fn(out)
    return out

  def get_bn_weight(self):
    return self.bn_layer.parameters()


class Stream_PreNet(nn.Module):
  def __init__(self, in_channels, out_channels, kernels=[[5, 1], [1, 5]],
               conv2d_activation=None, conv2d_bn=False):
    '''
    channel_out: output channel
    kernels: kernel for layers
    '''
    super(Stream_PreNet, self).__init__()
    self.nn_layers = nn.ModuleList()
    for i, kernel in enumerate(kernels):
      conv2d = SelfConv2d(in_channels if i==0 else out_channels, out_channels,
                          kernel_size=kernel,
                          activation=(None if conv2d_bn else conv2d_activation),
                          padding="same")
      self.nn_layers.append(conv2d)
      if conv2d_bn:
        bn_fn = BatchNormAndActivate(out_channels, activation=conv2d_activation)
        self.nn_layers.append(bn_fn)


  def forward(self, feature_in):
    '''
    feature_in : [batch, channel_in, F, T]
    return : [batch, channel_out, F, T]
    '''
    if len(self.nn_layers) == 0:
      return feature_in
    out = feature_in
    for layer_fn in self.nn_layers:
      out = layer_fn(out)
    return out


class NodeReshape(nn.Module):
  def __init__(self, shape):
    super(NodeReshape, self).__init__()
    self.shape = shape

  def forward(self, feature_in:torch.Tensor):
    shape = feature_in.size()
    batch = shape[0]
    new_shape = [batch]
    new_shape.extend(list(self.shape))
    return feature_in.reshape(new_shape)

class MagMaskNet(nn.Module):
  def __init__(self, frequency_dim, input_channel):
    super(MagMaskNet, self).__init__()
    self.reshape_in_blstm = NodeReshape([-1, frequency_dim*input_channel])
    self.blstm_layers = nn.ModuleList()
    for i in range(PARAM.blstm_layers_M):
      blstm = nn.LSTM(frequency_dim*input_channel if i==0 else PARAM.rnn_units_M[i-1]*2,
                      PARAM.rnn_units_M[i], batch_first=True, bidirectional=True)
      self.blstm_layers.append(blstm)
    self.out_fc = nn.Linear(PARAM.rnn_units_M[-1]*2, frequency_dim)

  def forward(self, feature_in:torch.Tensor):
    # feature_in: [N, T, frequency, input_channel]
    output = feature_in
    # print(output.size())
    output = self.reshape_in_blstm(output)
    for layer_fn in self.blstm_layers:
      output = layer_fn(output)[0]
    output = self.out_fc(output)
    return output # [N, T, F]


class PhaseNet(nn.Module):
  def __init__(self, frequency_dim, input_channel):
    super(PhaseNet, self).__init__()
    self.reshape_in_blstm = NodeReshape([-1, frequency_dim*input_channel])
    self.blstm_layers = nn.ModuleList()
    for i in range(PARAM.blstm_layers_P):
      blstm = nn.LSTM(frequency_dim*input_channel if i==0 else PARAM.rnn_units_P[i-1]*2,
                      PARAM.rnn_units_P[i], batch_first=True, bidirectional=True)
      self.blstm_layers.append(blstm)
    self.out_fc = nn.Linear(PARAM.rnn_units_P[-1]*2, frequency_dim*2)
    self.reshape_out = NodeReshape([-1, frequency_dim, 2])

  def forward(self, feature_in:torch.Tensor):
    # feature_in: [N, T, frequency_dim, input_channel]
    output = feature_in
    output = self.reshape_in_blstm(output) # [N, T, F*C]
    for layer_fn in self.blstm_layers:
      output = layer_fn(output)[0]
    output = self.out_fc(output) # [N, T, F*2]
    output = self.reshape_out(output) # [N, T, F, 2]
    out_real = output[:, :, :, :1]
    out_imag = output[:, :, :, 1:]
    out_angle = out_imag.atan2(out_real) # [N, T, F, 1]
    if PARAM.stft_norm_method == "atan2":
      normed_stft = torch.cat([torch.cos(out_angle),
                               torch.sin(out_angle)], dim=-1)
    elif PARAM.stft_norm_method == "div":
      normed_stft = torch.div(
          output, torch.sqrt(out_real**2+out_imag**2)+PARAM.stft_div_norm_eps)
    else:
      raise NotImplementedError
    return normed_stft, out_angle.squeeze(1) # [N, T, F, 2] [N, T, F]


class WavFeatures(
    collections.namedtuple("WavFeatures",
                           ("wav_batch", # [N, L]
                            "stft_batch", #[N, 2, F, T]
                            "mag_batch", # [N, F, T]
                            "angle_batch", # [N, F, T]
                            "normed_stft_batch", # [N, F, T]
                            ))):
  pass


class MPNet_OUT(
    collections.namedtuple("MPNet_OUT",
                           ("mag_mask", # [N, F, T]
                            "normalized_complex_phase", # [N, 2, F, T]
                            "angle"))): # [N, F, T]
  pass


class MPNet(nn.Module):
  def __init__(self):
    super(MPNet, self).__init__()
    M_in_channel = {
      "stft":2,
      "mag":1,
    }[PARAM.stream_A_feature_type]
    P_in_channel = {
      "stft":2,
      "normed_stft":2,
    }[PARAM.stream_P_feature_type]
    self.mag_mash_net = MagMaskNet(PARAM.frequency_dim, M_in_channel)
    self.phase_net = PhaseNet(PARAM.frequency_dim, P_in_channel)

  def forward(self, mixed_wav_features:WavFeatures):
    '''
    Args:
      mixed_wav_features
    Return :
      mag_batch[N, F, T]->real,
      normalized_complex_phase[N, 2, F, T]->(real, imag)
    '''
    Mnet_inputs = {
      "stft":mixed_wav_features.stft_batch, # [N, 2, F, T]
      "mag":mixed_wav_features.mag_batch.unsqueeze(1), # [N, 1, F, T]
    }[PARAM.stream_A_feature_type]
    Pnet_inputs = {
      "stft":mixed_wav_features.stft_batch, # [N, 2, F, T]
      "normed_stft":mixed_wav_features.normed_stft_batch,
    }[PARAM.stream_P_feature_type]

    est_mask = self.mag_mash_net(
        torch.transpose(Mnet_inputs, 1, 3))  # [batch, T, F]
    est_normed_stft, est_angle = self.phase_net(
        torch.transpose(Pnet_inputs, 1, 3))  # [batch, T, F, 2] [N, T, F]
    est_mask = torch.transpose(est_mask, 1, 2)
    est_normed_stft = torch.transpose(est_normed_stft, 1, 3)
    est_angle = torch.transpose(est_angle, 1, 2)

    return MPNet_OUT(mag_mask=est_mask, # [N, F, T]
                     normalized_complex_phase=est_normed_stft, # [N, 2, F, T]
                     angle=est_angle) # [N, F, T]


class Losses(
    collections.namedtuple("Losses",
                           ("sum_loss", "show_losses", "stop_criterion_loss"))):
  pass


class MODEL(nn.Module):
  def __init__(self, mode, device):
    super(MODEL, self).__init__()
    self.mode = mode
    self.device = device
    self._net_model = MPNet()
    self._stft_fn = conv_stft.ConvSTFT(PARAM.frame_length, PARAM.frame_step, PARAM.fft_length) # [N, 2, F, T]
    self._istft_fn = conv_stft.ConviSTFT(PARAM.frame_length, PARAM.frame_step, PARAM.fft_length) # [N, L]

    if mode == PARAM.MODEL_VALIDATE_KEY or mode == PARAM.MODEL_INFER_KEY:
      # self.eval(True)
      self.to(self.device)
      return

    # other params to save
    self._global_step = 1
    self._start_epoch = 1
    self._nan_grads_batch = 0

    # choose optimizer
    if PARAM.optimizer == "Adam":
      self._optimizer = torch.optim.Adam(self.parameters(), lr=PARAM.learning_rate)
    elif PARAM.optimizer == "RMSProp":
      self._optimizer = torch.optim.RMSprop(self.parameters(), lr=PARAM.learning_rate)

    # for lr warmup
    self._lr_scheduler = None
    if PARAM.use_lr_warmup:
      def warmup(step):
        return misc_utils.warmup_coef(step, warmup_steps=PARAM.warmup_steps)
      self._lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self._optimizer, warmup)

    # self.train(True)
    self.to(self.device)

  def save_every_epoch(self, ckpt_path):
    self._start_epoch += 1
    torch.save({
                "global_step": self._global_step,
                "start_epoch": self._start_epoch,
                "nan_grads_batch": self._nan_grads_batch,
                "other_state": self.state_dict(),
            }, ckpt_path)

  def load(self, ckpt_path):
    ckpt = torch.load(ckpt_path)
    self._global_step = ckpt["global_step"]
    self._start_epoch = ckpt["start_epoch"]
    self._nan_grads_batch = ckpt["nan_grads_batch"]
    self.load_state_dict(ckpt["other_state"])

  def update_params(self, loss):
    self.zero_grad()
    loss.backward()
    # deal grads

    # grads check nan or inf
    has_nan_inf = 0
    for params in self.parameters():
      if params.requires_grad:
        has_nan_inf += torch.sum(torch.isnan(params.grad))
        has_nan_inf += torch.sum(torch.isinf(params.grad))

    # print('has_nan', has_nan_inf)

    if has_nan_inf == 0:
      clip_grad_norm_(self.parameters(), PARAM.max_gradient_norm, norm_type=2)
      self._optimizer.step()
      self._lr_scheduler.step(self._global_step)
      self._global_step += 1
      return
    self._nan_grads_batch += 1

  def __call__(self, mixed_wav_batch):
    mixed_wav_batch = mixed_wav_batch.to(self.device)
    mixed_stft_batch = self._stft_fn(mixed_wav_batch) # [N, 2, F, T]
    mixed_stft_real = mixed_stft_batch[:, 0, :, :] # [N, F, T]
    mixed_stft_imag = mixed_stft_batch[:, 1, :, :] # [N, F, T]
    mixed_mag_batch = torch.sqrt(mixed_stft_real**2+mixed_stft_imag**2) # [N, F, T]
    mixed_angle_batch = torch.atan2(mixed_stft_imag, mixed_stft_real) # [N, F, T]
    if PARAM.stft_norm_method == "atan2":
      mixed_normed_stft_batch = torch.cat([torch.cos(mixed_angle_batch).unsqueeze_(1),
                                           torch.sin(mixed_angle_batch).unsqueeze_(1)], dim=1)
    elif PARAM.stft_norm_method == "div":
      _N, _F, _T = mixed_mag_batch.size()
      mixed_normed_stft_batch = torch.div(
          mixed_stft_batch, mixed_mag_batch.view(_N, 1, _F, _T)+PARAM.stft_div_norm_eps)
    else:
      raise NotImplementedError
    self.mixed_wav_features = WavFeatures(wav_batch=mixed_wav_batch,
                                          stft_batch=mixed_stft_batch,
                                          mag_batch=mixed_mag_batch,
                                          angle_batch=mixed_angle_batch,
                                          normed_stft_batch=mixed_normed_stft_batch)

    feature_in = self.mixed_wav_features # [N, 2, F, T]

    mp_net_out = self._net_model(feature_in)

    # print(self.mixed_wav_features.mag_batch.size(), mp_net_out.mag_mask.size())
    est_clean_angle_batch = mp_net_out.angle
    est_mag_batch = torch.functional.F.relu(torch.mul(
        self.mixed_wav_features.mag_batch, mp_net_out.mag_mask), inplace=True)  # [batch, F, T]
    mag_shape = est_mag_batch.size()
    est_normed_stft_batch = mp_net_out.normalized_complex_phase # [bathch, 2, F, T]
    est_stft_batch = torch.mul(
        est_mag_batch.view([mag_shape[0], 1, mag_shape[1], mag_shape[2]]),
        est_normed_stft_batch)
    est_wav_batch = self._istft_fn(est_stft_batch)
    _mixed_wav_length = self.mixed_wav_features.wav_batch.size()[-1]
    est_wav_batch = est_wav_batch[:, :_mixed_wav_length]

    return WavFeatures(wav_batch=est_wav_batch,
                       stft_batch=est_stft_batch,
                       mag_batch=est_mag_batch,
                       angle_batch=est_clean_angle_batch,
                       normed_stft_batch=est_normed_stft_batch)

  def get_losses(self, est_wav_features:WavFeatures, clean_wav_batch):
    self.clean_wav_batch = clean_wav_batch.to(self.device)
    self.clean_stft_batch = self._stft_fn(self.clean_wav_batch) # [N, 2, F, T]
    clean_stft_real = self.clean_stft_batch[:, 0, :, :] # [N, F, T]
    clean_stft_imag = self.clean_stft_batch[:, 1, :, :] # [N, F, T]
    self.clean_mag_batch = torch.sqrt(clean_stft_real**2+clean_stft_imag**2) # [N, F, T]
    self.clean_angle_batch = torch.atan2(clean_stft_imag, clean_stft_real) # [N, F, T]
    _N, _F, _T = self.clean_mag_batch.size()
    if PARAM.stft_norm_method == "atan2":
      self.clean_normed_stft_batch = torch.cat([torch.cos(self.clean_angle_batch).unsqueeze_(1),
                                                torch.sin(self.clean_angle_batch).unsqueeze_(1)], dim=1)
    elif PARAM.stft_norm_method == "div":
      self.clean_normed_stft_batch = torch.div(
          self.clean_stft_batch, self.clean_mag_batch.view(_N, 1, _F, _T)+PARAM.stft_div_norm_eps)
    else:
      raise NotImplementedError

    est_mag_batch = est_wav_features.mag_batch
    est_stft_batch = est_wav_features.stft_batch
    est_wav_batch = est_wav_features.wav_batch
    est_normed_stft_batch = est_wav_features.normed_stft_batch

    all_losses = list()
    all_losses.extend(PARAM.sum_losses)
    all_losses.extend(PARAM.show_losses)
    all_losses.extend(PARAM.stop_criterion_losses)
    all_losses = set(all_losses)

    self.loss_compressedMag_mse = 0
    self.loss_compressedStft_mse = 0
    self.loss_mag_mse = 0
    self.loss_mag_reMse = 0
    self.loss_stft_mse = 0
    self.loss_stft_reMse = 0
    self.loss_mag_mae = 0
    self.loss_mag_reMae = 0
    self.loss_stft_mae = 0
    self.loss_stft_reMae = 0
    self.loss_wav_L1 = 0
    self.loss_wav_L2 = 0
    self.loss_wav_reL2 = 0
    self.loss_CosSim = 0
    self.loss_SquareCosSim = 0
    self.FTloss_mag_mse = 0
    self.FTcleanmag_EstP_stftMSE = 0
    self.FTestmag_EstP_stftMSE = 0

    # region losses
    if "loss_compressedMag_mse" in all_losses:
      self.loss_compressedMag_mse = losses.FSum_compressedMag_mse(
          est_mag_batch, self.clean_mag_batch, PARAM.loss_compressedMag_idx)
    if "loss_compressedStft_mse" in all_losses:
      self.loss_compressedStft_mse = losses.FSum_compressedStft_mse(
          est_mag_batch, est_normed_stft_batch,
          self.clean_mag_batch, self.clean_normed_stft_batch,
          PARAM.loss_compressedMag_idx)


    if "loss_mag_mse" in all_losses:
      self.loss_mag_mse = losses.FSum_MSE(est_mag_batch, self.clean_mag_batch)
    if "loss_mag_reMse" in all_losses:
      self.loss_mag_reMse = losses.FSum_relativeMSE(est_mag_batch, self.clean_mag_batch,
                                                    PARAM.relative_loss_epsilon, PARAM.RL_idx)
    if "loss_stft_mse" in all_losses:
      self.loss_stft_mse = losses.FSum_MSE(est_stft_batch, self.clean_stft_batch)
    if "loss_stft_reMse" in all_losses:
      self.loss_stft_reMse = losses.FSum_relativeMSE(est_stft_batch, self.clean_stft_batch,
                                                     PARAM.relative_loss_epsilon, PARAM.RL_idx)


    if "loss_mag_mae" in all_losses:
      self.loss_mag_mae = losses.FSum_MAE(est_mag_batch, self.clean_mag_batch)
    if "loss_mag_reMae" in all_losses:
      self.loss_mag_reMae = losses.FSum_relativeMAE(est_mag_batch, self.clean_mag_batch,
                                                    PARAM.relative_loss_epsilon)
    if "loss_stft_mae" in all_losses:
      self.loss_stft_mae = losses.FSum_MAE(est_stft_batch, self.clean_stft_batch)
    if "loss_stft_reMae" in all_losses:
      self.loss_stft_reMae = losses.FSum_relativeMAE(est_stft_batch, self.clean_stft_batch,
                                                     PARAM.relative_loss_epsilon)


    if "loss_wav_L1" in all_losses:
      self.loss_wav_L1 = losses.FSum_MAE(est_wav_batch, self.clean_wav_batch)
    if "loss_wav_L2" in all_losses:
      self.loss_wav_L2 = losses.FSum_MSE(est_wav_batch, self.clean_wav_batch)
    if "loss_wav_reL2" in all_losses:
      self.loss_wav_reL2 = losses.FSum_relativeMSE(est_wav_batch, self.clean_wav_batch,
                                                   PARAM.relative_loss_epsilon, PARAM.RL_idx)

    if "loss_CosSim" in all_losses:
      self.loss_CosSim = losses.batchMean_CosSim_loss(est_wav_batch, self.clean_wav_batch)
    if "loss_SquareCosSim" in all_losses:
      self.loss_SquareCosSim = losses.batchMean_SquareCosSim_loss(
          est_wav_batch, self.clean_wav_batch)
    # self.loss_stCosSim = losses.batch_short_time_CosSim_loss(
    #     est_wav_batch, self.clean_wav_batch,
    #     PARAM.st_frame_length_for_loss,
    #     PARAM.st_frame_step_for_loss)
    # self.loss_stSquareCosSim = losses.batch_short_time_SquareCosSim_loss(
    #     est_wav_batch, self.clean_wav_batch,
    #     PARAM.st_frame_length_for_loss,
    #     PARAM.st_frame_step_for_loss)

    def FixULawT_fn(x, u):
      # x: [batch, time, fea]
      y = torch.log(x * u + 1.0) / np.log(u + 1.0)
      return y
    clean_mag_batch_FT = FixULawT_fn(self.clean_mag_batch, PARAM.fixU)
    est_mag_batch_FT = FixULawT_fn(est_mag_batch, PARAM.fixU)
    cleanmagFT_cleanP_stft = clean_mag_batch_FT.view(_N, 1, _F, _T) * self.clean_normed_stft_batch
    if "FTloss_mag_mse" in all_losses:
      self.FTloss_mag_mse = losses.FSum_MSE(est_mag_batch_FT, clean_mag_batch_FT)
    if "FTcleanmag_EstP_stftMSE" in all_losses:
      cleanmagFT_estP_stft = clean_mag_batch_FT.view(_N, 1, _F, _T) * est_normed_stft_batch
      self.FTcleanmag_EstP_stftMSE = losses.FSum_MSE(cleanmagFT_estP_stft, cleanmagFT_cleanP_stft)
    if "FTestmag_EstP_stftMSE" in all_losses:
      estmagFT_estP_stft = est_mag_batch_FT.view(_N, 1, _F, _T) * est_normed_stft_batch
      self.FTestmag_EstP_stftMSE = losses.FSum_MSE(estmagFT_estP_stft, cleanmagFT_cleanP_stft)

    loss_dict = {
        'loss_compressedMag_mse': self.loss_compressedMag_mse,
        'loss_compressedStft_mse': self.loss_compressedStft_mse,
        'loss_mag_mse': self.loss_mag_mse,
        'loss_mag_reMse': self.loss_mag_reMse,
        'loss_stft_mse': self.loss_stft_mse,
        'loss_stft_reMse': self.loss_stft_reMse,
        'loss_mag_mae': self.loss_mag_mae,
        'loss_mag_reMae': self.loss_mag_reMae,
        'loss_stft_mae': self.loss_stft_mae,
        'loss_stft_reMae': self.loss_stft_reMae,
        'loss_wav_L1': self.loss_wav_L1,
        'loss_wav_L2': self.loss_wav_L2,
        'loss_wav_reL2': self.loss_wav_reL2,
        'loss_CosSim': self.loss_CosSim,
        'loss_SquareCosSim': self.loss_SquareCosSim,
        # 'loss_stCosSim': self.loss_stCosSim,
        # 'loss_stSquareCosSim': self.loss_stSquareCosSim,
        'FTloss_mag_mse': self.FTloss_mag_mse,
        'FTcleanmag_EstP_stftMSE': self.FTcleanmag_EstP_stftMSE,
        'FTestmag_EstP_stftMSE': self.FTestmag_EstP_stftMSE,
    }
    # endregion losses

    # region sum_loss
    sum_loss = 0.0
    sum_loss_names = PARAM.sum_losses
    for i, name in enumerate(sum_loss_names):
      loss_t = loss_dict[name]
      if len(PARAM.sum_losses_w) > 0:
        loss_t *= PARAM.sum_losses_w[i]
      sum_loss += loss_t
    # endregion sum_loss

    # region show_losses
    show_losses = []
    show_loss_names = PARAM.show_losses
    for i, name in enumerate(show_loss_names):
      loss_t = loss_dict[name]
      if len(PARAM.show_losses_w) > 0:
        loss_t *= PARAM.show_losses_w[i]
      show_losses.append(loss_t)
    show_losses = torch.stack(show_losses)
    # endregion show_losses

    # region stop_criterion_losses
    stop_criterion_losses_sum = 0.0
    stop_criterion_loss_names = PARAM.stop_criterion_losses
    for i, name in enumerate(stop_criterion_loss_names):
      loss_t = loss_dict[name]
      if len(PARAM.stop_criterion_losses_w) > 0:
        loss_t *= PARAM.stop_criterion_losses_w[i]
      stop_criterion_losses_sum += loss_t
    # endregion stop_criterion_losses

    return Losses(sum_loss=sum_loss,
                  show_losses=show_losses,
                  stop_criterion_loss=stop_criterion_losses_sum)


  @property
  def global_step(self):
    return self._global_step

  @property
  def start_epoch(self):
    return self._start_epoch

  @property
  def nan_grads_batch(self):
    return self._nan_grads_batch

  @property
  def optimizer_lr(self):
    return self._optimizer.param_groups[0]['lr']
