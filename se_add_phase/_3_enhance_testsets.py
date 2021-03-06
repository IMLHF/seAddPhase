import os
import tensorflow as tf
import collections
from pathlib import Path
import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
import sys

from .utils import misc_utils
from .utils import audio
from .inference import build_model
from .inference import enhance_one_wav

from .FLAGS import PARAM

test_processor = 1
ckpt = None
noisy_phase=False
model = None

def enhance_mini_process(noisy_dir, enhanced_save_dir):
  global model
  if model is None:
    model = build_model(ckpt_dir=ckpt)
  noisy_wav, sr = audio.read_audio(noisy_dir)
  enhanced_wav = enhance_one_wav(model, noisy_wav, noisy_phase)
  noisy_name = Path(noisy_dir).stem
  audio.write_audio(os.path.join(enhanced_save_dir, noisy_name+'_enhanced.wav'),
                    enhanced_wav, PARAM.sampling_rate)


def enhance_one_testset(testset_dir, enhanced_save_dir):
  testset_path = Path(testset_dir)
  noisy_path_list = list(map(str, testset_path.glob("*.wav")))
  func = partial(enhance_mini_process, enhanced_save_dir=enhanced_save_dir)
  # for noisy_path in noisy_path_list:
  #   func(noisy_path)
  job = Pool(test_processor).imap(func, noisy_path_list)
  list(tqdm(job, "Enhancing", len(noisy_path_list), unit="test wav", ncols=60))


def main():
  for testset_name in PARAM.test_noisy_sets:
    print("Enhancing %s:" % testset_name, flush=True)
    _dir = misc_utils.enhanced_testsets_save_dir(testset_name)
    if _dir.exists():
      import shutil
      shutil.rmtree(str(_dir))
    _dir.mkdir(parents=True)
    testset_dir = str(misc_utils.datasets_dir().joinpath(testset_name))
    enhanced_save_dir = str(_dir)
    enhance_one_testset(testset_dir, enhanced_save_dir)

if __name__ == "__main__":
  misc_utils.initial_run(sys.argv[0].split("/")[-2])

  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--n_process', default=1, type=int, help="n processor")
  parser.add_argument('--ckpt', default=None, type=str, help="ckpt dir")
  parser.add_argument('--noisy_phase', default=0, type=int, help='if use noisy phase')
  args = parser.parse_args()

  test_processor = args.n_process
  ckpt = args.ckpt
  noisy_phase = bool(args.noisy_phase)

  print('n_process:', args.n_process)
  print("noisy_phase:", noisy_phase)
  print('ckpt:', args.ckpt)

  main()

  """
  run cmd:
  `OMP_NUM_THREADS=1 python -m xx._3_enhance_testsets --n_process=2 --noisy_phase=False`
  [csig,cbak,cvol,pesq,snr,ssnr]=evaluate_all('/home/lhf/worklhf/seAddPhase/noisy_datasets_16k/clean_testset_wav','/home/lhf/worklhf/seAddPhase/exp/se_reMagMSE_cnn/enhanced_testsets/noisy_testset_wav')
  """
