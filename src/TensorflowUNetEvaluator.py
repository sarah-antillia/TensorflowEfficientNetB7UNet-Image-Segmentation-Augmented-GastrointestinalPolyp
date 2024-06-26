# Copyright 2023 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# TensorflowUNetEvaluator.py
# 2023/05/30 to-arai
# 2023/08/22 Updated to use test dataset on 
#[model]
#GENERATOR = True

import os
import sys

import shutil

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"]="true"

import traceback

from ConfigParser import ConfigParser
from ImageMaskDataset import ImageMaskDataset
from BaseImageMaskDataset import BaseImageMaskDataset

from TensorflowUNet import TensorflowUNet
from TensorflowAttentionUNet import TensorflowAttentionUNet 
from TensorflowEfficientUNet import TensorflowEfficientUNet
from TensorflowMultiResUNet import TensorflowMultiResUNet
from TensorflowSwinUNet import TensorflowSwinUNet
from TensorflowTransUNet import TensorflowTransUNet
from TensorflowUNet3Plus import TensorflowUNet3Plus
from TensorflowU2Net import TensorflowU2Net
from TensorflowSharpUNet import TensorflowSharpUNet
#from TensorflowBASNet    import TensorflowBASNet
from TensorflowDeepLabV3Plus import TensorflowDeepLabV3Plus
from TensorflowEfficientNetB7UNet import TensorflowEfficientNetB7UNet

MODEL  = "model"
TRAIN  = "train"
EVAL   = "eval"
TEST   = "test"
# 2024/03/05 Added
DATASETCLASS ="datasetclass"

if __name__ == "__main__":
  try:
    config_file    = "./train_eval_infer.config"
    if len(sys.argv) == 2:
      config_file = sys.argv[1]

    config = ConfigParser(config_file)
    generator  = config.get(MODEL, "generator", dvalue=False)
    print("=== TensorflowUNetEvaluator")
    print("=== config generator {}".format(generator))

    ModelClass = eval(config.get(MODEL, "model", dvalue="TensorflowUNet"))
    print("=== ModelClass {}".format(ModelClass))
    model     = ModelClass(config_file)

    # Create a DatasetClass
    # 2024/03/05 MODEL -> DATASETCLASS
    DatasetClass = eval(config.get(DATASETCLASS, "datasetclass", dvalue="ImageMaskDataset"))
    print("=== DatasetClass {}".format(DatasetClass))
    dataset = DatasetClass(config_file)

    target = EVAL
    if generator:
      target = TEST
    x_test, y_test = dataset.create(dataset=target)
  
    model.evaluate(x_test, y_test)

  except:
    traceback.print_exc()
    
