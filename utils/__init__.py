# add current path
import os, sys
sys.path.append(os.path.dirname(__file__))

CURRENT_VERSION = 1.0
print("SCRIPT::{} version is {}".format(__file__, CURRENT_VERSION))

import dataset_utils
from dataset_utils import *