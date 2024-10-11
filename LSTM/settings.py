import sys 
import os 

import torch.nn as nn
from soft_dtw_cuda import SoftDTW

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datetime import datetime, timedelta

from pytorch_forecasting.metrics import MAPE
from credentials import user_pass


LOOP_NAME = "My_loop"

##VERSION: folder name. Results stored in ./lstm/[version]
VERSION = '12'

#### DB settings

# Nagios database connection string
# http://lin033dsy/nagios/
DB_CONNECTION_STRING = 'mysql+pymysql://ndoutils:{}@lin033dsy/nagios'.format(user_pass)

#### Dataset settings

#date limits dataset
NUM_WEEKS = 9
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(weeks = NUM_WEEKS)

# Options to pass to DB query function
DB_QUERY_ARGS = dict(
    connection_string=DB_CONNECTION_STRING,
    interval='hour',
    start_date=  START_DATE, #datetime.fromisoformat('2021-03-25 11:00:00')
    end_date=  END_DATE, # datetime.fromisoformat('2021-05-27 10:00:00') 
    os='windows',
    machines_to_include= None, #['client14dsy', 'client14xdsy'],
    machines_to_exclude= ['wip132dsy', 'ssdwip010dsy', 'ssdwip017dsy', 'ssdwip018dsy', 'ssdwip020dsy', 'ssdwip021dsy', 'ssdwip022dsy'],
    limit=None
)

# If specified, increases interval between records to the specified
# interval, with CPU values averaged accordingly
INTERVAL_SECONDS = None

# CPU values normalization thresholds
# Eg, if set to [10, 30, 60, 100], with input series
# (7, 12, 30, 55, 80, 90, 100) would get normalized to
# (0, 1, 1, 2, 3, 3, 3)
# Not applied if set to None
CPU_THRESHOLD_NORMALIZATION = None # [5, 30, 60, 100] 


PARAM = dict(
    seq_len = 24,
    batch_size = 1, 
    criterion = nn.MSELoss(), #MAPE(), SoftDTW()
    max_epochs = 10,
    hidden_size = 200,
    num_layers = 2,
    dropout = 0.2,
    learning_rate = 0.001,
)

#specify if GPU is desired
GPU_DESIRED = True
