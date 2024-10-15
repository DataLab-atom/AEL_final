import warnings

warnings.filterwarnings('ignore')

test = True
if test:
    import sys
    import os
    ABS_PATH = os.path.dirname(os.path.abspath(__file__))
    ROOT_PATH = os.path.join(ABS_PATH, "..", "..")
    sys.path.append(ROOT_PATH)  # This is for finding all the modules
    from aell.src.aell import ael_no_threading
    from aell.src.aell.utils import createFolders
else:
    from aell import ael
    from aell.utils import createFolders

### LLM settings  ###

use_local_llm = False # if use local model
url= None # your local server 'http://127.0.0.1:11012/completions'

api_endpoint = "openai.api2d.net"
api_key = "fk221426-8c6wYuZWYR5wrztZey4YxiRSgTeCGAbg" # use your key 
#llm_model = "gpt-3.5-turbo-1106"
llm_model = "gpt-3.5-turbo"
### output path ###
output_path = "./" # default folder for ael outputs
createFolders.create_folders(output_path)
load_data = {
    'use_seed' : True,
    'seed_path' : output_path+"/ael_seeds/seeds.json",
    "use_pop" : False,
    "pop_path" : output_path+"/ael_results/pops/population_generation_6.json",
    "n_pop_initial": 0
    }

### Experimental settings ###
pop_size = 10 # number of algorithms in each population, default = 10
n_pop = 3 # number of populations, default = 10
operators = ['e1','e2','m1','m2']  # evolution operators: ['e1','e2','m1','m2'], default = ['e1','m1']
m = 2  # number of parents for 'e1' and 'e2' operators, default = 2
operator_weights = [1,1,1,1] # weights for operators, i.e., the probability of use the operator in each iteration , default = [1,1,1,1]

### Debug model ###
debug_mode = False# if debug

########日志处理
import logging

# 创建一个日志记录器
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)  # 设置日志级别

# 创建一个文件处理器，将日志写入文件
file_handler = logging.FileHandler(f'runael_PROTEIN_{pop_size}_{n_pop}_{m}.log')
file_handler.setLevel(logging.DEBUG)  # 设置文件处理器的日志级别

# 创建一个日志格式器，并将其添加到处理器中
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# 将处理器添加到记录器中
logger.addHandler(file_handler)
#########

# AEL
print(">>> Start AEL ")
algorithmEvolution = ael_no_threading.AEL(use_local_llm, url,\
    api_endpoint,api_key,llm_model,pop_size,n_pop,operators,m,operator_weights,load_data,output_path,debug_mode, logger=logger)

# run AEL
algorithmEvolution.run()

print("AEL successfully finished !")


