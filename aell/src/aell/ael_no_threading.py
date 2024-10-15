import threading

import numpy as np
import json
import random
from threading import Thread
from datetime import datetime

from ael_evaluation import Evaluation
from .ec.management import population_management
from .ec.interface_EC import InterfaceEC

# main class for AEL
class AEL:

    # initilization
    def __init__(self, use_local_llm, url, api_endpoint, api_key, model, pop_size, n_pop, operators, m,
                 operator_weights, load_pop,
                 out_path, debug_mode, logger, **kwargs):

        # LLM settings
        self.use_local_llm = use_local_llm
        self.url = url
        self.api_endpoint = api_endpoint  # currently only API2D + GPT
        self.api_key = api_key
        self.llm_model = model

        #
        self.logger = logger

        # ------------------ RZ: use local LLM ------------------
        # self.use_local_llm = kwargs.get('use_local_llm', False)

        assert isinstance(self.use_local_llm, bool)
        '''
        if self.use_local_llm:
            assert 'url' in kwargs, 'The keyword "url" should be provided when use_local_llm is True.'
            assert isinstance(kwargs.get('url'), str)
            self.url = kwargs.get('url')
        # -------------------------------------------------------
        '''

        # Experimental settings
        self.pop_size = pop_size  # popopulation size, i.e., the number of algorithms in population
        self.n_pop = n_pop  # number of populations

        self.operators = operators
        self.operator_weights = operator_weights
        if m > pop_size or m == 1:
            self.logger.info("m should not be larger than pop size or smaller than 2, adjust it to m=2")
            m = 2
        self.m = m

        self.debug_mode = debug_mode  # if debug
        self.ndelay = 1  # default
        self.load_pop = load_pop
        self.output_path = out_path

        # Set a random seed
        random.seed(2024)

    # add new individual to population
    def add2pop(self, population, offspring):
        for ind in population:
            if ind['objective'] == offspring['objective']:
                self.logger.info("duplicated result, retrying ... ")
                return False
        population.append(offspring)
        return True

    # run ael
    def run(self):

        t0 = datetime.now()

        # interface for large language model (llm)
        # interface_llm = PromptLLMs(self.api_endpoint,self.api_key,self.llm_model,self.debug_mode)

        # interface for evaluation
        interface_eval = Evaluation()

        # interface for ec operators
        interface_ec = InterfaceEC(self.pop_size, self.m, self.api_endpoint, self.api_key, self.llm_model,
                                   self.debug_mode, interface_eval, use_local_llm=self.use_local_llm, url=self.url)

        # initialization
        population = []
        if 'use_seed' in self.load_pop and self.load_pop['use_seed']:
            import sys, os
            self.logger.info(os.listdir())
            # os.chdir(r'D:\300work\303HKTasks\AEL\AEL-main\examples\QuadraticFunction')
            with open(self.load_pop['seed_path']) as file:
                data = json.load(file)
            s = datetime.now()
            population = interface_ec.population_generation_seed(data)
            e = datetime.now()
            self.logger.info(f"test population generation seed time: {(e-s).seconds / 60} m")
            # th: 将seed的准确度重新计算后写回
            with open(self.load_pop['seed_path'], 'w') as file:
                json.dump(population, file, indent=5)
            self.logger.info("write back seeds done.")
            ####
            filename = self.output_path + "/ael_results/pops/population_generation_0.json"
            with open(filename, 'w') as f:
                json.dump(population, f, indent=5)
            n_start = 0
        else:
            if self.load_pop['use_pop']:  # load population from files
                self.logger.info("load initial population from " + self.load_pop['pop_path'])
                with open(self.load_pop['pop_path']) as file:
                    data = json.load(file)
                for individual in data:
                    population.append(individual)
                self.logger.info("initial population has been loaded!")
                n_start = self.load_pop['n_pop_initial']
            else:  # create new population
                self.logger.info("creating initial population:")
                population = interface_ec.population_generation()
                population = population_management(population, self.pop_size)
                self.logger.info("initial population has been created!")
                # Save population to a file
                filename = self.output_path + "/ael_results/pops/population_generation_0.json"
                with open(filename, 'w') as f:
                    json.dump(population, f, indent=5)
                n_start = 0

        t1 = datetime.now()

        # main loop
        n_op = len(self.operators)

        for pop in range(n_start, self.n_pop):  # 对于每个population
            # Perform crossover and mutation
            for na in range(self.pop_size):  # 对于内部的每个增强方法
                for i in range(n_op):  # 选择不同的操作

                    s = datetime.now()

                    op = self.operators[i]
                    op_w = self.operator_weights[i]
                    if (np.random.rand() < op_w):
                        parents, offspring = interface_ec.get_algorithm(population, op)  # 运行大模型+进化操作得到结果
                    is_add = self.add2pop(population, offspring)  # Check duplication, and add the new offspring
                    self.logger.info("generate new algorithm using " + op + " with fitness value: " + str(float(offspring['objective'])))
                    if is_add:
                        data = {}
                        for i in range(len(parents)):
                            data[f"parent{i + 1}"] = parents[i]
                        data["offspring"] = offspring
                        with open(self.output_path + f"/ael_results/history/pop_" + str(pop + 1) + "_" + str(
                                na) + "_" + op + ".json", "w") as file:  # 保存中间结果
                            json.dump(data, file, indent=5)

                    e = datetime.now()
                    self.logger.info(f"evolution 1 cost time: {(e - s).seconds / 60} m")

                # populatin management
                size_act = min(len(population), self.pop_size)
                population = population_management(population, size_act)  # 会根据表现删除掉最差的population
                self.logger.info(f">> {na + 1} of {self.pop_size} finished ")

            # 等待所有线程退出
            self.logger.info("fitness values of current population: ")
            for i in range(self.pop_size):  # 每种population对应不同的表现
                self.logger.info(str(population[i]['objective']) + " ")

            # Save population to a file（保存所有）
            filename = self.output_path + f"/ael_results/pops/population_generation_" + str(pop + 1) + ".json"
            with open(filename, 'w') as f:
                json.dump(population, f, indent=5)

            # Save the best one to a file（保存最好的）
            filename = self.output_path + f"/ael_results/pops_best/population_generation_" + str(pop + 1) + ".json"
            with open(filename, 'w') as f:
                json.dump(population[0], f, indent=5)

            self.logger.info(f">>> {pop + 1} of {self.n_pop} populations finished ")

        t2 = datetime.now()
        self.logger.info(f"t0~t1: {(t1-t0).seconds}s")
        self.logger.info(f"t1~t2: {(t2-t1).seconds}s")





































"""
import numpy as np
import json
import random

from ael_evaluation import Evaluation
from .ec.management import population_management
from .ec.interface_EC import InterfaceEC


# main class for AEL
class AEL:

    # initilization
    def __init__(self, use_local_llm,url,api_endpoint, api_key, model, pop_size, n_pop, operators, m, operator_weights, load_pop,
                 out_path, debug_mode, **kwargs):

        # LLM settings
        self.use_local_llm = use_local_llm
        self.url = url
        self.api_endpoint = api_endpoint  # currently only API2D + GPT
        self.api_key = api_key
        self.llm_model = model

        # ------------------ RZ: use local LLM ------------------
        self.use_local_llm = kwargs.get('use_local_llm', False)
        assert isinstance(self.use_local_llm, bool)
        if self.use_local_llm:
            assert 'url' in kwargs, 'The keyword "url" should be provided when use_local_llm is True.'
            assert isinstance(kwargs.get('url'), str)
            self.url = kwargs.get('url')
        # -------------------------------------------------------

        # Experimental settings       
        self.pop_size = pop_size  # popopulation size, i.e., the number of algorithms in population
        self.n_pop = n_pop  # number of populations

        self.operators = operators
        self.operator_weights = operator_weights
        if m > pop_size or m == 1:
            self.logger.info("m should not be larger than pop size or smaller than 2, adjust it to m=2")
            m = 2
        self.m = m

        self.debug_mode = debug_mode  # if debug
        self.ndelay = 1  # default
        self.load_pop = load_pop
        self.output_path = out_path

        # Set a random seed
        random.seed(2024)

    # add new individual to population
    def add2pop(self, population, offspring):
        for ind in population:
            if ind['objective'] == offspring['objective']:
                self.logger.info("duplicated result, retrying ... ")
                return False
        population.append(offspring)
        return True

    # run ael 
    def run(self):

        # interface for large language model (llm)
        # interface_llm = PromptLLMs(self.api_endpoint,self.api_key,self.llm_model,self.debug_mode)

        # interface for evaluation
        interface_eval = Evaluation()

        # interface for ec operators
        interface_ec = InterfaceEC(self.pop_size, self.m, self.api_endpoint, self.api_key, self.llm_model,
                                   self.debug_mode, interface_eval, use_local_llm=self.use_local_llm, url=self.url)

        # initialization
        population = []
        if 'use_seed' in self.load_pop and self.load_pop['use_seed']:
            with open(self.load_pop['seed_path']) as file:
                data = json.load(file)
            population = interface_ec.population_generation_seed(data)
            filename = self.output_path + "/ael_results/pops/population_generation_0.json"
            with open(filename, 'w') as f:
                json.dump(population, f, indent=5)
            n_start = 0
        else:
            if self.load_pop['use_pop']:  # load population from files
                self.logger.info("load initial population from " + self.load_pop['pop_path'])
                with open(self.load_pop['pop_path']) as file:
                    data = json.load(file)
                for individual in data:
                    population.append(individual)
                self.logger.info("initial population has been loaded!")
                n_start = self.load_pop['n_pop_initial']
            else:  # create new population
                self.logger.info("creating initial population:")
                population = interface_ec.population_generation()
                population = population_management(population, self.pop_size)
                self.logger.info("initial population has been created!")
                # Save population to a file
                filename = self.output_path + "/ael_results/pops/population_generation_0.json"
                with open(filename, 'w') as f:
                    json.dump(population, f, indent=5)
                n_start = 0

        # main loop
        n_op = len(self.operators)

        for pop in range(n_start, self.n_pop):
            # Perform crossover and mutation
            for na in range(self.pop_size):
                for i in range(n_op):
                    op = self.operators[i]
                    op_w = self.operator_weights[i]
                    if (np.random.rand() < op_w):
                        parents, offspring = interface_ec.get_algorithm(population, op)
                    is_add = self.add2pop(population, offspring)  # Check duplication, and add the new offspring
                    self.logger.info("generate new algorithm using " + op + " with fitness value: ", offspring['objective'])
                    if is_add:
                        data = {}
                        for i in range(len(parents)):
                            data[f"parent{i + 1}"] = parents[i]
                        data["offspring"] = offspring
                        with open(self.output_path + "/ael_results/history/pop_" + str(pop + 1) + "_" + str(
                                na) + "_" + op + ".json", "w") as file:
                            json.dump(data, file, indent=5)

                # populatin management
                size_act = min(len(population), self.pop_size)
                population = population_management(population, size_act)
                self.logger.info(f">> {na + 1} of {self.pop_size} finished ")

            self.logger.info("fitness values of current population: ")
            for i in range(self.pop_size):
                self.logger.info(str(population[i]['objective']) + " ")

            # Save population to a file
            filename = self.output_path + "/ael_results/pops/population_generation_" + str(pop + 1) + ".json"
            with open(filename, 'w') as f:
                json.dump(population, f, indent=5)

            # Save the best one to a file
            filename = self.output_path + "/ael_results/pops_best/population_generation_" + str(pop + 1) + ".json"
            with open(filename, 'w') as f:
                json.dump(population[0], f, indent=5)

            self.logger.info(f">>> {pop + 1} of {self.n_pop} populations finished ")
"""