# -*- coding: utf-8

import argparse
import datetime
import logging
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import tespy

from digripy import tools
from digripy.simulation import Simulation

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--input_dir", required=False, type=str,
                default='../example40C',	help="Directory with input files")
ap.add_argument("-nm", "--no_mp", required=False, type=bool, default=False,
                help="Disable multi-processing")
ap.add_argument("-nc", "--no_cuda", required=False, type=bool, default=False,
                help="Disable CUDA")
ap.add_argument("-f", "--first_frame", required=False, type=int, default=0,
                help="First frame to start calculation from. Default: 0")
ap.add_argument("-l", "--last_frame", required=False, type=int, default=0,
                help="Last frame to calculate. 0 for last. Default: 0")

args = vars(ap.parse_args())
input_dir = args['input_dir']
no_mp = args['no_mp']
no_cuda = args['no_cuda']
first_frame = args['first_frame']
last_frame = args['last_frame']


def sim_process(setting):
    """
    Starts a simulation subprocess.
    :param setting: pandas dataframe holding simulation parameters
    """
    net_settings = setting[1]
    run_name = f'{settings.index.get_loc(setting[0])}: {setting[0]}'

    sim = Simulation(pipe_data=pipe_data, sim_settings=net_settings,
                     file_prefix=result_file_prefix, first_frame=first_frame,
                     display_graph=False, run_name=run_name,
                     output_dir=output_dir, input_dir=input_dir,
                     demands=demands,
                     min_demand=net_settings['min_consumer_demand'],
                     log_file=log_file)
    return sim.run_process()


if __name__ == '__main__':
    start_time = datetime.datetime.now()

    # First check if we are running in debug mode
    is_debug = sys.gettrace() is not None
    if is_debug:
        logging.warning('Running in debug mode. No multi-processing.')
    if no_mp:
        logging.warning('Disabled multi-processing.')
    use_multi_processing = not no_mp and not is_debug

    network_definitions_file_name = 'network.csv'

    project_name = os.path.basename(input_dir)
    result_file_prefix = f'{datetime.datetime.now().strftime("%H-%M-%S")}_' \
                         f'{project_name}'

    output_dir = f'{Path.home()}/digripy-results/' \
                 f'{datetime.date.today().isoformat()}/'
    os.makedirs(output_dir, exist_ok=True)

    log_file = f'{result_file_prefix}.log'
    tespy.logger.define_logging(log_path=True, log_version=True,
                                logpath=output_dir, logfile=log_file,
                                screen_level=logging.ERROR,
                                file_level=logging.ERROR,
                                file_format=f'{__file__} - %(asctime)s-'
                                            f'%(levelname)s-%(message)s',
                                screen_format=f'{__file__} - %(asctime)s-'
                                              f'%(levelname)s-%(message)s')

    settings_file_name = 'settings.csv'
    reference_file_name = 'references.csv'
    time_series_file_name = 'demands.csv'

    demands = tools.read_demands_file(input_dir, time_series_file_name,
                                      first_frame=first_frame,
                                      last_frame=last_frame)
    pipe_data = \
        tools.read_network_definitions_file(input_dir,
                                            network_definitions_file_name)
    num_settings, settings = tools.read_settings_file(input_dir,
                                                      settings_file_name)

    if no_cuda:
        settings['no_cuda'] = True
        logging.warning('Disabled CUDA.')

    total_frames = len(demands.index)

    references = tools.read_demands_file(input_dir, reference_file_name,
                                         first_frame, last_frame)

    process_results = []
    try:
        # Use multiprocessing only if not in debug mode and if >2 cores are
        # available
        # Number of cores to be used (reduced by 1 to keep system responsive)
        cores = int(np.ceil(num_settings /
                            np.ceil(num_settings / (mp.cpu_count() - 1))))
        if use_multi_processing and cores > 1:
            logging.info("Using multi-processing.")
            with mp.Pool(cores) as p:
                process_results = p.map(sim_process, settings.iterrows())
        else:
            logging.info("Running single process.")
            process_results = map(sim_process, settings.iterrows())
    except Exception as e:
        logging.error(e)
    finally:
        tools.export_results(process_results, references, output_dir,
                             result_file_prefix)

        end_time = datetime.datetime.now()
        runtime = end_time - start_time
        logging.critical(f'Simulation runtime: {runtime}')
        logging.critical(f'Runtime per setting: {runtime / len(settings)}')
