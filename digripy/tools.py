# -*- coding: utf-8

import os

import jsonref
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from bokeh.io import output_file
from bokeh.layouts import gridplot
from bokeh.models import DatetimeTickFormatter
from bokeh.palettes import Spectral10
from bokeh.plotting import figure
from bokeh.plotting import show
from tespy.tools.logger import logging


def calc_pipe_attrs(dn_size: int, length: float, sim_settings, in_dir,
                    pipe_definitions_file=None):
    """
    Calculates district heating pipe parameters according to standard
    definitions given in res/pipe_definitions.json.
    Reference: http://www.verenum.ch/Dokumente_QMFW.html
    :param in_dir: project directory. Put pipe_definitions_file here to expand
                   pipe library
    :param sim_settings: simulation settings
    :param dn_size: nominal pipe diameter
    :param length: pipe length [m]
    :param pipe_definitions_file: JSON file that holds the known pipe
           attributes
    :return: {'kA': Heat transfer coefficient multiplied with the transfer
              area, 'd': pipe inner diameter [m]}

    Definitions file expected content example:

    all_types = {"KMR": {20: [0.0216, 0.00265, [.09, 0.11, .125]],
                         25: [0.0285, 0.0026, [.09, .11, .125]],
                         32: [0.0372, 0.0026, [.11, .125, .14]],
                         40: [0.0431, 0.0026, [.11, .125, .14]],
                         50: [0.0545, 0.0029, [.125, .14, .16]],
                         65: [0.0703, 0.0029, [.14, .16, .18]],
                         80: [0.0825, 0.0032, [.16, .18, .2]],
                         100: [0.1071, 0.0036, [.2, .225, .25]],
                         125: [0.1325, 0.0036, [.225, .25, .28]],
                         150: [0.1603, 0.004, [.25, .28, .315]],
                         200: [0.2101, 0.0045, [.315, .355, .4]],
                         250: [0.263, 0.005, [.4, .45, .5]],
                         300: [0.3127, 0.0056, [.45, .5, .58]],
                         350: [0.3444, 0.0056, [.5, .56, .63]],
                         400: [0.3938, 0.0063, [.56, .63, .73]],
                         450: [0.4446, 0.0063, [.63, .67, .8]],
                         500: [0.4954, 0.0063, [.71, .8, .9]],
                         600: [0.5958, 0.0071, [.8, .9, 1]],
                         700: [0.695, 0.008, [.9, 1, 1.1]],
                         800: [0.7954, 0.0088, [1, 1.1, 1.2]],
                         900: [0.894, 0.01, [1.1, 1.2, None]],
                         1000: [0.994, 0.011, [1.2, 1.3, None]]}}
    """

    # lambda_ins: thermal conductivity of the pipe insulation [W/(m*K)]
    # lambda_soil: thermal conductivity of the soil between feed and back
    #              pipes [W/(m*K)]
    # depth: height of the soil above the pipes [m]
    # dist: distance between pipe outer (incl. insulation) diameters [m]
    # ins_level: insulation level as defined in definitions file [-]
    # pipe_type: pipe type name as in definitions file [-]

    lambda_ins = sim_settings['lambda_ins']
    lambda_soil = sim_settings['lambda_soil']
    depth = sim_settings['depth']
    dist = sim_settings['dist']
    ins_level = sim_settings['insulation_level']
    pipe_type = sim_settings['pipe_type']
    # If no pipe definition file is given use the builtin one
    if pipe_definitions_file is None:
        digripy_path = os.path.split(__file__)[0]
        pipe_definitions_file = os.path.join(digripy_path,
                                             "pipe_definitions.json")

    user_pipe_def = os.path.join(in_dir, pipe_definitions_file)
    if os.path.isfile(user_pipe_def):
        pipe_definitions_file = user_pipe_def

    with open(pipe_definitions_file, 'r') as fp:
        all_types = jsonref.load(fp)
    try:
        selected_type = all_types[pipe_type]
        dn_size = str(min(selected_type, key=lambda x: abs(int(x) - dn_size)))
        pipe_attributes = selected_type.get(dn_size)
    except KeyError as e:
        logging.error('Could not find pipe type in definition file! '
                      'Aborting', e)
        raise SystemExit
    except jsonref.JsonRefError as e:
        logging.error('JSON file could not be loaded due to illegal '
                      'structure.', e)
        raise SystemExit

    d = pipe_attributes[0]
    thickness = pipe_attributes[1]
    d_out = pipe_attributes[2][ins_level - 1]
    if d_out is None:
        logging.error('No data for that insulation level!')
        raise SystemExit

    r_r = d / 2 + thickness
    r_m = d_out / 2

    u_r = 1 / (r_r / lambda_ins * np.log(r_m / r_r) +
               r_r / lambda_soil * np.log(4 * (depth + r_m) / r_m) +
               r_r / lambda_soil * np.log(
                (((2 * (depth + r_m) / (dist + 2 * r_m)) ** 2) + 1) ** .5))
    return {'kA': u_r * np.pi * length * d, 'D': d, 'u_r': u_r}


def read_demands_file(input_folder: str, file_name: str, first_frame: int = 0,
                      last_frame: int = 0):
    """
    Reads a CSV file containing the consumer demand time series.
    It expects a column time with datetime format like '01/01/17 01:00 AM' or
    similar with days first.
    The other columns shall reflect the pipe id matching the house connection
    as defined in network.csv.

    :param input_folder: Folder to find the demands time series file in.
    :param file_name: Demands time series file name.
    :param first_frame: Index fo the first frame to be read.
    :param last_frame: Index of the last frame to be read. Set to 0 to read
                       all.
    :return: Returns a Pandas DataFrame with each consumer's demands over time.
    """
    logging.debug(f'Reading file {file_name}.')
    try:
        d = pd.read_csv(os.path.join(input_folder, file_name),
                        parse_dates=['time'], index_col=['time'],
                        dayfirst=False)
        if last_frame == 0:
            last_frame = d.size
        d = d[first_frame:last_frame]
        logging.debug(f'File {file_name} read successfully.')
        return d
    except FileNotFoundError:
        logging.debug(f'File {file_name} could not be found.')
        return None


def read_settings_file(input_folder: str, file_name: str):
    """
    Reads a settings file that defines several settings of the district heating
    network and the simulation process. As multiple settings can be run
    sequentially and paralleled each line in the csv can represent a setting.
    Each of these settings must contain values as defined in the file
    settings_howto.csv.

    :param input_folder: Folder to find the settings file in.
    :param file_name: Settings file name.
    :return: Returns a Pandas DataFrame with all settings. See
             settings_howto.csv for what is expected.
    """
    logging.debug(f'Reading file {file_name}.')
    try:
        settings = pd.read_csv(os.path.join(input_folder, file_name),
                               index_col='name')
        settings = settings[settings['active'] == 1]
        logging.debug(f'File {file_name} read successfully.')
        return settings.shape[0], settings
    except FileNotFoundError:
        logging.debug(f'File {file_name} could not be found.')
        return None


def read_network_definitions_file(input_folder, file_name):
    """
    Reads a file that defines the district heating network's structure.
    See network_howto.csv for information on how it shall be structured.

    :param input_folder: Folder to find the file in.
    :param file_name: Network structure file name.
    :return: Returns a Pandas DataFrame.
    """
    logging.debug(f'Reading file {file_name}.')
    try:
        return pd.read_csv(os.path.join(input_folder, file_name),
                           index_col='pipe_ID')
    except FileNotFoundError:
        logging.debug(f'File {file_name} could not be found.')
        return None


def draw_graph(network):
    """
    Draws a simple directed graph from a tespy network.
    :param network: A tespy network.
    """
    plt.figure(1, figsize=(20, 20))
    gdf = pd.DataFrame(columns=['s', 't'])
    cats = pd.DataFrame(columns=['cat'])
    conns = network.conns
    gdf['source'] = [o.label for o in conns['source']]
    gdf['target'] = [o.label for o in conns['target']]
    g = nx.convert_matrix.from_pandas_edgelist(conns, 'source', 'target',
                                               create_using=nx.DiGraph())
    cats['cat'] = [o.component() for o in g.nodes]
    g = nx.convert_matrix.from_pandas_edgelist(gdf, 'source', 'target',
                                               create_using=nx.DiGraph())
    cats['cat'] = pd.Categorical(cats['cat'])
    nx.draw(g, with_labels=True, alpha=0.3, arrows=True,
            node_color=cats['cat'].cat.codes, cmap=plt.cm.Set1, node_size=500,
            pos=nx.nx_agraph.graphviz_layout(g))
    plt.show()


def export_results(process_results, references, output_dir,
                   results_file_prefix=""):
    """
    Exports an xlsx file with results over all run simulations and an html file
    showing plots of the results.
    The Plot shows the source demand of all settings. If a reference time
    series has been set there will also be plots showing relative and absolute
    deviations to the references data where it is given.
    :param process_results: The results mapped over the simulations.
    :param references: A reference file
    :param output_dir: Where shall the result files be saved?
    :param results_file_prefix: A prefix added to all result files.
    """

    res_info = pd.DataFrame()
    res_values = pd.DataFrame()
    no_references = False
    filename = f'{output_dir + os.path.sep + results_file_prefix}'

    for p in process_results:
        res_info = res_info.append(p[0])
        res_values.loc[:, p[0].name] = p[1].loc[:, 'Demand Heat Source']

    with pd.ExcelWriter(f'{filename}_results.xlsx') as xlsx:
        pd.DataFrame(res_info).to_excel(xlsx, index=False, sheet_name='Info')
        if references is not None and not no_references:
            common_cols = res_values.columns.intersection(references.columns)
            if not common_cols.empty:
                res_rel_dev_percent = (res_values.loc[:, common_cols].div(
                    references.loc[:, common_cols], axis=1) - 1) * 100
                res_abs_dev = res_values.loc[:, common_cols] - \
                              references.loc[:, common_cols]
                pd.DataFrame(res_rel_dev_percent). \
                    to_excel(xlsx, index=False, sheet_name='Rel_dev')
                pd.DataFrame(res_abs_dev). \
                    to_excel(xlsx, index=False, sheet_name='Abs_dev')
        else:
            no_references = True

    # Use Bokeh to print plots and export as html file
    plots = []
    line_width = 1.5
    output_file(f'{filename}_results.html')

    # Plot absolute deviations and actual demand
    plot = figure(width=1600, height=500, x_axis_type='datetime',
                  title='Resulting heat source demands')

    time = res_values.index.values
    for n in res_values:
        i = res_values.columns.get_loc(n)
        if not no_references and n in res_abs_dev:
            plot.line(x=time, y=res_abs_dev.loc[:, n],
                      legend_label=n + ' deviation to given reference [W]',
                      line_color=Spectral10[i],
                      line_width=line_width, line_alpha=0.9)
        plot.line(x=time, y=res_values.loc[:, n], legend_label=n + ' [W]',
                  line_color=Spectral10[i], line_width=line_width,
                  line_alpha=0.9)
    plot.xaxis.formatter = DatetimeTickFormatter(days=['%B %d'],
                                                 hours=['%H:%M'])
    plot.legend.location = "top_right"
    plot.legend.click_policy = "hide"

    plots.append(plot)

    # Plot relative deviations
    if not no_references:
        plot = figure(width=1600, height=300, x_axis_type='datetime',
                      title='Results compared to given references in percent')
        for n in res_rel_dev_percent:
            i = res_values.columns.get_loc(n)
            plot.line(x=time, y=res_rel_dev_percent.loc[:, n],
                      legend_label=n + ' [%]', line_color=Spectral10[i],
                      line_width=line_width, line_alpha=0.9)
        plot.xaxis.formatter = DatetimeTickFormatter(days=['%B %d'],
                                                     hours=['%H:%M'])
        plot.legend.location = "top_right"
        plot.legend.click_policy = "hide"

        plots.append(plot)

    show(gridplot(children=plots, sizing_mode='stretch_width', ncols=1))
