# -*- coding: utf-8

import logging
import numpy as np
import pandas as pd
from tespy.components import Valve, Merge, Splitter, CycleCloser, Pipe, Pump, \
    HeatExchangerSimple
from tespy.connections import Connection
from tespy.networks import Network as TespyNetwork

from digripy.tools import calc_pipe_attrs


class DigripyNetwork:
    def __init__(self, pipe_data, sim_settings, in_dir):
        self._in_dir = in_dir
        self._couplings = sim_settings['simulate_couplings']
        self._t_amb = sim_settings['t_amb']
        self._pipes = pipe_data
        self._sim_settings = sim_settings

        logging.debug('Loaded Data: \n\n' + str(self._pipes))

        self.nw = TespyNetwork(fluids=['BICUBIC::water'], T_unit='C',
                               p_unit='bar', memorise_fluid_properties=False)
        self.nw.set_attr(p_range=[1, 10])
        # Suppress the generation of the residual table for each iteration
        self.nw.iterinfo = False

        self._closer = CycleCloser('closer')
        self._pump = Pump('pump')
        self._pump.set_attr(eta_s=1)
        self._pipeObjects = []
        self._consumers = pd.DataFrame(columns=['object', 'name',
                                                'dist_from_source', 'in_con',
                                                'out_con', 'valve'])

        self._heat_source = HeatExchangerSimple(label='Heat source', pr=.99,
                                                Q='var')
        self._heat_source_con_out = None

        self._add_pipe_to_network([], [], 0, 0)

        self._pump_con_in = Connection(self._closer, 'out1', self._pump, 'in1',
                                       state='l', fluid={'water': 1})
        self._pump_con_out = self._heat_source_con_in = \
            Connection(self._pump, 'out1', self._heat_source, 'in1', state='l')

        self.nw.add_conns(self._pump_con_in, self._pump_con_out)
        self.nw.check_network()
        for c in self.nw.comps['object']:
            if isinstance(c, Pipe):
                c.get_attr('zeta').set_attr(max_val=1e300)

    # Recursively build a Tespy network from the data imported
    def _add_pipe_to_network(self, parent_out, parent_in, cmp_id,
                             dist_from_source):
        children = pd.DataFrame(self._pipes.loc[self._pipes['prior_ID'] ==
                                                cmp_id])
        num_children = children.shape[0]
        if cmp_id == 0:
            self._add_pipe_to_network([self._heat_source, 'out1'],
                                      [self._closer, 'in1'],
                                      children.index.values[0],
                                      dist_from_source)
        else:
            pipe = self._pipes.loc[cmp_id]
            pipe_length = pipe['Shape_Length']
            is_first = pipe['prior_ID'] not in self._pipes.index

            # Call method in tools file to compute pipe parameters
            pipe_attrs = calc_pipe_attrs(dn_size=pipe['DN'],
                                         length=pipe_length,
                                         sim_settings=self._sim_settings,
                                         in_dir=self._in_dir)
            # Create pipe components for supply and return pipe
            dirs = ['supply', 'return']
            (pf, pb) = (Pipe(label=f'pipe {d} {cmp_id}',
                        L=pipe_length,
                        kA=pipe_attrs['kA'],
                        D=pipe_attrs['D'],
                        Tamb=self._t_amb,
                        hydro_group='HW',
                        ks=100) for d in dirs)

            for p in (pf, pb):
                p.get_attr('ks').set_attr(max_val=100)
                # Value set only as a start value. Then set as variable
                p.get_attr('Q').set_attr(val=-50)
                p.get_attr('Q').set_attr(max_val=0)
                p.get_attr('Q').set_attr(is_set=True)
                p.get_attr('Q').set_attr(is_var=True)
                self._pipeObjects.append(p)

            d = pipe_attrs['D']

            dist_from_source += pipe_length

            logging.debug(f'Created components {pf.get_attr("label")} and '
                          f'{pb.get_attr("label")}.')

            supply_in = Connection(parent_out[0], parent_out[1], pf, 'in1',
                                   state='l', printout=True)
            supply_out = Connection(pb, 'out1', parent_in[0], parent_in[1],
                                    state='l', printout=True)
            if is_first:
                self._heat_source_con_out = supply_in
                self._heat_source_con_out.set_attr(printout=True)
                self._heat_source.set_attr(D=d)
            self.nw.add_conns(supply_in, supply_out)

            # Create consumer if pipe is a house connector
            if num_children == 0:
                consumer = HeatExchangerSimple(label=f'House {cmp_id}',
                                               pr=.98, D=d)
                consumer_ctrl = Valve(label=f'Control Valve {cmp_id}')
                consumer_in = Connection(pf, 'out1', consumer, 'in1',
                                         state='l')
                consumer_out = Connection(consumer, 'out1', consumer_ctrl,
                                          'in1', state='l')
                ctrl_out = Connection(consumer_ctrl, 'out1', pb, 'in1',
                                      state='l')
                self._consumers.loc[cmp_id] = {
                    'object': consumer,
                    'name': consumer.label,
                    'dist_from_source': dist_from_source,
                    'in_con': consumer_in,
                    'out_con': consumer_out,
                    'valve': consumer_ctrl}
                self.nw.add_conns(consumer_in, consumer_out, ctrl_out)
                logging.debug(f'Created consumer {consumer.get_attr("label")}')

            # If there is only one connection to another pipe
            elif num_children == 1:
                # Recursively call this method again if we don't want couplings
                # or if the connection to the next pipe is straight
                if not self._couplings or children['passage'].iloc[0] == 0:
                    self._add_pipe_to_network([pf, 'out1'], [pb, 'in1'],
                                              children.index.values[0],
                                              dist_from_source)
                # Else integrate valves to simulate the flow losses of a
                # coupling
                else:
                    valve_in = Valve(label=f'90 deg coupling at output of '
                                           f'pipe {cmp_id}',
                                     zeta=2 / (d ** 4))
                    valve_out = Valve(label=f'90 deg coupling at input of '
                                            f'pipe {cmp_id}',
                                      zeta=2 / (d ** 4))
                    con_valve_in = Connection(pf, 'out1', valve_in, 'in1',
                                              state='l', printout=True)
                    con_valve_out = Connection(valve_out, 'out1', pb, 'in1',
                                               state='l', printout=True)
                    self.nw.add_conns(con_valve_out, con_valve_in)
                    self._add_pipe_to_network([valve_in, 'out1'],
                                              [valve_out, 'in1'],
                                              children.index.values[0],
                                              dist_from_source)

            # Create splitters and mergers if multiple connecting pipes are
            # present
            else:
                split = Splitter(label=f'Splitter at {cmp_id}',
                                 num_out=num_children)
                splitter_in = Connection(pf, 'out1', split, 'in1', state='l')
                merger = Merge(label=f'Merger at {cmp_id}',
                               num_in=num_children)
                merge_out = Connection(merger, 'out1', pb, 'in1', state='l')

                self.nw.add_conns(splitter_in, merge_out)
                for i in range(num_children):
                    child = children.astype(int).iloc[i]
                    if self._couplings:
                        is_straight = child['passage'] == 0
                        valve_out = Valve(label=('Straight ' if is_straight
                                                 else '90 deg ') + 'coupling' +
                                                f'{i} at splitter output '
                                                f'{cmp_id}',
                                          zeta=(0.2 if is_straight else 2) /
                                               (d ** 4))
                        valve_in = Valve(label=('Straight ' if is_straight else
                                                '90 deg ') + f'coupling {i} at'
                                                f' merger input {cmp_id}',
                                         zeta=(1 if is_straight else 1.5) /
                                              (d ** 4))
                        con_valve_out = Connection(split, 'out' + str(i + 1),
                                                   valve_out, 'in1', state='l',
                                                   printout=False)
                        con_valve_in = Connection(valve_in, 'out1', merger,
                                                  'in' + str(i + 1), state='l',
                                                  printout=False)
                        self.nw.add_conns(con_valve_out, con_valve_in)
                        self._add_pipe_to_network([valve_out, 'out1'],
                                                  [valve_in, 'in1'],
                                                  child.name,
                                                  dist_from_source)
                    else:
                        self._add_pipe_to_network([split, 'out' + str(i + 1)],
                                                  [merger, 'in' + str(i + 1)],
                                                  child.name,
                                                  dist_from_source)

    def set_src_t_out(self, temp):
        self._heat_source_con_out.set_attr(T=temp)

    def get_src_t_out(self):
        return self._heat_source_con_out.get_attr('T').get_attr('val')

    def get_src_t_in(self):
        return self._heat_source_con_in.get_attr('T').get_attr('val')
    
    def set_pump_p_out(self, p):
        self._pump_con_out.set_attr(p=p)
        
    def get_pump_p_out(self):
        return self._pump_con_out.get_attr('p').get_attr('val')

    def get_pump_p_in(self):
        return self._pump_con_in.get_attr('p').get_attr('val')

    def get_p_drop(self):
        return self.get_pump_p_out() - self.get_pump_p_in()

    def get_src_flow_speed(self):
        return self._heat_source_con_out.v.val / \
               ((self._heat_source.get_attr('D').get_attr('val') / 2) ** 2) \
               / np.pi

    def get_pump_power(self):
        return self._pump.get_attr('P').get_attr('val')

    def get_src_heat_demand(self):
        return self._heat_source.get_attr('Q').get_attr('val')

    def get_cons_df(self):
        return self._consumers

    def get_total_pipe_heat_losses(self):
        heat_losses = 0
        for pipe in self._pipeObjects:
            loss = pipe.Q.val
            if loss >= 0:
                msg = f'Invalid loss values. Pipe loss is {-loss}W ' \
                      f'at {pipe.label}!'
                logging.error(msg)
                raise ValueError(msg)
            heat_losses += loss
        return heat_losses

    def set_cons_demands(self, demands, t_out: float):
        fix_pr_is_set = False
        max_demand = demands.max()
        for c in self._consumers.itertuples():
            demand = demands[str(c.Index)]
            c.object.set_attr(Q=-demand)
            c.out_con.set_attr(T=t_out)
            # For one consumer with highest current demand a fix pressure
            # ratio is set such that the pressure ratios for
            # the other consumers can be derived.
            if demand == max_demand and not fix_pr_is_set:
                c.valve.set_attr(
                    pr=self._sim_settings['pr_at_largest_consumer'])
                fix_pr_is_set = True
            else:
                c.valve.set_attr(pr=None)

    # Resets the network. This is intended to be used after a failed sub-frame
    # has corrupted the component and connection values in a way that prevents
    # further calculations from being successful.
    def reset(self):
        self.__init__(pipe_data=self._pipes, sim_settings=self._sim_settings,
                      in_dir=self._in_dir)
