import pandapower as pp
import pandapower.networks as pn
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import pyomo.environ as pyo
import numpy as np
from pandapower.pypower import makePTDF
from pandapower.converter.pypower import to_ppc
from pandapower.results import init_results
from pandapower.pf.makeYbus_numba import makeYbus
import pandapower.plotting as plot
import pandapower.networks as nw

def transmission_network():
    net = pn.case5()

    net.sgen.drop(net.sgen.index, inplace=True)
    net.load.drop(net.load.index, inplace=True)
    net.gen.drop(net.gen.index, inplace=True)
    net.ext_grid.drop(net.ext_grid.index, inplace=True)

    pp.create_load(net, 3, 106)
    pp.create_load(net, 2, 7, name="DN1")
    pp.create_load(net, 4, 7, name="DN2")

    pp.create_gen(net, 0, 70)
    pp.create_gen(net, 1, 50)
    pp.create_ext_grid(net, 0)

    np.random.seed(1)
    net.line['max_i_ka'].loc[1:4] = 0.1 + 10 * np.random.random(4)  # tighten thermal limits on these lines

    net.flexibility = pd.DataFrame(columns=['bus', 'up_quantity', 'down_quantity', 'up_cost', 'down_cost'])
    # net.flexibility['bus'] = [0, 1, 3]
    # net.flexibility['up_quantity'] = [10, 12, 18]
    # net.flexibility['down_quantity'] = [20, 15, 14]
    # net.flexibility['up_cost'] = [101, 110, 120]
    # net.flexibility['down_cost'] = [95, 99, 80]
    net.flexibility['bus'] = [0, 1]
    net.flexibility['up_quantity'] = [10, 12]
    net.flexibility['down_quantity'] = [20, 15]
    net.flexibility['up_cost'] = [101, 110]
    net.flexibility['down_cost'] = [95, 99]

    pp.rundcpp(net)

    return net

# net_tn = nw.mv_oberrhein()
# plot.simple_plot(net_tn, show_plot=True)
def create_dn_1():
    net = pp.create_empty_network()

    for _ in range(5):
        pp.create_bus(net, 10, max_vm_pu=1.05, min_vm_pu=0.95)

    from_bus = [0, 1, 2, 2]
    to_bus = [1, 2, 3, 4]
    types = ['149-AL1/24-ST1A 10.0', '149-AL1/24-ST1A 10.0', '48-AL1/8-ST1A 10.0', '48-AL1/8-ST1A 10.0']

    for f, t, type in zip(from_bus, to_bus, types):
        pp.create_line(net, f, t, 1, std_type=type, max_loading_percent=100)

    net.line['max_i_ka'].loc[0:1] = [0.7, 0.7]
    pp.create_load(net, 1, p_mw=5, q_mvar=2.5)  # pf: 0.844, default values: p=5, q=2.5
    pp.create_load(net, 3, p_mw=1.5, q_mvar=0.75)  # pf: 0.844, default values: p=1.5, q=0.75
    pp.create_load(net, 4, p_mw=3.5, q_mvar=1.5)  # pf: 0.9191, ratio = 0.42857, , default values: p=3.5, q=1.5

    # pp.create_sgen(net, 0, p_mw=0)
    pp.create_sgen(net, 2, p_mw=3, q_mvar=3, max_q_mvar=4, min_q_mvar=0)  # default values: p=3, q=3

    pp.create_ext_grid(net, 0)

    pp.runpp(net)

    net.flexibility = pd.DataFrame(columns=['bus', 'up_quantity', 'down_quantity', 'up_cost', 'down_cost'])
    net.flexibility['bus'] = [2, 4]
    net.flexibility['up_quantity'] = [1, 1]
    net.flexibility['down_quantity'] = [1, 1.5]
    net.flexibility['up_cost'] = [101, 110]
    net.flexibility['down_cost'] = [95, 99]

    return net


def create_dn_2():
    net = pp.create_empty_network()

    for _ in range(5):
        pp.create_bus(net, 10, max_vm_pu=1.05, min_vm_pu=0.95)

    from_bus = [0, 1, 2, 2, 3]
    to_bus = [1, 2, 3, 4, 4]
    types = ['149-AL1/24-ST1A 10.0', '48-AL1/8-ST1A 10.0', '48-AL1/8-ST1A 10.0',
             '48-AL1/8-ST1A 10.0', '149-AL1/24-ST1A 10.0']

    for f, t, type in zip(from_bus, to_bus, types):
        pp.create_line(net, f, t, 1, std_type=type, max_loading_percent=100)

    net.line['max_i_ka'].loc[0:1] = [0.7, 0.2]
    pp.create_load(net, 1, p_mw=3.5, q_mvar=1.75)  # default p = 3.5, q=1.75
    pp.create_load(net, 2, p_mw=4, q_mvar=2)  # default p = 4, q=2
    # pp.create_load(net, 3, p_mw=2)
    pp.create_load(net, 4, p_mw=3.5, q_mvar=1.75)  # default p = 3.5, q=1.75
    pp.create_sgen(net, 3, p_mw=4, q_mvar=4, max_q_mvar=5, min_q_mvar=0)  # default p = 4, q=4

    pp.create_ext_grid(net, 0)

    net.flexibility = pd.DataFrame(columns=['bus', 'up_quantity', 'down_quantity', 'up_cost', 'down_cost'])
    net.flexibility['bus'] = [1, 2]
    net.flexibility['up_quantity'] = [1, 1]
    net.flexibility['down_quantity'] = [1, 1.5]
    net.flexibility['up_cost'] = [102, 109]
    net.flexibility['down_cost'] = [94, 98]

    # w/ sharing
    # net.flexibility['bus'] = [0, 1, 2]
    # net.flexibility['up_quantity'] = [100, 1, 1]
    # net.flexibility['down_quantity'] = [100, 1, 1.5]
    # net.flexibility['up_cost'] = [3000, 101, 110]
    # net.flexibility['down_cost'] = [-500, 95, 99]

    pp.runpp(net)

    return net


def merged_nets():
    tn = transmission_network()
    # drop the loads that represent the dns
    load_ids_to_drop = [1, 2]
    tn.load.drop(load_ids_to_drop, inplace=True)
    dn1 = create_dn_1()
    dn2 = create_dn_2()

    dns = [dn1, dn2]
    tn.bus.name = [f"tn {a}" for a in tn.bus.index]
    tn.line.name = [f"tn {a}" for a in tn.line.index]

    # node of the TN each DN is connected to
    connection_to_tn = [2, 4]

    for i, dn in enumerate(dns):
        dn.bus.name = [f'dn_{i} {a}' for a in dn.bus.index]
        tn.bus = tn.bus.append(dn.bus, ignore_index=True)
        # find current dn's indexes
        idxs = tn.bus[tn.bus['name'].str.find(f"dn_{i}") > -1].index
        bus_mapping = dict(zip(dn.bus.index, idxs))

        # add trafos
        pp.create_transformer(tn, hv_bus=connection_to_tn[i], lv_bus=bus_mapping[0], std_type="63 MVA 110/10 kV")

        dn.line['name'] = [f"dn_{i} {a}" for a in dn.line.index]
        dn.line['from_bus'] = dn.line['from_bus'].apply(lambda x: bus_mapping[x])
        dn.line['to_bus'] = dn.line['to_bus'].apply(lambda x: bus_mapping[x])
        tn.line = tn.line.append(dn.line, ignore_index=True)

        dn.load.name = [f"dn_{i} {a}" for a in dn.load.index]
        dn.load.bus = dn.load.bus.apply(lambda x: bus_mapping[x])
        tn.load = tn.load.append(dn.load, ignore_index=True)

        dn.sgen.name = [f"dn_{i} {a}" for a in dn.sgen.index]
        dn.sgen.bus = dn.sgen.bus.apply(lambda x: bus_mapping[x])
        tn.sgen = tn.sgen.append(dn.sgen, ignore_index=True)

        dn.flexibility.bus = dn.flexibility.bus.apply(lambda x: bus_mapping[x])
        tn.flexibility = tn.flexibility.append(dn.flexibility, ignore_index=True)

    tn.trafo['vn_hv_kv'] = 230

    return tn


if __name__ == '__main__':
    # this is the network used for the test case:
    test_case_network = merged_nets()
