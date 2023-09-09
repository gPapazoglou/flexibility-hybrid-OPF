import pandapower as pp
import pandapower.networks as pn
import numpy as np
import pandas as pd
import warnings
from sest2023_example import model_hybrid_opf

# mutes a warning caused by pp.create_sgen
warnings.simplefilter(action='ignore', category=FutureWarning)

def make_random_dn(seed: int):
    # the seed is included to make the generated data pseudorandom and, therefore, replicable
    # the case33bw network is used as the template network, based on which all networks are generated
    net = pn.case33bw()
    net.bus['vn_kv'] = 13.8  # change the voltage
    net.bus.name = f"dn_{seed}"
    net.line['in_service'] = True

    # the default maximum current on lines is 9999 kA. the limits are tightened to create line congestions
    np.random.seed(seed)
    net.line['max_i_ka'] = 0.5 + np.random.random(len(net.line))

    np.random.seed(seed + 100)
    load_variantion_coefficients = np.random.random(len(net.load))
    # change load active and reactive power injections proportionally, to maintain a fixed power factor
    net.load.p_mw = (0.5 + load_variantion_coefficients)*net.load.p_mw
    net.load.q_mvar = (0.5 + load_variantion_coefficients)*net.load.q_mvar
    net.load.name = f"dn_{seed}"

    # add 2 - 6 sgens
    np.random.seed(seed+200)
    number_of_sgens = np.random.choice(list(range(2, 7)))

    np.random.seed(seed+300)
    bus_location = np.random.choice(list(range(1, len(net.bus))), number_of_sgens, replace=False)

    np.random.seed(seed+400)
    injections = 3 - 2*np.random.random(number_of_sgens)

    for bus, inj in zip(bus_location, injections):
        pp.create_sgen(net, bus=bus, p_mw=inj, max_q_mvar=1.2*inj, min_q_mvar=0, name=f"dn_{seed}")

    # choose number (5 to 10) and indexes of flexible loads. all sgens are assumed to be flexible
    np.random.seed(seed+500)
    number_of_flex_loads = np.random.choice((list(range(5, 11))))

    np.random.seed(seed+600)
    flex_load_ids = np.random.choice(net.load.index, number_of_flex_loads, replace=False)

    net.flexibility = pd.DataFrame(columns=['element', 'et', 'cost_up', 'cost_down', 'quantity_up', 'quantity_down'])
    net.flexibility['element'] = list(flex_load_ids) + list(net.sgen.index)
    net.flexibility['et'] = ['load']*number_of_flex_loads + ['sgen']*len(net.sgen)

    np.random.seed(seed+700)
    net.flexibility['cost_up'] = 53 + 5*np.random.rand(len(net.flexibility))
    np.random.seed(seed+800)
    net.flexibility['cost_down'] = 55 - 5*np.random.rand(len(net.flexibility))
    net.flexibility['quantity_up'] = (1.25*net.load.p_mw.loc[flex_load_ids]).to_list() + (1.25*net.sgen.p_mw).to_list()
    net.flexibility['quantity_down'] = net.load.p_mw.loc[flex_load_ids].to_list() + net.sgen.p_mw.to_list()

    return net

def create_merged_net():
    distribution_networks = []
    number_of_dns = 5

    for n in range(number_of_dns):
        dn = make_random_dn(n)
        distribution_networks.append(dn)

    transmission_network = pn.case300()
    transmission_network.bus['name'] = [f"tn_{i}" for i in transmission_network.bus.name]
    transmission_network['flexibility'] = pd.DataFrame(columns=distribution_networks[0].flexibility.columns, data=0)
    dn_connection_candidate_buses = transmission_network.bus[transmission_network.bus['vn_kv'] == 13.8].index
    np.random.seed(1)
    tn_dn_connection_buses = np.random.choice(dn_connection_candidate_buses, number_of_dns)

    for i, dn in enumerate(distribution_networks):
        dn.bus.name = [f'dn_{i} {a}' for a in dn.bus.index]
        transmission_network.bus = transmission_network.bus.append(dn.bus, ignore_index=True)
        # find current dn's indexes
        idxs = transmission_network.bus[transmission_network.bus['name'].str.find(f"dn_{i}") > -1].index
        # create mapping between original DN bus ids and the indexes they have on the TN
        bus_mapping = dict(zip(dn.bus.index, idxs))

        # add line that connects TN and DN
        pp.create_line_from_parameters(transmission_network, from_bus=tn_dn_connection_buses[i], to_bus=bus_mapping[0],
                                       length_km=1, max_i_ka=5, r_ohm_per_km=0.1, x_ohm_per_km=0.05, c_nf_per_km=0)

        dn.line['name'] = [f"dn_{i} {a}" for a in dn.line.index]
        dn.line['from_bus'] = dn.line['from_bus'].apply(lambda x: bus_mapping[x])
        dn.line['to_bus'] = dn.line['to_bus'].apply(lambda x: bus_mapping[x])
        transmission_network.line = transmission_network.line.append(dn.line, ignore_index=True)

        dn.load.name = [f"dn_{i} {a}" for a in dn.load.index]
        dn.load.bus = dn.load.bus.apply(lambda x: bus_mapping[x])
        transmission_network.load = transmission_network.load.append(dn.load, ignore_index=True)
        load_idxs = transmission_network.load[transmission_network.load['name'].str.find(f"dn_{i}") > -1].index
        # create mapping between load ids in the DN and the ones they have in the TN
        load_mapping = dict(zip(dn.load.index, load_idxs))

        dn.sgen.name = [f"dn_{i} {a}" for a in dn.sgen.index]
        dn.sgen.bus = dn.sgen.bus.apply(lambda x: bus_mapping[x])
        transmission_network.sgen = transmission_network.sgen.append(dn.sgen, ignore_index=True)

        sgen_idxs = transmission_network.sgen[transmission_network.sgen['name'].str.find(f"dn_{i}") > -1].index
        # create mapping between sgen ids in the DN and the ones they have in the TN
        sgen_mapping = dict(zip(dn.sgen.index, sgen_idxs))

        flex_loads = dn.flexibility[dn.flexibility['et'] == 'load'].index
        flex_sgen = dn.flexibility[dn.flexibility['et'] == 'sgen'].index

        dn.flexibility['element'].loc[flex_loads] = \
            dn.flexibility['element'].loc[flex_loads].apply(lambda x: load_mapping[x])
        dn.flexibility['element'].loc[flex_sgen] = \
            dn.flexibility['element'].loc[flex_sgen].apply(lambda x: sgen_mapping[x])
        transmission_network.flexibility = transmission_network.flexibility.append(dn.flexibility, ignore_index=True)

    return transmission_network

if __name__ == "__main__":
    net = create_merged_net()
    # m = model_hybrid_opf(net)