import random
import os
import sys
import bokeh
from neuron import h, gui
from bioaffrat import bioaffrat
from interneuron import interneuron
from motoneuron import motoneuron
from muscle import muscle
import numpy as np

bokeh.sampledata.download()

from bokeh.plotting import figure, output_file, show

sys.path.append('../')
my_path = os.path.abspath('')
h.load_file('nrngui.hoc')
h.load_file('stdrun.hoc')

themes = 'light_minimal'
nInt = 5
nAff = 12
nMn = 21
N = 5

weight_changes = h.Vector()
time_t = h.Vector()


class CPG:

    def __init__(self):

        h.dt = 0.1
        h.tstop = 100

        self.Ia_aff_E = bioaffrat()
        self.RG_E = interneuron()

        self.generator(self.Ia_aff_E)

        self.connectcells(self.Ia_aff_E, self.RG_E, weight=0.15, stdptype=True)

        h.run()

    def addpool(self, num, name="test", neurontype="int"):

        gids = []
        gid = 0
        if neurontype.lower() == "delay":
            delaytype = True
        else:
            delaytype = False

        if neurontype.lower() == "moto":
            diams = motodiams(num)
        cells = []
        for i in range(num):
            if neurontype.lower() == "moto":
                cell = motoneuron(diams[i])

                self.motoneurons.append(cell)
            elif neurontype.lower() == "aff":
                cell = bioaffrat()
                self.afferents.append(cell)
            elif neurontype.lower() == "muscle":
                cell = muscle()
                self.muscles.append(cell)
            else:
                cell = interneuron()
                self.interneurons.append(cell)
            cells.append(cell)

            # while pc.gid_exists(gid) != 0:
            #     gid += 1
            # gids.append(gid)
            # pc.set_gid2node(gid, rank)
            # nc = cell.connect2target(None)
            # pc.cell(gid, nc)
        return cells

    def connectcells(self, pre, post, weight, delay=1, threshold=10, inhtype=False, stdptype=False, N=30):
        # nsyn = random.randint(N - 15, N)
        nsyn = 5
        # for i in post:
        #     id = random.randint(0, len(pre))
        #     id = 0
        for j in range(nsyn):
            if stdptype:
                nc = h.NetCon(pre.soma(0.5)._ref_v,
                              post.synlistexstdp[j],
                              threshold,
                              delay,
                              weight,
                              sec=pre.soma)
                dummy = h.Section()  # Create a dummy section to put the point processes in
                stdpmech = h.STDP(0, dummy)

                presyn = h.NetCon(pre.soma(0.5)._ref_v,
                                  stdpmech,
                                  threshold,
                                  delay,
                                  2,
                                  sec=pre.soma
                                  )
                pstsyn = h.NetCon(post.soma(1)._ref_v,
                                  stdpmech,
                                  threshold,
                                  delay,
                                  -2,
                                  sec=post.soma
                                  )
                h.setpointer(nc._ref_weight[0], 'synweight', stdpmech)
                stdpmech.verbose = 2
                # Create array to store weight changes

                weight_changes.record(stdpmech._ref_synweight)
                print(stdpmech._ref_synweight)
                time_t.record(h._ref_t)
            else:
                if inhtype:
                    nc = h.NetCon(pre[id].soma(0.5)._ref_v,
                                  post.synlistinh[j],
                                  threshold,
                                  delay,
                                  weight,
                                  sec=pre[id].soma)
                else:
                    nc = h.NetCon(pre[id].soma(0.5)._ref_v,
                                  post.synlistex[j],
                                  threshold,
                                  delay,
                                  weight,
                                  sec=pre[id].soma)

    def generator(self, cell, weight=0.3, delay=1):
        netstim = h.NetStim()
        netstim.number = 30  # Количество генерируемых спайков
        netstim.start = 0  # Время начала генерации спайков
        netstim.interval = 10  # Интервал между спайками (в мс)

        nsyn = 1
        # for cell in cells:
        for j in range(nsyn):
            print('Я в первом цикле')
            nc_es_stim = h.NetCon(netstim, cell.synlistees[j])
            nc_es_stim.delay = delay
            nc_es_stim.weight[0] = weight
            nc_in_stim = h.NetCon(netstim, cell.synlistinh[j])
            nc_in_stim.delay = delay
            nc_in_stim.weight[0] = weight


def our_stim():
    # Настройка симуляции
    h.dt = 0.1
    h.tstop = 100

    threshold = 10  # Set voltage threshold
    delay = 1  # Set connection delay

    cell_la = bioaffrat()
    cell_RG = interneuron()

    netstim = h.NetStim()
    netstim.number = 30  # Количество генерируемых спайков
    netstim.start = 0  # Время начала генерации спайков
    netstim.interval = 10  # Интервал между спайками (в мс)

    nsyn = 5
    for i in range(nsyn):
        print('Я в первом цикле')
        nc_es_stim = h.NetCon(netstim, cell_la.synlistees[i])
        nc_es_stim.delay = delay
        nc_es_stim.weight[0] = 0.3
        nc_in_stim = h.NetCon(netstim, cell_la.synlistinh[i])
        nc_in_stim.delay = delay
        nc_in_stim.weight[0] = 0.3

    # weight_changes_1 = h.Vector()
    # time_t = h.Vector()

    for i in range(nsyn):
        print('Я во втором цикле')
        # cell_la.node[len(cell_la.node) - 1](0.5)._ref_v
        nc = h.NetCon(cell_la.soma(0.5)._ref_v,
                      cell_RG.synlistexstdp[i],
                      threshold,
                      delay,
                      0.15,
                      sec=cell_la.soma)
        dummy = h.Section()  # Create a dummy section to put the point processes in
        stdpmech = h.STDP(0, dummy)
        presyn = h.NetCon(cell_la.soma(0.5)._ref_v,
                          stdpmech,
                          threshold,
                          delay,
                          2,
                          sec=cell_la.soma
                          )
        pstsyn = h.NetCon(cell_RG.soma(1)._ref_v,
                          stdpmech,
                          threshold,
                          delay,
                          -2,
                          sec=cell_RG.soma
                          )
        h.setpointer(nc._ref_weight[0], 'synweight', stdpmech)
        stdpmech.verbose = 2
        # Create array to store weight changes

        weight_changes.record(stdpmech._ref_synweight)
        print(stdpmech._ref_synweight)
        time_t.record(h._ref_t)

        # Run the simulation
    h.run()


def motodiams(number):
    nrn_number = number
    standby_percent = 70
    active_percent = 100 - standby_percent

    standby_size = int(nrn_number * standby_percent / 100)
    active_size = nrn_number - standby_size

    loc_active, scale_active = 27, 3
    loc_stanby, scale_stanby = 44, 4

    x2 = np.concatenate([np.random.normal(loc=loc_active, scale=scale_active, size=active_size),
                         np.random.normal(loc=loc_stanby, scale=scale_stanby, size=standby_size)])

    return x2


def draw():
    # Convert weight_changes to a NumPy array
    weight_changes_array = np.array(weight_changes.as_numpy())
    time_array = np.array(time_t.as_numpy())

    figur = figure(x_axis_label='time (ms)', y_axis_label='weight')
    output_file('change_res.html')
    figur.line(time_array, weight_changes_array, line_width=2)

    show(figur)


if __name__ == '__main__':

    # our_stim()

    cpg = CPG()

    draw()
