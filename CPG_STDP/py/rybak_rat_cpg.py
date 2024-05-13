import logging

logging.basicConfig(filename='logs.log',
                    filemode='w',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
logging.info("let's get it started")
import numpy as np
from neuron import h
from neuron.units import ms, mV
import h5py as hdf5

h.load_file('nrngui.hoc')

# paralleling NEURON stuff
pc = h.ParallelContext()
rank = int(pc.id())
nhost = int(pc.nhost())

# param
speed = 50  # duration of layer 25 = 21 cm/s; 50 = 15 cm/s; 125 = 6 cm/s
# 100 Hz is the motor cortex frequency
bs_fr = 100  # 40 # frequency of brainstem inputs
versions = 1

# step_number = 5  # number of steps

CV_number = 6
nMN = 21  # 21 # 210 # Number of motor neurons
nAff = 12  # 120 # Number of afferents
nInt = 5 # 19 # 196 # Number of neurons in interneuronal pools
N = 5  # 5 #50

k = 0.017  # CV weights multiplier to take into account air and toe stepping
CV_0_len = 12  # 125 # Duration of the CV generator with no sensory inputs
extra_layers = 0  # 1 + layers


step_number = 5


one_step_time = int((6 * speed + CV_0_len) / (int(1000 / bs_fr))) * (int(1000 / bs_fr))
#time_sim = 100 + one_step_time * step_number
time_sim = 600

exnclist = []
inhnclist = []
exstdpnclist = []
inhstdpnclist = []
eesnclist = []
stimnclist = []
exstdpmexlist = []
inhstdpmexlist = []

# Создание векторов для отслеживания deltaw и synweight
time_vector = h.Vector()
deltaw_vector = h.Vector()
synweight_vector = h.Vector()

from interneuron import interneuron
from motoneuron import motoneuron
from bioaffrat import bioaffrat
from muscle import muscle

import random

'''
network topology https://github.com/max-talanov/bypass/blob/main/figs/CPG_feedback_loops.png
'''

class CPG_L:
    def __init__(self, speed, bs_fr, inh_p, step_number, N):

        self.interneurons = []
        self.motoneurons = []
        self.muscles = []
        self.afferents = []
        self.stims = []
        self.ncell = N
        self.groups = []
        self.motogroups = []
        self.musclegroups = []
        self.affgroups = []
        self.RG_E = []  # Rhythm generators of extensors
        self.RG_F = []  # Rhythm generators of flexor
        self.V3F = []
        self.V0d = []
        self.V2a = []
        self.C_1 = []
        self.C_0 = []
        self.V0v = []



        for layer in range(CV_number):
            '''cut and muscle feedback'''
            ## self.dict_CV = {layer: 'CV{}'.format(layer + 1)}
            self.dict_CV_1 = {layer: 'CV{}_1'.format(layer + 1)}
            self.dict_RG_E = {layer: 'RG{}_E'.format(layer + 1)}
            self.dict_RG_F = {layer: 'RG{}_F'.format(layer + 1)}
            self.dict_V3F = {layer: 'V3{}_F'.format(layer + 1)}
            self.dict_V2a = {layer: 'V2a{}_F'.format(layer + 1)}
            self.dict_V0v = {layer: 'V0v{}_F'.format(layer + 1)}

            self.dict_C = {layer: 'C{}'.format(layer + 1)}

        for layer in range(CV_number):
            '''Cutaneous pools'''
            ##self.dict_CV[layer] = self.addpool(self.ncell, "CV" + str(layer + 1), "aff")
            self.dict_CV_1[layer] = self.addpool(self.ncell, "CV" + str(layer + 1) + "_1", "aff")

            '''Rhythm generator pools'''
            self.dict_RG_E[layer] = self.addpool(self.ncell, "RG" + str(layer + 1) + "_E", "int")
            self.dict_RG_F[layer] = self.addpool(self.ncell, "RG" + str(layer + 1) + "_F", "int")
            self.RG_E.append(self.dict_RG_E[layer])
            self.RG_F.append(self.dict_RG_F[layer])

        '''RG'''
        self.RG_E = sum(self.RG_E, [])
        self.InE = self.addpool(nInt, "InE", "int")
        self.RG_F = sum(self.RG_F, [])
        self.InF = self.addpool(nInt, "InF", "int")
        self.In1 = self.addpool(nInt, "In1", "int")

        '''sensory and muscle afferents and brainstem and V3F'''
        ## self.sens_aff = self.addpool(nAff, "sens_aff", "aff")
        self.Ia_aff_E = self.addpool(nAff, "Ia_aff_E", "aff")
        self.Ia_aff_F = self.addpool(nAff, "Ia_aff_F", "aff")
        self.BS_aff_E = self.addpool(nAff, "BS_aff_E", "aff")
        self.BS_aff_F = self.addpool(nAff, "BS_aff_F", "aff")
        self.V3F = self.addpool(nInt, "V3F", "int")
        self.V0v = self.addpool(nInt, "V0v", "int")
        self.V2a = self.addpool(nInt, "V2a", "int")
        self.V0d = self.addpool(nInt, "V0d", "int")


        '''moto neuron pools'''
        self.mns_E = self.addpool(nMN, "mns_E", "moto")
        self.mns_F = self.addpool(nMN, "mns_F", "moto")

        '''muscles'''
        self.muscle_E = self.addpool(nMN * 30, "muscle_E", "muscle")
        self.muscle_F = self.addpool(nMN * 20, "muscle_F", "muscle")

        '''reflex arc'''
        self.Ia_E = self.addpool(nInt, "Ia_E", "int")
        self.R_E = self.addpool(nInt, "R_E", "int")  # Renshaw cells
        self.Ia_F = self.addpool(nInt, "Ia_F", "int")
        self.R_F = self.addpool(nInt, "R_F", "int")  # Renshaw cells
        # self.Iagener_E = []
        # self.Iagener_F = []

        '''BS'''
        # periodic stimulation
        self.E_bs_gids, self.F_bs_gids = self.add_bs_geners(bs_fr, 10)

        '''muscle afferents generators'''
        self.Iagener_E = self.addIagener(self.muscle_E, self.muscle_E, 10, weight=20)
        self.Iagener_F = self.addIagener(self.muscle_F, self.muscle_F, speed * 6, weight=20)
        Iagener_E_1000 = self.addIagener(self.muscle_E, self.muscle_E, 1000, weight=20)
        Iagener_F_1000 = self.addIagener(self.muscle_E, self.muscle_E, 1000 + (speed * 6), weight=20)

        '''cutaneous inputs'''
        cfr = 200
        c_int = 1000 / cfr

        '''cutaneous inputs generators'''
        for layer in range(CV_number):
            self.dict_C[layer] = []
            for i in range(step_number):
                self.dict_C[layer].append(self.addgener(25 + speed * layer + i * (speed * CV_number + CV_0_len),
                                                        random.gauss(cfr, cfr / 10), (speed / c_int + 1)))



        '''Generators'''
        for i in range(step_number):
            self.C_0.append(self.addgener(25 + speed * 6 + i * (speed * 6 + CV_0_len), cfr, CV_0_len / c_int, False))


        # self.C_0.append(self.addgener(0, cfr, (speed / c_int)))

        for layer in range(CV_number):
            self.C_1.append(self.dict_CV_1[layer])
        self.C_1 = sum(self.C_1, [])
        # self.C_0 = sum(self.C_0, [])



        #not sure


        #need to check weight
        ''' BS '''
        for E_bs_gid in self.E_bs_gids:
            genconnect(E_bs_gid, self.BS_aff_E, 3.5, 3)

        for F_bs_gid in self.F_bs_gids:
            genconnect(F_bs_gid, self.BS_aff_F, 3.5, 3)

        #connectcells(self.BS_aff_F, self.V3F, 1.5, 3)

        '''STDP synapse'''
        connectcells(self.BS_aff_F, self.RG_F, 0.001, 3, stdptype=True)
        connectcells(self.BS_aff_E, self.RG_E, 0.001, 3, stdptype=True)

        '''generators of Ia aff'''
        ## TODO originally: 00005 and 0001
        ## TODO fix Iagener
        genconnect(self.Iagener_E, self.Ia_aff_E, 0.5, 1, False, 5)
        genconnect(self.Iagener_F, self.Ia_aff_F, 1.5, 1, False, 15)
        # genconnect(Iagener_E_1000, self.Ia_aff_E, 5.0, 1, False, 5)
        # genconnect(Iagener_F_1000, self.Ia_aff_F, 5.0, 1, False, 15)

        '''Ia2motor'''
        #ichanged the weight
        connectcells(self.Ia_aff_E, self.mns_E, 1.55, 1.5)
        connectcells(self.Ia_aff_F, self.mns_F, 1.5, 1.5)
        '''motor2muscles'''
        connectcells(self.mns_E, self.muscle_E, 15.5, 2, False, 45)
        connectcells(self.mns_F, self.muscle_F, 15.5, 2, False, 45)

        for layer in range(CV_number):
            '''Internal to RG topology'''
            connectinsidenucleus(self.dict_RG_F[layer])
            connectinsidenucleus(self.dict_RG_E[layer])

            '''RG2Motor'''
            connectcells(self.dict_RG_E[layer], self.mns_E, 2.75, 3)
            connectcells(self.dict_RG_F[layer], self.mns_F, 2.75, 3)

            '''Neg feedback RG -> Ia'''
            ## TODO why do we have this neg feedback ?
            if layer > 3:
                connectcells(self.dict_RG_E[layer], self.Ia_aff_E, layer * 0.0002, 1, True)
            else:
                '''RG2Ia'''
                connectcells(self.dict_RG_E[layer], self.Ia_aff_E, 0.0001, 1, True)

            '''RG2Motor, RG2Ia'''
            connectcells(self.dict_RG_F[layer], self.mns_F, 3.75, 2)
            '''Neg feedback loop RG->Ia'''
            connectcells(self.dict_RG_F[layer], self.Ia_aff_F, 0.95, 1, True)

        '''cutaneous inputs'''
        for layer in range(CV_number):
            connectcells(self.dict_C[layer], self.dict_CV_1[layer], 0.15 * k * speed, 2)
            connectcells(self.dict_CV_1[layer], self.dict_RG_E[layer], 0.00035 * k * speed, 3)

        # connectcells(self.IP_F, self.Ia_aff_F, 0.0015, 2, True)
        # connectcells(self.IP_E, self.Ia_aff_E, 0.0015, 2, True)

        '''Rhythm generators'''
        for layer in range(CV_number):
            connectcells(self.dict_RG_E[layer], self.InE, 0.001, 1)
            ## TODO weight 0.0001
            connectcells(self.dict_RG_F[layer], self.InF, 0.001, 1)
            connectcells(self.dict_RG_F[layer], self.In1, 0.001, 1,True)

        '''Ia2RG, RG2Motor'''
        connectcells(self.InE, self.RG_F, 0.5, 1, True)
        '''STDP synapse'''
        connectcells(self.Ia_aff_F, self.RG_F, 0.5, 1, stdptype=True)
        connectcells(self.In1, self.RG_F, 0.5, 1, True)

        # TODO check this too many reciprocal inh connections
        connectcells(self.InE, self.Ia_aff_F, 1.2, 1, True)
        connectcells(self.InE, self.mns_F, 0.8, 1, True)

        connectcells(self.InF, self.RG_E, 0.8, 1, True)
        '''STDP synapse'''
        connectcells(self.Ia_aff_E, self.RG_E, 0.5, 1, stdptype=True)

        # TODO check this too many reciprocal inh connections
        ## connectcells(self.InF, self.InE, 0.5, 1, True)
        connectcells(self.InF, self.Ia_aff_E, 0.5, 1, True)
        connectcells(self.InF, self.mns_E, 0.4, 1, True)

        '''reflex arc'''
        connectcells(self.InE, self.Ia_E, 0.001, 1)
        connectcells(self.Ia_aff_E, self.Ia_E, 0.008, 1)
        connectcells(self.mns_E, self.R_E, 0.00015, 1)
        connectcells(self.Ia_E, self.mns_F, 0.08, 1, True)
        connectcells(self.R_E, self.mns_E, 0.00015, 1, True)
        connectcells(self.R_E, self.Ia_E, 0.001, 1, True)

        connectcells(self.InF, self.Ia_F, 0.001, 1)
        connectcells(self.Ia_aff_F, self.Ia_F, 0.008, 1)
        connectcells(self.mns_F, self.R_F, 0.00015, 1)
        connectcells(self.Ia_F, self.mns_E, 0.08, 1, True)
        connectcells(self.R_F, self.mns_F, 0.00015, 1, True)
        connectcells(self.R_F, self.Ia_F, 0.001, 1, True)

        connectcells(self.R_E, self.R_F, 0.04, 1, True)
        connectcells(self.R_F, self.R_E, 0.04, 1, True)
        connectcells(self.Ia_E, self.Ia_F, 0.08, 1, True)
        connectcells(self.Ia_F, self.Ia_E, 0.08, 1, True)
        ## TODO check the inh connection
        connectcells(self.InE, self.InF, 0.04, 1, True)
        connectcells(self.InF, self.InE, 0.04, 1, True)

        ## TODO possibly project to RG_F
        connectcells(self.RG_F, self.V2a, 3.75, 3)
       #connectcells(self.RG_F, self.V3F, 3.75, 3)
        connectcells(self.RG_F, self.V0d, 3.75, 3)
        connectcells(self.V2a, self.V0v, 3.75, 3)
        # for layer in range(CV_number):
        #     connectcells(self.dict_V0v[layer], self.dict_V2a[layer], 0.001, 1)

    def addpool(self, num, name="test", neurontype="int"):
        '''
        Creates pool of cells determined by the neurontype and returns gids of the pool
        Parameters
        ----------
        num: int
            neurons number in pool
        name: string
            the name of the pool
        neurontype: string
            int: interneuron
            delay: interneuron with 5ht
            bursting: interneuron with bursting
            moto: motor neuron
            aff: afferent
            muscle: muscle fiber
        Returns
        -------
        gids: list
            the list of cells gids
        '''
        gids = []
        gid = 0
        if neurontype.lower() == "delay":
            delaytype = True
        else:
            delaytype = False

        if neurontype.lower() == "moto":
            diams = motodiams(num)
        for i in range(rank, num, nhost):
            if neurontype.lower() == "moto":
                cell = motoneuron(diams[i])
                self.motoneurons.append(cell)
            elif neurontype.lower() == "aff":
                cell = bioaffrat()
                self.afferents.append(cell)
            elif neurontype.lower() == "muscle":
                cell = muscle()
                self.motoneurons.append(cell)
                self.muscles.append(cell)
            elif neurontype.lower() == "bursting":
                cell = interneuron(False, bursting_mode=True)
                self.interneurons.append(cell)
            else:
                cell = interneuron(delaytype)
                self.interneurons.append(cell)

            while pc.gid_exists(gid) != 0:
                gid += 1
            gids.append(gid)
            pc.set_gid2node(gid, rank)
            nc = cell.connect2target(None)
            pc.cell(gid, nc)

        # Groups
        if (neurontype.lower() == "muscle"):
            self.musclegroups.append((gids, name))
            self.motogroups.append((gids, name))
        elif (neurontype.lower() == "moto"):
            self.motogroups.append((gids, name))
        elif neurontype.lower() == "aff":
            self.affgroups.append((gids, name))
        else:
            self.groups.append((gids, name))

        return gids

    def add_bs_geners(self, freq, spikes_per_step):
        E_bs_gids = []
        F_bs_gids = []
        for step in range(step_number):
            E_bs_gids.append(self.addgener(int(one_step_time * (step + 0.5)), freq, spikes_per_step, False))
            F_bs_gids.append(self.addgener(int(one_step_time * step), freq, spikes_per_step, False))
        logging.info(E_bs_gids)
        logging.info(F_bs_gids)
        return E_bs_gids, F_bs_gids

    def addgener(self, start, freq, nums, noise=True):
        '''
        Creates generator and returns generator gid

        Parameters
        ----------
        start: int
            generator start up
        freq: int
            generator frequency
        nums: int
            signals number
        noise: bool
            generates noizy output
        Returns
        -------
        gid: int
            generator gid
        '''
        gid = 0
        stim = h.NetStim()
        stim.number = nums
        if noise:
            stim.start = random.uniform(start - 3, start + 3)
            stim.noise = 0.05
        else:
            stim.start = start
        stim.interval = int(1000 / freq)
        # skinstim.noise = 0.1
        self.stims.append(stim)
        while pc.gid_exists(gid) != 0:
            gid += 1
        pc.set_gid2node(gid, rank)
        ncstim = h.NetCon(stim, None)
        pc.cell(gid, ncstim)
        return gid

    def addIagener(self, mn, mn2, start, weight=1.0):
        '''
        Creates self.Ia generators and returns generator gids
        Parameters
        ----------
        mn:
            motor neurons of agonist muscle that contract spindle
        mn2:
            motor neurons of antagonist muscle that extend spindle
        start: int
            generator start up
        num: int
            number in pool
        w_in: int
            weight of the connection
        Returns
        -------
        gids: list
            generators gids
        '''
        gid = 0
        moto = pc.gid2cell(random.randint(mn[0], mn[-1]))
        moto2 = pc.gid2cell(random.randint(mn2[0], mn2[-1]))
        stim = h.IaGenerator(0.5)
        stim.start = start
        h.setpointer(moto.muscle_unit(0.5)._ref_F_fHill, 'fhill', stim)
        h.setpointer(moto2.muscle_unit(0.5)._ref_F_fHill, 'fhill2', stim)
        self.stims.append(stim)
        while pc.gid_exists(gid) != 0:
            gid += 1
        pc.set_gid2node(gid, rank)
        ncstim = h.NetCon(stim, None)
        ncstim.weight[0] = weight
        pc.cell(gid, ncstim)

        return gid


class CPG_R:
    def __init__(self, speed, bs_fr, inh_p, step_number, N):

        self.interneurons = []
        self.motoneurons = []
        self.muscles = []
        self.afferents = []
        self.stims = []
        self.ncell = N
        self.groups = []
        self.motogroups = []
        self.musclegroups = []
        self.affgroups = []
        self.RG_E = []  # Rhythm generators of extensors
        self.RG_F = []  # Rhythm generators of flexor
        self.V3F = []
        self.V0d = []
        self.V2a = []
        self.C_1 = []
        self.C_0 = []
        self.V0v = []



        for layer in range(CV_number):
            '''cut and muscle feedback'''
            ## self.dict_CV = {layer: 'CV{}'.format(layer + 1)}
            self.dict_CV_1 = {layer: 'CV{}_1'.format(layer + 1)}
            self.dict_RG_E = {layer: 'RG{}_E'.format(layer + 1)}
            self.dict_RG_F = {layer: 'RG{}_F'.format(layer + 1)}
            self.dict_V3F = {layer: 'V3{}_F'.format(layer + 1)}
            self.dict_V2a = {layer: 'V2a{}_F'.format(layer + 1)}
            self.dict_V0v = {layer: 'V0v{}_F'.format(layer + 1)}

            self.dict_C = {layer: 'C{}'.format(layer + 1)}

        for layer in range(CV_number):
            '''Cutaneous pools'''
            ##self.dict_CV[layer] = self.addpool(self.ncell, "CV" + str(layer + 1), "aff")
            self.dict_CV_1[layer] = self.addpool(self.ncell, "CV" + str(layer + 1) + "_1", "aff")

            '''Rhythm generator pools'''
            self.dict_RG_E[layer] = self.addpool(self.ncell, "RG" + str(layer + 1) + "_E", "int")
            self.dict_RG_F[layer] = self.addpool(self.ncell, "RG" + str(layer + 1) + "_F", "int")
            self.RG_E.append(self.dict_RG_E[layer])
            self.RG_F.append(self.dict_RG_F[layer])

        '''RG'''
        self.RG_E = sum(self.RG_E, [])
        self.InE = self.addpool(nInt, "InE", "int")
        self.RG_F = sum(self.RG_F, [])
        self.InF = self.addpool(nInt, "InF", "int")
        self.In1 = self.addpool(nInt, "In1", "int")

        '''sensory and muscle afferents and brainstem and V3F'''
        ## self.sens_aff = self.addpool(nAff, "sens_aff", "aff")
        self.Ia_aff_E = self.addpool(nAff, "Ia_aff_E", "aff")
        self.Ia_aff_F = self.addpool(nAff, "Ia_aff_F", "aff")
        self.BS_aff_E = self.addpool(nAff, "BS_aff_E", "aff")
        self.BS_aff_F = self.addpool(nAff, "BS_aff_F", "aff")
        self.V3F = self.addpool(nInt, "V3F", "int")
        self.V0v = self.addpool(nInt, "V0v", "int")
        self.V2a = self.addpool(nInt, "V2a", "int")
        self.V0d = self.addpool(nInt, "V0d", "int")


        '''moto neuron pools'''
        self.mns_E = self.addpool(nMN, "mns_E", "moto")
        self.mns_F = self.addpool(nMN, "mns_F", "moto")

        '''muscles'''
        self.muscle_E = self.addpool(nMN * 30, "muscle_E", "muscle")
        self.muscle_F = self.addpool(nMN * 20, "muscle_F", "muscle")

        '''reflex arc'''
        self.Ia_E = self.addpool(nInt, "Ia_E", "int")
        self.R_E = self.addpool(nInt, "R_E", "int")  # Renshaw cells
        self.Ia_F = self.addpool(nInt, "Ia_F", "int")
        self.R_F = self.addpool(nInt, "R_F", "int")  # Renshaw cells
        # self.Iagener_E = []
        # self.Iagener_F = []

        '''BS'''
        # periodic stimulation
        self.E_bs_gids, self.F_bs_gids = self.add_bs_geners(bs_fr, 10)

        '''muscle afferents generators'''
        self.Iagener_E = self.addIagener(self.muscle_E, self.muscle_E, 10, weight=20)
        self.Iagener_F = self.addIagener(self.muscle_F, self.muscle_F, speed * 6, weight=20)
        Iagener_E_1000 = self.addIagener(self.muscle_E, self.muscle_E, 1000, weight=20)
        Iagener_F_1000 = self.addIagener(self.muscle_E, self.muscle_E, 1000 + (speed * 6), weight=20)

        '''cutaneous inputs'''
        cfr = 200
        c_int = 1000 / cfr

        '''cutaneous inputs generators'''
        for layer in range(CV_number):
            self.dict_C[layer] = []
            for i in range(step_number):
                self.dict_C[layer].append(self.addgener(25 + speed * layer + i * (speed * CV_number + CV_0_len),
                                                        random.gauss(cfr, cfr / 10), (speed / c_int + 1)))



        '''Generators'''
        for i in range(step_number):
            self.C_0.append(self.addgener(25 + speed * 6 + i * (speed * 6 + CV_0_len), cfr, CV_0_len / c_int, False))


        # self.C_0.append(self.addgener(0, cfr, (speed / c_int)))

        for layer in range(CV_number):
            self.C_1.append(self.dict_CV_1[layer])
        self.C_1 = sum(self.C_1, [])
        # self.C_0 = sum(self.C_0, [])



        #not sure


        #need to check weight
        ''' BS '''
        for E_bs_gid in self.E_bs_gids:
            genconnect(E_bs_gid, self.BS_aff_E, 3.5, 3)

        for F_bs_gid in self.F_bs_gids:
            genconnect(F_bs_gid, self.BS_aff_F, 3.5, 3)

       # connectcells(self.BS_aff_F, self.V3F, 1.5, 3)

        '''STDP synapse'''
        connectcells(self.BS_aff_F, self.RG_F, 0.001, 3, stdptype=True)
        connectcells(self.BS_aff_E, self.RG_E, 0.001, 3, stdptype=True)

        '''generators of Ia aff'''
        ## TODO originally: 00005 and 0001
        ## TODO fix Iagener
        genconnect(self.Iagener_E, self.Ia_aff_E, 0.5, 1, False, 5)
        genconnect(self.Iagener_F, self.Ia_aff_F, 1.5, 1, False, 15)
        # genconnect(Iagener_E_1000, self.Ia_aff_E, 5.0, 1, False, 5)
        # genconnect(Iagener_F_1000, self.Ia_aff_F, 5.0, 1, False, 15)

        '''Ia2motor'''
        #ichanged the weight
        connectcells(self.Ia_aff_E, self.mns_E, 1.55, 1.5)
        connectcells(self.Ia_aff_F, self.mns_F, 1.5, 1.5)
        '''motor2muscles'''
        connectcells(self.mns_E, self.muscle_E, 15.5, 2, False, 45)
        connectcells(self.mns_F, self.muscle_F, 15.5, 2, False, 45)

        for layer in range(CV_number):
            '''Internal to RG topology'''
            connectinsidenucleus(self.dict_RG_F[layer])
            connectinsidenucleus(self.dict_RG_E[layer])

            '''RG2Motor'''
            connectcells(self.dict_RG_E[layer], self.mns_E, 2.75, 3)
            connectcells(self.dict_RG_F[layer], self.mns_F, 2.75, 3)

            '''Neg feedback RG -> Ia'''
            ## TODO why do we have this neg feedback ?
            if layer > 3:
                connectcells(self.dict_RG_E[layer], self.Ia_aff_E, layer * 0.0002, 1, True)
            else:
                '''RG2Ia'''
                connectcells(self.dict_RG_E[layer], self.Ia_aff_E, 0.0001, 1, True)

            '''RG2Motor, RG2Ia'''
            connectcells(self.dict_RG_F[layer], self.mns_F, 3.75, 2)
            '''Neg feedback loop RG->Ia'''
            connectcells(self.dict_RG_F[layer], self.Ia_aff_F, 0.95, 1, True)

        '''cutaneous inputs'''
        for layer in range(CV_number):
            connectcells(self.dict_C[layer], self.dict_CV_1[layer], 0.15 * k * speed, 2)
            connectcells(self.dict_CV_1[layer], self.dict_RG_E[layer], 0.00035 * k * speed, 3)

        # connectcells(self.IP_F, self.Ia_aff_F, 0.0015, 2, True)
        # connectcells(self.IP_E, self.Ia_aff_E, 0.0015, 2, True)

        '''Rhythm generators'''
        for layer in range(CV_number):
            connectcells(self.dict_RG_E[layer], self.InE, 0.001, 1)
            ## TODO weight 0.0001
            connectcells(self.dict_RG_F[layer], self.InF, 0.001, 1)
            connectcells(self.dict_RG_F[layer], self.In1, 0.001, 1,True)

        '''Ia2RG, RG2Motor'''
        connectcells(self.InE, self.RG_F, 0.5, 1, True)
        '''STDP synapse'''
        connectcells(self.Ia_aff_F, self.RG_F, 0.5, 1, stdptype=True)
        connectcells(self.In1, self.RG_F, 0.5, 1, True)

        # TODO check this too many reciprocal inh connections
        connectcells(self.InE, self.Ia_aff_F, 1.2, 1, True)
        connectcells(self.InE, self.mns_F, 0.8, 1, True)

        connectcells(self.InF, self.RG_E, 0.8, 1, True)
        '''STDP synapse'''
        connectcells(self.Ia_aff_E, self.RG_E, 0.5, 1, stdptype=True)

        # TODO check this too many reciprocal inh connections
        ## connectcells(self.InF, self.InE, 0.5, 1, True)
        connectcells(self.InF, self.Ia_aff_E, 0.5, 1, True)
        connectcells(self.InF, self.mns_E, 0.4, 1, True)

        '''reflex arc'''
        connectcells(self.InE, self.Ia_E, 0.001, 1)
        connectcells(self.Ia_aff_E, self.Ia_E, 0.008, 1)
        connectcells(self.mns_E, self.R_E, 0.00015, 1)
        connectcells(self.Ia_E, self.mns_F, 0.08, 1, True)
        connectcells(self.R_E, self.mns_E, 0.00015, 1, True)
        connectcells(self.R_E, self.Ia_E, 0.001, 1, True)

        connectcells(self.InF, self.Ia_F, 0.001, 1)
        connectcells(self.Ia_aff_F, self.Ia_F, 0.008, 1)
        connectcells(self.mns_F, self.R_F, 0.00015, 1)
        connectcells(self.Ia_F, self.mns_E, 0.08, 1, True)
        connectcells(self.R_F, self.mns_F, 0.00015, 1, True)
        connectcells(self.R_F, self.Ia_F, 0.001, 1, True)

        connectcells(self.R_E, self.R_F, 0.04, 1, True)
        connectcells(self.R_F, self.R_E, 0.04, 1, True)
        connectcells(self.Ia_E, self.Ia_F, 0.08, 1, True)
        connectcells(self.Ia_F, self.Ia_E, 0.08, 1, True)
        ## TODO check the inh connection
        connectcells(self.InE, self.InF, 0.04, 1, True)
        connectcells(self.InF, self.InE, 0.04, 1, True)

        ## TODO possibly project to RG_F
        connectcells(self.RG_F, self.V2a, 3.75, 3)
        #connectcells(self.RG_F, self.V3F, 3.75, 3)
        connectcells(self.RG_F, self.V0d, 3.75, 3)
        connectcells(self.V2a, self.V0v, 3.75, 3)
        # for layer in range(CV_number):
        #     connectcells(self.dict_V0v[layer], self.dict_V2a[layer], 0.001, 1)

    def addpool(self, num, name="test", neurontype="int"):
        '''
        Creates pool of cells determined by the neurontype and returns gids of the pool
        Parameters
        ----------
        num: int
            neurons number in pool
        name: string
            the name of the pool
        neurontype: string
            int: interneuron
            delay: interneuron with 5ht
            bursting: interneuron with bursting
            moto: motor neuron
            aff: afferent
            muscle: muscle fiber
        Returns
        -------
        gids: list
            the list of cells gids
        '''
        gids = []
        gid = 0
        if neurontype.lower() == "delay":
            delaytype = True
        else:
            delaytype = False

        if neurontype.lower() == "moto":
            diams = motodiams(num)
        for i in range(rank, num, nhost):
            if neurontype.lower() == "moto":
                cell = motoneuron(diams[i])
                self.motoneurons.append(cell)
            elif neurontype.lower() == "aff":
                cell = bioaffrat()
                self.afferents.append(cell)
            elif neurontype.lower() == "muscle":
                cell = muscle()
                self.motoneurons.append(cell)
                self.muscles.append(cell)
            elif neurontype.lower() == "bursting":
                cell = interneuron(False, bursting_mode=True)
                self.interneurons.append(cell)
            else:
                cell = interneuron(delaytype)
                self.interneurons.append(cell)

            while pc.gid_exists(gid) != 0:
                gid += 1
            gids.append(gid)
            pc.set_gid2node(gid, rank)
            nc = cell.connect2target(None)
            pc.cell(gid, nc)

        # Groups
        if (neurontype.lower() == "muscle"):
            self.musclegroups.append((gids, name))
            self.motogroups.append((gids, name))
        elif (neurontype.lower() == "moto"):
            self.motogroups.append((gids, name))
        elif neurontype.lower() == "aff":
            self.affgroups.append((gids, name))
        else:
            self.groups.append((gids, name))

        return gids

    def add_bs_geners(self, freq, spikes_per_step):
        E_bs_gids = []
        F_bs_gids = []
        for step in range(step_number):
            F_bs_gids.append(self.addgener(int(one_step_time * (step + 0.5)), freq, spikes_per_step, False))
            E_bs_gids.append(self.addgener(int(one_step_time * step), freq, spikes_per_step, False))
        logging.info(E_bs_gids)
        logging.info(F_bs_gids)
        return E_bs_gids, F_bs_gids

    def addgener(self, start, freq, nums, noise=True):
        '''
        Creates generator and returns generator gid

        Parameters
        ----------
        start: int
            generator start up
        freq: int
            generator frequency
        nums: int
            signals number
        noise: bool
            generates noizy output
        Returns
        -------
        gid: int
            generator gid
        '''
        gid = 0
        stim = h.NetStim()
        stim.number = nums
        if noise:
            stim.start = random.uniform(start - 3, start + 3)
            stim.noise = 0.05
        else:
            stim.start = start
        stim.interval = int(1000 / freq)
        # skinstim.noise = 0.1
        self.stims.append(stim)
        while pc.gid_exists(gid) != 0:
            gid += 1
        pc.set_gid2node(gid, rank)
        ncstim = h.NetCon(stim, None)
        pc.cell(gid, ncstim)
        return gid

    def addIagener(self, mn, mn2, start, weight=1.0):
        '''
        Creates self.Ia generators and returns generator gids
        Parameters
        ----------
        mn:
            motor neurons of agonist muscle that contract spindle
        mn2:
            motor neurons of antagonist muscle that extend spindle
        start: int
            generator start up
        num: int
            number in pool
        w_in: int
            weight of the connection
        Returns
        -------
        gids: list
            generators gids
        '''
        gid = 0
        moto = pc.gid2cell(random.randint(mn[0], mn[-1]))
        moto2 = pc.gid2cell(random.randint(mn2[0], mn2[-1]))
        stim = h.IaGenerator(0.5)
        stim.start = start
        h.setpointer(moto.muscle_unit(0.5)._ref_F_fHill, 'fhill', stim)
        h.setpointer(moto2.muscle_unit(0.5)._ref_F_fHill, 'fhill2', stim)
        self.stims.append(stim)
        while pc.gid_exists(gid) != 0:
            gid += 1
        pc.set_gid2node(gid, rank)
        ncstim = h.NetCon(stim, None)
        ncstim.weight[0] = weight
        pc.cell(gid, ncstim)

        return gid

def connectcells(pre, post, weight, delay=1, inhtype=False, N=50, stdptype=False, threshold=10):
    ''' Connects with excitatory synapses
      Parameters
      ----------
      pre: list
          list of presynase neurons gids
      post: list
          list of postsynapse neurons gids
      weight: float
          weight of synapse
          used with Gaussself.Ian distribution
      delay: int
          synaptic delay
          used with Gaussself.Ian distribution
      nsyn: int
          numder of synapses
      inhtype: bool
          is this connection inhibitory?
      N: int
          number of synapses
      stdptype: bool
           is connection stdp?
      threshold: int
            voltage thershold
    '''
    nsyn = random.randint(N - 15, N)
    for post_gid in post:
        if pc.gid_exists(post_gid):
            for j in range(nsyn):
                src_gid = random.randint(pre[0], pre[-1])
                target = pc.gid2cell(post_gid)
                if stdptype:
                    if inhtype:
                        syn = target.synlistinhstdp[j]
                        nc = pc.gid_connect(src_gid, syn)
                        nc.delay = delay
                        pc.threshold(src_gid, threshold)
                        """Create STDP synapses"""
                        dummy = h.Section()  # Create a dummy section to put the point processes in
                        stdpmech = h.STDP(0, sec=dummy)  # Create the STDP mechanism
                        # TODO check target, threshold,
                        presyn = pc.gid_connect(src_gid,
                                                stdpmech)  # threshold, delay, 1)  # Feed presynaptic spikes to the STDP mechanism -- must have weight >0
                        presyn.delay = delay
                        presyn.weight = 1
                        pstsyn = pc.gid_connect(post_gid,
                                                stdpmech)  # threshold, delay, -1)  # Feed postsynaptic spikes to the STDP mechanism -- must have weight <0
                        pstsyn.delay = delay
                        pstsyn.weight = -1
                        pc.threshold(post_gid, threshold)
                        h.setpointer(nc._ref_weight[0], 'synweight',
                                     stdpmech)  # Point the STDP mechanism to the connection weight
                        inhstdpnclist.append(nc)
                    else:
                        syn = target.synlistex[j]
                        nc = pc.gid_connect(src_gid, syn)
                        nc.delay = delay
                        pc.threshold(src_gid, threshold)
                        """Create STDP synapses"""
                        dummy = h.Section()  # Create a dummy section to put the point processes in
                        stdpmech = h.STDP(0, sec=dummy)  # Create the STDP mechanism
                        # TODO check target, threshold,
                        presyn = pc.gid_connect(src_gid,
                                                stdpmech)  # threshold, delay, 1)  # Feed presynaptic spikes to the STDP mechanism -- must have weight >0
                        presyn.delay = delay
                        presyn.weight[0] = 1
                        pstsyn = pc.gid_connect(post_gid,
                                                stdpmech)  # threshold, delay, -1)  # Feed postsynaptic spikes to the STDP mechanism -- must have weight <0
                        pstsyn.delay = delay
                        pstsyn.weight[0] = -1
                        pc.threshold(post_gid, threshold)
                        h.setpointer(nc._ref_weight[0], 'synweight',
                                     stdpmech)  # Point the STDP mechanism to the connection weight
                        exstdpnclist.append(nc)
                        logging.info(nc._ref_weight[0])
                        # nc.weight[0] = random.gauss(weight, weight / 6) # str

                else:
                    if inhtype:
                        syn = target.synlistinh[j]
                        nc = pc.gid_connect(src_gid, syn)
                        inhnclist.append(nc)
                    else:
                        syn = target.synlistex[j]
                        nc = pc.gid_connect(src_gid, syn)
                        exnclist.append(nc)
                        # nc.weight[0] = random.gauss(weight, weight / 6) # str

                # if mode == 'STR':
                #     nc.weight[0] = 0 # str
                # else:
                nc.weight[0] = random.gauss(weight, weight / 5)
                nc.delay = random.gauss(delay, delay / 5)


def genconnect(gen_gid, afferents_gids, weight, delay, inhtype=False, N=50):
    ''' Connects with generator
      Parameters
      ----------
      afferents_gids: list
          list of presynase neurons gids
      gen_gid: int
          generator gid
      weight: float
          weight of synapse
          used with Gaussian distribution
      delay: int
          synaptic delay
          used with Gaussian distribution
      nsyn: int
          numder of synapses
      inhtype: bool
          is this connection inhibitory?
    '''
    nsyn = random.randint(N, N + 5)
    for i in afferents_gids:
        if pc.gid_exists(i):
            for j in range(nsyn):
                target = pc.gid2cell(i)
                if inhtype:
                    syn = target.synlistinh[j]
                    # nc = pc.gid_connect(gen_gid, syn)
                    # stimnclist.append(nc)
                    # nc.delay = random.gauss(delay, delay / 6)
                    # nc.weight[0] = 0
                else:
                    syn = target.synlistees[j]
                    # nc = pc.gid_connect(gen_gid, syn)
                    # stimnclist.append(nc)
                    # nc.delay = random.gauss(delay, delay / 6)
                    # nc.weight[0] = random.gauss(weight, weight / 6)
                nc = pc.gid_connect(gen_gid, syn)
                stimnclist.append(nc)
                nc.delay = random.gauss(delay, delay / 5)
                nc.weight[0] = random.gauss(weight, weight / 6)


def connectinsidenucleus(nucleus):
    connectcells(nucleus, nucleus, 0.25, 0.5)


def spike_record(pool, extra=False):
    ''' Records spikes from gids
      Parameters
      ----------
      pool: list
        list of neurons gids
      extra: bool
          extracellular or intracellular voltages to record
      Returns
      -------
      v_vec: list of h.Vector()
          recorded voltages
    '''
    v_vec = []

    for i in pool:
        cell = pc.gid2cell(i)
        vec = h.Vector(np.zeros(int(time_sim / 0.025 + 1), dtype=np.float32))
        if extra:
            vec.record(cell.soma(0.5)._ref_vext[0])
        else:
            vec.record(cell.soma(0.5)._ref_v)
        v_vec.append(vec)
    return v_vec


def force_record(pool):
    ''' Records force from gids of motor neurons muscle unit
      Parameters
      ----------
      pool: list
        list of neurons gids
      Returns
      -------
      v_vec: list of h.Vector()
          recorded voltage
    '''
    v_vec = []
    for i in pool:
        cell = pc.gid2cell(i)
        vec = h.Vector(np.zeros(int(time_sim / 0.025 + 1), dtype=np.float32))
        vec.record(cell.muscle_unit(0.5)._ref_F_fHill)
        v_vec.append(vec)
    return v_vec


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


def spikeout(pool_left, pool_right, name, version, v_vec_left, v_vec_right):

    ''' Reports simulation results for both legs
      Parameters
      ----------
      pool_left, pool_right: list
        list of neurons gids for left and right legs
      name: string
        pool name
      version: int
          test number
      v_vec_left, v_vec_right: list of h.Vector()
          recorded voltage for left and right legs
    '''
    global rank
    pc.barrier()

    # Process data for left leg
    vec_left = h.Vector()
    for i in range(nhost):
        if i == rank:
            outavg_left = [list(v_vec_left[j]) for j in range(len(pool_left))]
            outavg_left = np.mean(np.array(outavg_left), axis=0, dtype=np.float32)
            vec_left = vec_left.from_python(outavg_left)
        pc.barrier()
    pc.barrier()
    logging.info("start hey2 ")

    result_left = pc.py_gather(vec_left, 0)

    # Process data for right leg
    vec_right = h.Vector()
    for i in range(nhost):
        if i == rank:
            outavg_right = [list(v_vec_right[j]) for j in range(len(pool_right))]
            outavg_right = np.mean(np.array(outavg_right), axis=0, dtype=np.float32)
            vec_right = vec_right.from_python(outavg_right)
        pc.barrier()
    pc.barrier()


    result_right = pc.py_gather(vec_right, 0)
    logging.info("start hey ")

    if rank == 0:
        logging.info("start recording")
        result = np.mean(np.array(result_right), axis=0, dtype=np.float32)
        with hdf5.File('./res/{}_sp_{}_CVsR_{}_bs_{}.hdf5'.format(name, speed, CV_number, bs_fr), 'w') as file:
            for i in range(step_number):
                sl = slice((int(1000 / bs_fr) * 40 + i * one_step_time * 40),
                           (int(1000 / bs_fr) * 40 + (i + 1) * one_step_time * 40))
                file.create_dataset('#0_step_{}'.format(i), data=np.array(result)[sl], compression="gzip")

        result = np.mean(np.array(result_left), axis=0, dtype=np.float32)
        with hdf5.File('./res/{}_sp_{}_CVsL_{}_bs_{}.hdf5'.format(name, speed, CV_number, bs_fr), 'w') as file:
            for i in range(step_number):
                sl = slice((int(1000 / bs_fr) * 40 + i * one_step_time * 40),
                           (int(1000 / bs_fr) * 40 + (i + 1) * one_step_time * 40))
                file.create_dataset('#0_step_{}'.format(i), data=np.array(result)[sl], compression="gzip")
        logging.info("done recording")
    else:
        logging.info(rank)


def prun(speed, step_number):
    ''' simulation control
    Parameters
    ----------
    speed: int
      duration of each layer

    Returns
    -------
    t: list of h.Vector()
      recorded time
    '''
    pc.timeout(0)
    t = h.Vector().record(h._ref_t)
    for neuron in (cpg_left_leg.interneurons + cpg_left_leg.motoneurons +
                   cpg_right_leg.interneurons + cpg_right_leg.motoneurons):
        neuron.soma(0.5).v = -70 * mV  # Set initial membrane potential
    tstop = time_sim  # 25 + (6 * speed + 125) * step_number
    pc.set_maxstep(10)
    h.stdinit()
    h.finitialize(-70 * mV)
    pc.psolve(tstop)
    return t


def finish():
    ''' proper exit '''
    pc.runworker()
    pc.done()
    print("hi after finish")
    h.quit()

def connect_cpg_legs_E(cpg_first_leg, cpg_second_leg, weight, delay):

    connections = [
        (cpg_first_leg.V0v, cpg_second_leg.In1),
    ]
    connections_1 = [
        (cpg_first_leg.In1, cpg_second_leg.V0v),
    ]
    # Establish the connections from first leg to second leg
    for first_leg_group, second_leg_group in connections:
        connectcells(first_leg_group, second_leg_group, weight, delay,False)

    # Establish the connections from second leg to first leg (vice versa)
    for second_leg_group, first_leg_group in connections_1:
        connectcells(second_leg_group, first_leg_group, weight, delay,False)
    logging.info("Connected left and right legs done1")

def connect_cpg_legs_I(cpg_first_leg, cpg_second_leg, weight, delay):

    connections = [
        (cpg_first_leg.V0d, cpg_second_leg.RG_F),
    ]
    connections_1 = [
        (cpg_first_leg.RG_F, cpg_second_leg.V0d),
    ]
    # Establish the connections from first leg to second leg
    for first_leg_group, second_leg_group in connections:
        connectcells(first_leg_group, second_leg_group, weight, delay,True)

    # Establish the connections from second leg to first leg (vice versa)
    for second_leg_group, first_leg_group in connections_1:
      connectcells(second_leg_group, first_leg_group, weight, delay,True)
    logging.info("Connected left and right legs done")


def set_interleg_phase(cpg_left, cpg_right, cycle_duration):
    """
    Adjusts the phase of activation between the left and right CPG neuron groups to synchronize leg movements.

    Parameters:
    - cpg_left: Instance of the CPG model for the left leg.
    - cpg_right: Instance of the CPG model for the right leg.
    - cycle_duration: Duration of one locomotion cycle in ms.
    """
    phase_shift = 0.5 * cycle_duration  # Half cycle phase shift

    # Assuming RG_E and RG_F are lists of neuron objects with a 'stim' attribute
    # Adjusting the stimulator's start times for the neurons

    # Left leg extensors synchronized with Right leg flexors
    for neuron in cpg_left.RG_E:
        if hasattr(neuron, 'stim'):
            neuron.stim.start = 0  # Starts at the beginning of the cycle
    for neuron in cpg_right.RG_F:
        if hasattr(neuron, 'stim'):
            neuron.stim.start = phase_shift  # Starts half a cycle later

    # Right leg extensors synchronized with Left leg flexors
    for neuron in cpg_right.RG_E:
        if hasattr(neuron, 'stim'):
            neuron.stim.start = phase_shift  # Starts half a cycle later
    for neuron in cpg_left.RG_F:
        if hasattr(neuron, 'stim'):
            neuron.stim.start = 0  # Starts at the beginning of the cycle


if __name__ == '__main__':
    k_nrns = 0
    k_name = 1

    for i in range(versions):
        # Create CPG models for both legs
        cpg_left_leg = CPG_L(speed, bs_fr, 100, step_number, N)
        cpg_right_leg = CPG_R(speed, bs_fr, 100, step_number, N)
        logging.info("Created both legs CPG models")
       #cpg_right_leg.set_phase(0.5)
        # Connect left and right legs and get the connected neuron groups
        connect_cpg_legs_E(cpg_left_leg, cpg_right_leg, weight=0.5, delay=1.0)
        connect_cpg_legs_I(cpg_left_leg, cpg_right_leg, weight=0.5, delay=1.0)
        # Example of how to call this function in  setup
       # set_interleg_phase(cpg_left_leg, cpg_right_leg, cycle_duration=300)  # Assuming 100 ms cycle duration

        logging.info("Connected left and right legs")

        # Initialize recorders for both legs
        motorecorders_left = []
        motorecorders_right = []
        motorecorders_mem_left = []
        motorecorders_mem_right = []
        affrecorders_left = []
        affrecorders_right = []
        recorders_left = []
        recorders_right = []
        connected_recordings_left = []
        connected_recordings_right = []
        # for left_gid, right_gid in connected_neurons:
        #     connected_recordings_left.append(spike_record([left_gid], True))  # Assuming `spike_record` can handle list of GIDs
        #     connected_recordings_right.append(spike_record([right_gid], True))
        # Setup recorders for each group in both CPG models
        for group_left, group_right in zip(cpg_left_leg.motogroups, cpg_right_leg.motogroups):
            motorecorders_left.append(spike_record(group_left[k_nrns], True))

        for group_left, group_right in zip(cpg_left_leg.motogroups, cpg_right_leg.motogroups):
            motorecorders_mem_left.append(spike_record(group_left[k_nrns]))
            motorecorders_mem_right.append(spike_record(group_right[k_nrns]))

        for group_left, group_right in zip(cpg_left_leg.affgroups, cpg_right_leg.affgroups):
            affrecorders_left.append(spike_record(group_left[k_nrns]))
            affrecorders_right.append(spike_record(group_right[k_nrns]))

        for group_right in cpg_right_leg.groups:
          recorders_right.append(spike_record(group_right[k_nrns]))
        for group_left in cpg_left_leg.groups:
           recorders_left.append(spike_record(group_left[k_nrns]))



        logging.info("Added recorders for both legs")

        # Start simulation
        print("- " * 10, "\nstart simulation")
        t = prun(speed, step_number)
        print("- " * 10, "\nend simulation")

        logging.info("Simulation done for both legs")

        # Save the time vector
        with open('./res/time.txt', 'w') as time_file:
            for time_point in t:
                time_file.write(str(time_point) + "\n")

        # Output results for both legs and the connections
        for group_left, group_right, recorder_left, recorder_right in zip(cpg_left_leg.motogroups, cpg_right_leg.motogroups, motorecorders_left, motorecorders_right):
            spikeout(group_left[k_nrns], group_right[k_nrns], group_left[k_name], i, recorder_left, recorder_right)

        for group_left, group_right, recorder_left, recorder_right in zip(cpg_left_leg.motogroups, cpg_right_leg.motogroups, motorecorders_mem_left, motorecorders_mem_right):
            spikeout(group_left[k_nrns], group_right[k_nrns], 'mem_' + group_left[k_name], i, recorder_left, recorder_right)

        for group_left, group_right, recorder_left, recorder_right in zip(cpg_left_leg.affgroups, cpg_right_leg.affgroups, affrecorders_left, affrecorders_right):
            spikeout(group_left[k_nrns], group_right[k_nrns], group_left[k_name], i, recorder_left, recorder_right)

        for group_left, group_right, recorder_left, recorder_right in zip(cpg_left_leg.groups, cpg_right_leg.groups, recorders_left, recorders_right):
            spikeout(group_left[k_nrns], group_right[k_nrns], group_left[k_name], i, recorder_left, recorder_right)

        # for index, (rec_left, rec_right) in enumerate(zip(connected_recordings_left, connected_recordings_right)):
        #     spikeout([left_gid for left_gid, _ in connected_neurons],
        #              [right_gid for _, right_gid in connected_neurons],
        #              f"connected_neurons_group_{index}", i, rec_left, rec_right)

        logging.info("Recorded activity for connected neurons between legs")
        logging.info("Recorded for both legs and their connections")

    # Finish the simulation
    finish()
