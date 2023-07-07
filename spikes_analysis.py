import numpy as np
from brian2 import *
import pickle
import os
from scipy.signal import savgol_filter
from matplotlib.pyplot import figure
from matplotlib.pyplot import cm
import seaborn as sns


def read_file(file_name):
    with open(file_name, 'rb') as m_f:
        m_mon = pickle.load(m_f)
        firing_times = m_mon["t"] / second
        neuron_ids = m_mon["i"]
        return neuron_ids, firing_times


def divide_in_steps(neuron_ids, firing_times):
    i = -1
    neuron_ids_steps = []
    firing_times_steps = []
    for neuron_id, firing_time in zip(neuron_ids, firing_times):
        if int(firing_time) > i:
            i = int(firing_time)
            neuron_ids_steps.append([])
            firing_times_steps.append([])
        neuron_ids_steps[i].append(neuron_id)
        firing_times_steps[i].append(firing_time)
    return neuron_ids_steps, firing_times_steps


def generate_plots(plots_n):
    for file_name in os.scandir('pickle_'):
        neuron_ids_steps, firing_times_steps = divide_in_steps(*read_file(file_name))
        steps_to_plot = np.linspace(0, len(neuron_ids_steps) - 1, plots_n).astype(int)
        file_name = file_name.name.split('.')[0]
        for step in steps_to_plot:
            plot(firing_times_steps[step], neuron_ids_steps[step], 'b,')
            savefig(f'plots/{file_name}_{step}.png')
            close()


def get_normalized_step(ids, times, i):
    step_out_ids, step_out_times = np.asarray(ids[i]), np.asarray(times[i])
    step_out_ids -= np.min(step_out_ids)
    step_out_times -= np.min(step_out_times)
    step_out_ids = step_out_ids / np.max(step_out_ids)
    step_out_times = step_out_times / np.max(step_out_times)

    return step_out_times, step_out_ids


def spikes_on_diagonal(ids, times, eps=0.1):
    diagonal_spikes = 0
    for id, time in zip(ids, times):
        if id < 0.25 + eps and time < 0.3 + eps:
            diagonal_spikes += 1
            continue
        if id < 0.5 + eps and 0.2 - eps < time < 0.6 + eps:
            diagonal_spikes += 1
            continue
        if id < 0.75 + eps and 0.4 - eps < time < 0.8 + eps:
            diagonal_spikes += 1
            continue
        if time > 0.75 - eps:
            diagonal_spikes += 1

    diagonal_spikes_share = diagonal_spikes / len(ids)
    return diagonal_spikes, diagonal_spikes_share

def test_correlation():
    in_ids, in_times = divide_in_steps(*read_file('pickle_/l_e_cut_spikes.pickle'))
    out_ids, out_times = divide_in_steps(*read_file('pickle_/l_e_rg_neurons_spikes.pickle'))

    absolute = []
    shared = []
    for i in range(len(in_ids)):
        rg_neurons_diagonal_spikes, rg_neurons_diagonal_spikes_share = spikes_on_diagonal(
            *get_normalized_step(in_ids, in_times, i))
        cut_spikes, _ = spikes_on_diagonal(*get_normalized_step(out_ids, in_times, i))
        absolute.append(rg_neurons_diagonal_spikes / cut_spikes)
        shared.append(rg_neurons_diagonal_spikes_share / cut_spikes)
    # plot(range(len(absolute)), absolute)
    plot(range(len(shared)), shared)
    show()

test_correlation()
generate_plots(3)