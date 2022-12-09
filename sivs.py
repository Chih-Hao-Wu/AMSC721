__author__ = "Chih Hao Wu"
__version__ = "0.0.4"

import numpy as np
from math import floor
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import yaml

with open("config.yaml") as config_file:
    parameters = yaml.safe_load(config_file)

class Update:
# update state distribution after intervention su -> sv when t = nT

    @staticmethod
    def not_pulse(model, initial, t):
        soln = odeint(model, initial, t)
        return soln.T

    @staticmethod
    def pulse(states: list[np.ndarray], a):
        su_f, i_f, sv_f, n_f, su_m, i_m, sv_m, n_m = states

        # only su and sv are changed, i and n remain the same
        su_f, sv_f = Update.pulse_rules(su_f, sv_f, a)
        su_m, sv_m = Update.pulse_rules(su_m, sv_m, a)

        return su_f, i_f, sv_f, n_f, su_m, i_m, sv_m, n_m

    @staticmethod
    def pulse_rules(su: np.ndarray, sv: np.ndarray, a: float):
        su_aslist, sv_aslist = su.tolist(), sv.tolist()
        last_su, last_sv = su_aslist.pop(), sv_aslist.pop()

        su_aslist.append((1 - a)*last_su)
        sv_aslist.append(a*last_su + last_sv)

        return np.asarray(su_aslist), np.asarray(sv_aslist)


class Model:

    def __init__(self, params: dict):
        self.theta = params["theta"]
        self.psi = params["psi"]
        self.p = params["p"]
        self.pi = params["pi"]
        self.mu = params["mu"]
        self.beta = params["beta"]
        self.gamma = params["gamma"]
        self.delta = params["delta"]
        
    def change_vaccine_efficacy(self, new_efficacy: dict):
        for key, val in new_efficacy.items():
            self.psi[key] = val

    @staticmethod
    def dSdt(su, sv, i1, i2, **kwargs):

        return  (1 - kwargs["p"]) * kwargs["pi"] + \
                kwargs["gamma"] * i1 + \
                kwargs["theta"] * sv - \
                (kwargs["mu"] + kwargs["beta"] * i2) * su

    @staticmethod
    def dIdt(su, sv, i1, i2, **kwargs):

        return  (su + (1 - kwargs["psi"]) * sv) * (kwargs["beta"] * i2) - \
                (kwargs["mu"] + kwargs["gamma"] + kwargs["delta"]) * i1
    
    @staticmethod
    def dVdt(sv, i, **kwargs):

        return  kwargs["p"] * kwargs["pi"] - \
                (kwargs["theta"] + kwargs["mu"] + (1 - kwargs["psi"]) * kwargs["beta"] * i) * sv

    @staticmethod
    def dNdt(i, n, **kwargs):

        return  kwargs["pi"] - \
                kwargs["mu"] * n - \
                kwargs["delta"] * i

    # define SIVS model describing how to evolve state distributions
    def sivs(self, x, t) -> list:
        
        su_f, i_f, sv_f, n_f, su_m, i_m, sv_m, n_m = x

        d_SU_F = Model.dSdt(
                    su_f, sv_f, i_f, i_m, 
                    p=self.p["female"],
                    pi=self.pi["female"],
                    gamma=self.gamma["female"],
                    theta=self.theta,
                    mu=self.mu["female"],
                    beta=self.beta["male"]
                    )

        d_IN_F = Model.dIdt(
                    su_f, sv_f, i_f, i_m,
                    psi=self.psi["female"],
                    beta=self.beta["male"],
                    mu=self.mu["female"],
                    gamma=self.gamma["female"],
                    delta=self.delta["female"]
                    )

        d_SV_F = Model.dVdt(
                    sv_f, i_m,
                    p=self.p["female"],
                    pi=self.pi["female"],
                    theta=self.theta,
                    mu=self.mu["female"],
                    psi=self.psi["female"],
                    beta=self.beta["male"]
                    )

        d_N_F = Model.dNdt(
                    i_f, n_f,
                    pi=self.pi["female"],
                    mu=self.mu["female"],
                    delta=self.delta["female"]
                    )
        
        d_SU_M = Model.dSdt(
                    su_m, sv_m, i_m, i_f, 
                    p=self.p["male"],
                    pi=self.pi["male"],
                    gamma=self.gamma["male"],
                    theta=self.theta,
                    mu=self.mu["male"],
                    beta=self.beta["female"]
                    )

        d_IN_M = Model.dIdt(
                    su_m, sv_m, i_m, i_f,
                    psi=self.psi["male"],
                    beta=self.beta["female"],
                    mu=self.mu["male"],
                    gamma=self.gamma["male"],
                    delta=self.delta["male"]
                    )

        d_SV_M = Model.dVdt(
                    sv_m, i_f,
                    p=self.p["male"],
                    pi=self.pi["male"],
                    theta=self.theta,
                    mu=self.mu["male"],
                    psi=self.psi["male"],
                    beta=self.beta["female"]
                    )

        d_N_M = Model.dNdt(
                    i_m, n_m,
                    pi=self.pi["male"],
                    mu=self.mu["male"],
                    delta=self.delta["male"]
                    )
        
        return [d_SU_F, d_IN_F, d_SV_F, d_N_F, d_SU_M, d_IN_M, d_SV_M, d_N_M]

# RUN NUMERICAL SIMULATIONS FROM HERE

# initial state distribution
INITIAL = (1000, 500, 0, 1500, 1000, 500, 0, 1500)
# simulation parameters
SAMPLING_INTERVAL = 500
TOTAL_TIME = 75
NAMED_STATES = [
    "susceptible unvaccinated female",
    "infected female",
    "susceptible vaccinated female",
    "total female",
    "susceptible unvaccinated male",
    "infected male",
    "susceptible vaccinated male",
    "total male",
]


def simulation_vaccination(model: Model, interpulse: int):
    t = np.linspace(0, interpulse-1, SAMPLING_INTERVAL)
    cycles = floor(TOTAL_TIME/interpulse)

    states = Update.not_pulse(model.sivs, INITIAL, t)
    states = Update.pulse(states, parameters["alpha"])
    states = dict(zip(NAMED_STATES, states))

    new_initial = [state[-1] for state in states.values()]

    for _ in range(cycles):

        next_nstates = Update.not_pulse(model.sivs, new_initial, t)
        next_nstates = Update.pulse(next_nstates, parameters["alpha"])
        # remove the first element in subsequently simulated, since that is just the initial condition
        # which is the last state distribution
        next_nstates = [i[1:] for i in next_nstates]

        # concatenate with running ndarrays
        for i, state in enumerate(states.keys()):
            states[state] = np.concatenate((states[state], next_nstates[i]))

        new_initial = [state[-1] for state in states.values()]

    fT = np.linspace(0, TOTAL_TIME-1, SAMPLING_INTERVAL+(SAMPLING_INTERVAL-1)*cycles)

    return (fT, states)

def simulation_nonvaccination(model: Model):
    t = np.linspace(0, TOTAL_TIME, SAMPLING_INTERVAL)

    states = Update.not_pulse(model.sivs, INITIAL, t)
    states = dict(zip(NAMED_STATES, states))

    return (t, states)

modelA = Model(parameters) # base model
fT, state = simulation_vaccination(modelA, interpulse=2)
fT1, state1 = simulation_vaccination(modelA, interpulse=4)
fT3, state3 = simulation_vaccination(modelA, interpulse=16)

fT_proposed, state_proposed = simulation_vaccination(modelA, interpulse=5) # "standard policy"
fT_no, state_no = simulation_nonvaccination(modelA) # "vaccine free"

# DIFFERENT MODELS VARYING TARGETED VACCINE PROFILE; CHANGING EFFICACY

modelB = Model(parameters)
modelB.change_vaccine_efficacy({"male": 0.01, "female": 0.01})
fT_B, state_B = simulation_vaccination(modelB, interpulse=3)

modelC = Model(parameters)
modelC.change_vaccine_efficacy({"male": 0.25, "female": 0.25})
fT_C, state_C = simulation_vaccination(modelC, interpulse=3)

modelD = Model(parameters)
modelD.change_vaccine_efficacy({"male": 0.75, "female": 0.75})
fT_D, state_D = simulation_vaccination(modelD, interpulse=3)

# PLOT FIGURES

fig, ax = plt.subplots()
""" OK
# how do state distributions look for this model
ax.plot(fT_proposed, state_proposed['susceptible vaccinated female'], lw=1, color='blue')
ax.plot(fT_proposed, state_proposed['susceptible unvaccinated female'], lw=1, color='red')
ax.plot(fT_proposed, state_proposed['infected female'], lw=1, color='black')
"""

""" OK
# how does the number of vaccinated change with pulse vaccination rather than vaccinated once before 6 months
# normalized by total group size
ax.plot(fT_proposed, state_proposed['susceptible vaccinated female']/state_proposed['total female'], lw=1, color='blue')
ax.plot(fT_no, state_no['susceptible vaccinated female']/state_no['total female'], lw=1, color='red')
"""

""" OK
# how does the number of infected change with pulse vaccination rather than vaccinated once before 6 months
# normalized by total group size
ax.plot(fT_proposed, state_proposed['infected female']/state_proposed['total female'], lw=1, color='blue')
ax.plot(fT_no, state_no['infected female']/state_no['total female'], lw=1, color='red')
"""

""" OK
# how does the number of infected change with pulse vaccination for different number of pulses
# normalized by total group size
ax.plot(fT, state['infected female']/state['total female'], lw=1, color='green')
ax.plot(fT, state['infected female']/state['total female'], lw=1, color='blue')
ax.plot(fT1, state1['infected female']/state1['total female'], lw=1, color='black')
ax.plot(fT3, state3['infected female']/state3['total female'], lw=1, color='red')
ax.plot(fT_no, state_no['infected female']/state_no['total female'], lw=1, color='green')
"""

"""
# how does the number of infected change with pulse vaccination for different number of psi
# normalized by total group size
ax.plot(fT_B, state_B['infected female']/state_B['total female'], lw=1, color='blue')
ax.plot(fT_C, state_C['infected female']/state_C['total female'], lw=1, color='red')
ax.plot(fT_D, state_D['infected female']/state_D['total female'], lw=1, color='black')
"""
# TODO contour plot R0
plt.show()