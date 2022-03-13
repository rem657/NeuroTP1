import matplotlib as plt
import os
from src.question_1 import display_trajectories, display3D_phaseplane, Q1display_eigenvalues_to_I, bifurcation_diagram
from src.question2 import HHModel, display_eigenvalues_phase, display_eigenvalues_to_I, show_potential_series, Q2Bifurcation_diagram
from src.question_3 import CoupleHH, question_3_a, question_3_b_1, question_3_b_2, question_3_b_3


def question_1():
	fig_folder = "figures/Q1/"
	os.makedirs(fig_folder, exist_ok=True)
	display_trajectories()
	imax = 10
	vmin = -3.5
	vmax = 3.5
	display3D_phaseplane(imax, vmin, vmax, save=True)
	Q1display_eigenvalues_to_I(vmin, vmax, numtick=1000, i_max=7, save=True)
	bifurcation_diagram()


def question_2():
	vmin = -65
	vmax = -40
	display_eigenvalues_to_I(HHModel(), vmin, vmax, numtick=10_000, i_max=160, save=True)
	print("Vous avez le temps d'allez vous chercher un café en attendant pour la question 2 b)")
	Q2Bifurcation_diagram(HHModel(t_end=250.0), 160, 500)
	display_eigenvalues_phase(HHModel(), vmin, vmax, numtick=10_000, save=True)
	show_potential_series()


def question_3():
	plt.rcParams.update({'font.size': 12})
	CoupleHH().show_weights_in_func_of_g_syn()
	question_3_a()
	question_3_b_1()
	question_3_b_2_dict = question_3_b_2()
	question_3_b_3(question_3_b_2_dict)


if __name__ == '__main__':
	os.makedirs("figures/", exist_ok=True)
	question_1()
	question_2()
	question_3()



