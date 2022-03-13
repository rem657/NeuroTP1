import matplotlib as plt
import os

from src.question2 import HHModel, display_eigenvalues_phase, display_eigenvalues_to_I, show_potential_series
from src.question_3 import CoupleHH, question_3_a, question_3_b_1, question_3_b_2, question_3_b_3


def question_1():
	pass


def question_2():
	vmin = -65
	vmax = -40
	display_eigenvalues_to_I(HHModel(), vmin, vmax, numtick=10_000, i_max=160, save=True)
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



