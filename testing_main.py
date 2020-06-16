from scripts.parameter_optimization_analysis import run_case_fitness_analysis_benchmarking, \
                                                    optimization_comparison_GA_Adam_both, \
                                                    adam_plot_convergence, \
                                                    unearth_np_runtimeWarnings

from scripts.GA_Adam_optimization import generate_data, load_and_output_data

if __name__ == "__main__":
    # generate_data()
    load_and_output_data()
    # unearth_np_runtimeWarnings(bug_name='divide by 0')
    # adam_plot_convergence()
    # optimization_comparison_GA_Adam_both()
    # run_case_fitness_analysis_benchmarking()