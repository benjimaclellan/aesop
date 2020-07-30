from scripts.GA_Adam_optimization import generate_data_GA_Adam_comparison, load_and_output_data_GA_Adam_comparison, \
                                         generate_adam_convergence_data, \
                                         display_adam_convergence_data, \
                                         display_initial_pop, \
                                         get_noise_setting_benchmark, \
                                         compare_noise_setting_benchmark


if __name__ == "__main__":
    compare_noise_setting_benchmark()
    # get_noise_setting_benchmark()
    # display_initial_pop()
    # generate_adam_convergence_data()
    # display_adam_convergence_data()
    # generate_data_GA_Adam_comparison(with_noise=True, resample_per_individual=True)
    # load_and_output_data_GA_Adam_comparison(with_noise=True)

