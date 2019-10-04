 #%% UDR
#    std_udr = analysis_udr(at, exp, env, verbose=True)
#    
#    ## Monte Carlo
#    mu_mc, std_mc = analysis_mc(at, exp, env, 10**3, verbose=True)
#
#    # LHA
#    H0, eigenvalues, eigenvectors, basis_names = analysis_lha(at, exp, env, verbose=True)
#    
    #%%
#    plt.figure()
#    plt.stem(np.diag(H0)/np.max(np.abs(np.diag(H0))),label='lha')
#    plt.stem(std_udr/np.max(std_udr), label='udr', linefmt='-gx')
#    plt.legend()
#    plt.show()
#    
#    save_class('testing/experiment_example', exp)
#    save_class('testing/environment_example', env)
    
    
    
##    ## PLOTTING FROM HERE ON
#    plt.figure()
#    g = seaborn.heatmap((H0))
#    g.set_xticklabels(basis_names[1], rotation=30)
#    g.set_yticklabels(basis_names[1], rotation=60)
#
#
#    plt.figure()
#    plt.plot(env.t, P(field))
#    plt.plot(env.t, P(env.field0))
#    plt.show()


#    fig, ax = plt.subplots(eigenvectors.shape[1], 1, sharex=True, sharey=True)
#    for k in range(0, eigenvectors.shape[1]):
#        ax[k].stem(eigenvectors[:,k], linefmt='teal', markerfmt='o', label = 'Eigenvalue {} = {:1.3e}'.format(k, (eigenvalues[k])))
#        ax[k].legend()
#    plt.ylabel('Linear Coefficient')
#    plt.xlabel('Component Basis')
#    plt.xticks([j for j in range(0,eigenvectors.shape[0])], labels=at_name)
#
#    stop = time.time()
#    print("T: " + str(stop-start))
#
#    plt.figure()
#    xval = np.arange(0,eigenvalues.shape[0],1)
#    plt.stem(xval-0.05, ((np.diag(H0))),  linefmt='salmon', markerfmt= 'x', label='Hessian diagonal')
#    plt.stem(xval+0.05, (eigenvalues), linefmt='teal', markerfmt='o', label='eigenvalues')
#    plt.xticks(xval)
#    plt.xlabel('Component Basis')
#    plt.ylabel("Value")
#    plt.title("Hessian Spectrum")
#    plt.legend()
#    plt.show()





#    fig_log, ax_log = plt.subplots(1,1, figsize=[8,6])
#    ax_log.plot(log['gen'], log["Best [fitness, variance]"], label='Maximum', ls='-', color='salmon', alpha=1.0)
#    ax_log.plot(log['gen'], log["Average [fitness, variance]"], label='Mean', ls='-.', color='blue', alpha=0.7)
#    ax_log.legend()
#    ax_log.set_xlabel('Generation')
#    ax_log.set_ylabel(r'Fitness, $\mathcal{F}(\mathbf{x})$')