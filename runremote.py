import paramiko
import time
from scp import SCPClient
import shutil
import asope_main
import matplotlib.pyplot as plt
from assets.functions import plot_individual, load_experiment
import time 

if __name__ == '__main__': 
    runremote = False
    filename = 'results/' + time.strftime("%Y_%m_%d-%H_%M_%S")
    
    foldername = 'ASOPE_V2_SingleSetup'
    mainfile = 'asope_main.py'
    localpath = '/home/benjamin/Documents/INRS - Code/'
    remotepath = '~/'

    if not runremote:
        asope_main.main(filename)
        
    else:
        # update this to hide key later
        host = '10.50.113.131'
        user = 'MacLellan'
        password = 'uopsim'
    
        with paramiko.SSHClient() as client:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect('10.50.113.131', username='MacLellan', password='uopsim')
            
            # zip the folder
            shutil.make_archive(localpath+foldername, 'zip', localpath+foldername)
            
            # send zip to remote server
            with SCPClient(client.get_transport()) as scpclient:
                scpclient.put( localpath+foldername+'.zip', remotepath)
            
            stdin, stdout, stderr = client.exec_command('unzip -o {} -d {}'.format(remotepath + foldername + '.zip', foldername))
#            for line in stdout:
#                print(line.rstrip())
        
            stdin, stdout, stderr = client.exec_command("python -u {} '{}'".format(remotepath + foldername + '/' + mainfile, filename))
            for line in stdout:
                print(line.rstrip())
            time.sleep(2)
            with SCPClient(client.get_transport()) as scpclient:
                scpclient.get( remotepath+foldername+'/'+filename, localpath+foldername)
        
    
    (experiment, env) = load_experiment(filename)
    fitness = env.fitness()
    plot_individual(env, fitness)
    plt.show()
    
    
