# Automated Search of Optical Processing Experiments
ASOPE is a Python package for the inverse design of optical systems in the time-energy degree-of-freedom.
This README is a ever-changing document - please contact the authors for current status.

## New in v.3.0
Complete overhaul of the simulation library - which has added layers of abstraction and new extensibility.
New noise analysis, new components in the library, improved topology optimization, more methods of parameter optimization. 
## Getting Started
This package aims to design optical systems which accomplishes an user-defined goal. 
A system is described by a set of components, connected in a specific way and with specific control variables.

### Prerequisites

See [`requirements.txt`](../requirments.txt) for full, more up-to-date list of all dependencies.
Major packages used are:

`networkx` - Graph and network manipulation package, used for representing an experimental setup

`multiprocess` - Multiprocessing in Python, used to improve computation time with the Genetic Algorithm

`numpy` - Standard scientific/numerical package in Python

`scipy` - Scientific computing functions, including minimization

`matplotlib` - Plotting and visualization

`autograd` - Automatic differentiation of Python functions

### Installing

To install the ASOPE package, clone the repository from [Github](https://github.com/) at [https://github.com/benjimaclellan/ASOPE.git](https://github.com/benjimaclellan/ASOPE.git). 
All the prerequisite packages can be install from the `requirements.txt` file. 
You can install all packages via pip, or using `pip install requirements.txt`. 
A virtual environment is recommended.

## Running with GCS
A 16-CPU, 16 GB RAM virtual machine (VM) is reserved for use on Google Cloud Services (GCS).
GCS charges by the second used, so please turn the VM off after running batches.
To start, navigate to Google Cloud Console > Compute Engine > VM Instances, select the desired VM and Start. 
If no issues arise, an external IP address will be provided which can be used to SSH into the VM.
Click SSH to open an SSH terminal in your browser (avoids the need to set up SSH keys with your personal computer).
The machine is running Debian 10, and has minimal programs installed.
Git, Python, SSH, and a few other necessary tools have been installed on the instance.
Once access to the VM is started, there should be two main folders: ~/ASOPE and  ~/asope_data.
ASOPE stores the code repository, which uses a Git developer token to sync new commits/branches.
Please never commit or push commits to the origin (Github) from the GCS VM. 
The second folder, ~/asope_data, is the default storage location for ASOPE batch runs.
To keep costs low, we use Google Data Buckets - which have almost limitless storage, but higher latency (which is not a problem here).
The GCS project has one Data Bucket created, also called asope_data.
This data bucket can be mounted to the VM easily - but it must be done everytime the VM is started.
To mount the data bucket to the data folder on the VM, run 
`gcs asope_data asope_data`
from the `~/` folder.
If the bucket mounts successfully, any results from the repository will automatically be added to the bucket, which can be easily explored/downloaded via the browser.
Alternatively, SCP or other file transfer protocols can be used, though more work is needed to set them up.

To update the code to the latest version, change directories to the repository `cd ASOPE`. 
Here you can run `git` commands.
Again, please don't commit or push changes from the VM.
To hard reset the local repo to the remote (Github), run `git fetch` followed by `git reset --hard origin/branch-to-checkout`.
You can also use `git stash` to save any changes and `git pop` them back after pulling the changes.
Now you can run scripts with `python3 script-to-run.py` and edit with `nano script-to-edit.py`.
If new packages are required, install with `pip3 install package-to-install`.
Once the batch is finished running, ensure all desired data is saved in `~/asope_data` and download from the GCS Storage page, or with SCP.
Finally, exit the SSH shell and turn off the VM instance.

## Authors
* **Benjamin MacLellan** - [Email](benjamin.maclellan@emt.inrs.ca)
* **Piotr Roztocki**
* **Julie Belleville**
* **Kaleb Ruscitti**

## License
Please see [LICENSE.md](../LICENSE.md)


 
