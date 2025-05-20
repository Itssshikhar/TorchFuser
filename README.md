Collaborators can clone the main repository along with its submodules:

`git clone --recurse-submodules <superproject-url>`

If they've already cloned the repository without submodules, they can initialize and update the submodules:

`git submodule update --init --recursive`
