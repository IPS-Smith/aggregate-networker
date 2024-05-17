import argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import distances
from subprocess import run
from subprocess import PIPE
from os import path
from os import system
from tqdm import tqdm
import re

from matplotlib.colors import LinearSegmentedColormap

import time
import warnings

warnings.filterwarnings("ignore", message="Unknown element  found for some atoms. These have been given an empty element record. If needed they can be guessed using MDAnalysis.topology.guessers.")
warnings.filterwarnings("ignore", message="Failed to guess the mass for the following atom types:")


class TPRreader:
    """
    Class object to read parameters from tpr file used to run MD simulation

    Methods:

        integrator_timestep:       return the timestep used by the integrator (dt) in ps

        trajectory_timestep:       return the timestep of the generated trajectory (nstxout-compressed) in ps

        trajectory_duration:       return the final duration of the generated trajectory (dt * nsteps) in ps

    """

    def __init__(self, filename):

        self.filename = filename

        # Check to ensure that filename is a string - if script has been called from the command line with -deffnm flag this shouldn't fail
        
        try:
            type(self.filename)==str
        except TypeError:
            print('Expected string input, got {} filename'.format(type(self.filename)))

    
    def integrator_timestep(self):
        """
        Method to obtain the integrator timestep used to generate the trajectory
        Value read from the dt variable within the tpr file

        Input
        -----
            self : variables from parent class

        Output
        ------
            integrator_timestep : float
                        value of dt parameter (ps)

        """

        dt_line = run('gmx dump -s {} | grep " dt "'.format(self.filename), shell=True, capture_output=True, text=True).stdout
        dt = float(dt_line.split()[2])

        return dt

    def nsteps(self):
        """
        Method to obtain the total number of steps in trajectory
        Value read from the nsteps variable within the tpr file

        Input
        -----
            self : variables from parent class

        Output
        ------
            nsteps : int

        """

        steps_line = run('gmx dump -s {} | grep " nsteps "'.format(self.filename), shell=True, capture_output=True, text=True).stdout
        nsteps = int(steps_line.split()[2])

        return nsteps

    def nstxout_compressed(self):
        """
        Method to obtain the number of steps between consecutive frames within the trajectory
        Value read from the nstxout-compressed variable within the tpr file

        Input
        -----
            self : variables from parent class

        Output
        ------
            framesteps : int

        """
        

        framesteps_line = run('gmx dump -s {} | grep " nstxout-compressed "'.format(self.filename), shell=True, capture_output=True, text=True).stdout

        framesteps = int(framesteps_line.split()[2])

        return framesteps

    def trajectory_timestep(self):
        """
        Method to obtain the timestep between output frames of the generated trajectory
        Value calculated using the nstxout-compressed * dt variables within the mdp file

        Input
        -----
            self : variables from parent class

        Output
        ------
            trajectory_timestep : float

        """


        return self.nstxout_compressed() * self.integrator_timestep()


    def trajectory_duration(self):

        """
        Method to obtain the total duration of simulation

        Input
        -----
            self : variables from parent class

        Output
        ------
            trajectory_duration : float

        """

        return self.nsteps() * self.integrator_timestep()

class PDBreader:
    """
    Class object to read PDB file from MD trajectory

    Methods:
        read_lines:         parse all lines of .pdb file

        check_mindist:       calculate the minimum distance between two MDAnalysis atom groups
    """

    def __init__(self, filename):

        self.filename = filename

        # Check to ensure that filename is a string - if script has been called from the command line with -deffnm flag this shouldn't fail
        
        try:
            type(self.filename)==str
        except TypeError:
            print('Expected string input, got {} filename'.format(type(self.filename)))

        if self.filename.split('.')[1] not in ["pdb", "gro"]:
            raise TypeError("PDBreader detected non pdb/gro type file: {}".format(self.filename))

        self.lines = self.read_lines()

        self.universe = mda.Universe(self.filename)

    def read_lines(self):
        """
        Performs a read lines method on a file of title "filename" and returns the result.

        Input
        -----
            self : variables from parent class

        Output
        ------
            lines : list of strings
                    list containing strings of each line from file

        """

        try:
            with open('{}'.format(self.filename),'r') as f1:
                lines = f1.readlines()
        except FileNotFoundError:
            print('Could not find file {}, please choose a valid filename.'.format(self.filename))

        return lines

    def check_mindist(self, groupA, groupB):
        """
        Calculates the minimum distance between two MDAnalysis atom groups (Angstroms)

        Input
        -----
            self   : variables from parent class
            groupA : MDAnalysis atom group
            groupB : MDAnalysis atom group

        Output
        ------
            min_dist : float
                    minimum distance between any pair of atoms from groupA and groupB
        
        """

        if type(groupA) != str:
            raise TypeError("Atom group A definition in PDBreader.check_mindist() not a string")
        if type(groupB) != str:
            raise TypeError("Atom group B definition in PDBreader.check_mindist() not a string")

        # Select atoms from universe according to user specified atom groups
        selectA = self.universe.select_atoms('{}'.format(groupA))
        selectB = self.universe.select_atoms('{}'.format(groupB))
        
        # Generate array of pairwise distances between atoms in each group
        distance_array = distances.distance_array(selectA.positions, selectB.positions, box=self.universe.dimensions)

        # Return the minimum distance value found between any pair of atoms
        return(distance_array.min())


class molecular_network:
    """
    Class object containing methods to generate and analyse a networkx graph object, created from a PDB file
    
    Input
    -----
    filename: str - name of PDB file to be analysed


    Requirements
    ------------
    PDBreader: class - used for manipulation of PDB file

    """

    def __init__(self, filename, atomGroups, cluster_distance):

        self.filename = filename

        try:
            type(self.filename)==str
        except TypeError:
            print('filename expected string input, got {}'.format(type(self.filename)))

        if self.filename.split('.')[1] not in ["pdb", "gro"]:
            raise TypeError("PDBreader detected non pdb/gro type file: {}".format(self.filename))

        self.cluster_distance = cluster_distance
        try:
            type(self.cluster_distance) in [int, float]
        except TypeError:
            print('cluster_distance expected numeric input, got {}'.format(type(self.cluster_distance)))

        self.atomGroups = atomGroups

        self.system = PDBreader(self.filename)

        self.dGraph = self.dist_graph()

        self.dGraph_filtered = self.dist_graph_filtered(self.cluster_distance)

        

    def dist_graph(self):
        """
        Function to return the distance graph between selected molecules (atomGroups) within a system (MDA universe)
        
        Input
        -----
            universe    :  MDAnalysis universe object
            atomGroups  :  list of MDAnalysis atom groups defining each molecule

        Output
        ------
            network : networkx graph object
                    Graph object where nodes corresponding to molecules (referenced by their index within the atomGroups list)
                    and edges are given the value of the minimum distance between each molecule pair
        """

        network = nx.Graph()

        #populate graph with nodes
        network.add_nodes_from(range(len(self.atomGroups)))

        # iterate over all but the final atom group - this ensures searching for group pairs doesn't look past the end of the list

        print('Clustering...')
        for groupA_ID, groupA in tqdm(enumerate(self.atomGroups[:-1])):
            

            # iterate over new group pairs only - avoids double calculation of edges
            for groupB_ID, groupB in enumerate(self.atomGroups[groupA_ID+1:]):
                groupB_ID += groupA_ID+1
                AB_dist = self.system.check_mindist(groupA, groupB)
                network.add_edge(groupA_ID, groupB_ID, dist=AB_dist)

                ## validate that groups and distances match expected values from VMD visualisation
                # print(groupA_ID, groupB_ID, AB_dist)
            
        return network
    
    def dist_graph_filtered(self, threshold_val=4.7):
        """
        Function to return dist_graph including only the edges that correspond to molecular distances below the threshold value.
        """

        network_original = self.dGraph
        network_filtered = nx.Graph()
        
        for node in network_original:
            network_filtered.add_node(node)

        for groupA_ID, groupB_ID, AB_edge in network_original.edges(data=True):
            if AB_edge['dist'] <= threshold_val:
                network_filtered.add_edge(groupA_ID, groupB_ID)

        return network_filtered

    def calc_cluster_sizes(self):
        """
        Takes a NetworkX graph object (self.dGraph) and returns a list containing the number of nodes
        in each connected cluster of the graph.
        
        Input
        -----
            self.dGraph_filtered (nx.Graph): NetworkX graph object of contacting molecules
            
        Output
        ------
            List[int]: A list where each element is the number of nodes in a connected cluster.
        """

        cluster_sizes = []
        
        # Iterate over each connected component
        # connected_components returns a generator of sets, each set being a connected component
        for component in nx.connected_components(self.dGraph_filtered):
            # Append the size of this component to the list
            cluster_sizes.append(len(component))
        
        return cluster_sizes
    
    def calc_cluster_components(self):
        """
        Takes a NetworkX graph object and returns a list of sets, where each set contains
        the node labels of a connected component in the graph.
        
        Input
        -----
            self.dGraph_filtered (nx.Graph): NetworkX graph object of contacting molecules
            
        Output
        ------
            List[Set]: A list where each element is a set of node labels in a connected component.
        """

        # Get all connected components and convert each to a set of node labels
        components = list(nx.connected_components(self.dGraph_filtered))

        return components
    
    def save_clusters(self, fileOUT, current_time):
        """
        Save cluster components and formation time to a text file named clusters.txt

        Input
        -----
            fileOUT (str): name of file to which cluster data will be saved
            current_time  (float): simulation time (ps) at which new data to be added was calculated
        
        Output
        ------
            None

        TODO:   Need to modify meta_dict[sub_dict].append calls to only add a new element to the sub_dict list if the cluster is not already found in the list
                This will go hand in hand with a new function load_clusters that reads in the fileOUT from this function, and populates the meta_dict with obtained data
        """

        def cluster_in_dict(loaded_meta_dict, sub_dict, new_cluster):
            """
            Function to check whether a cluster of molecules exists within the meta_dict[sub_dict]['cluster'] list
            Since NetworkX returns an unordered set of components within each cluster the function must check
            whether the cluster components are present in any order within an existing entry
            """


            cluster_components_new = sorted(map(int, re.findall(r'\d+', new_cluster)))

            for existing_cluster_ID, existing_cluster in enumerate(loaded_meta_dict[sub_dict]['clusters']):
                cluster_components_existing = sorted(map(int, re.findall(r'\d+', existing_cluster)))

                if cluster_components_new == cluster_components_existing:
                    return True, existing_cluster_ID
                
            return False, None


        fileExists = path.exists(fileOUT)

        cluster_components = self.calc_cluster_components()
        cluster_sizes = self.calc_cluster_sizes()

        

        # if existing data file found, load into meta dict
        if fileExists:
            meta_dict = self.load_clusters(fileOUT)
        
        # otherwise initialise meta dict
        else:
            meta_dict = {}

        # for molecule cluster found during this call of the script
        for cluster_ID, cluster in enumerate(cluster_components):
            
            # determine which type of multimer cluster belongs to (1-mer, 2-mer... etc)
            sub_dict = '{}-mers'.format(cluster_sizes[cluster_ID])

            # if no molecule cluster of equivalent size has been found in this script call, or in the existing data file
            # initialise sub dict in meta dict to contain data on cluster types, formation time and break time
            if sub_dict not in meta_dict:
                meta_dict[sub_dict] = {'clusters': [],
                                       'formed': [],
                                       'broken': []}

            cluster = str([x for x in cluster])
            cluster_exists, existing_cluster_ID = cluster_in_dict(meta_dict, sub_dict, cluster)

            print(existing_cluster_ID)

            if cluster_exists:
                # these entries need work to ensure that formation times in existing data files are not overwritten 
                # the break time entry will work, however if clusters persist beyond the end of the simulation, the data file will imply that they break at the final frame
                meta_dict[sub_dict]['broken'][existing_cluster_ID] = current_time
            
            else:

                # append data from this script call to the list of multimer info in sub dict
                meta_dict[sub_dict]['clusters'].append(cluster)

                # these entries need work to ensure that formation times in existing data files are not overwritten 
                # the break time entry will work, however if clusters persist beyond the end of the simulation, the data file will imply that they break at the final frame
                meta_dict[sub_dict]['formed'].append(current_time)
                meta_dict[sub_dict]['broken'].append(current_time)


        # initialise list of lines to be saved to the output data file
        save_lines = []

        # for each multimer type
        for sub_dict in meta_dict:

            # append the multimer header lines
            save_lines.append(sub_dict.ljust(14) + '\n' + '-'*50 + '\n')

            # for each cluster of the given multimer type
            for cluster_ID, cluster in enumerate(meta_dict[sub_dict]['clusters']):
                # append cluster component line
                save_lines.append(cluster+'\n')
                # append cluster formation and break times
                save_lines.append("Formed: {}".format(meta_dict[sub_dict]['formed'][cluster_ID]).ljust(25) + \
                                  "Broken: {}".format(meta_dict[sub_dict]['broken'][cluster_ID]).rjust(25) + '\n')
            
            save_lines.append('\n')

       
        # add information relating cluster component IDs back to the original choice of atom groups
        save_lines.append('Atom Groups \n'  + '-'*50 + '\n')
        for groupID, group in enumerate(self.atomGroups):
            save_lines.append('{}:'.format(groupID).ljust(8) + group + '\n')
        
        # save new data file
        with open(fileOUT, 'w') as f:
            for section in save_lines:
                f.writelines(section)



    def load_clusters(self, fileIN):
        """
        Load cluster components from data file generated by save_clusters

        Input
        -----
            fileIN (str): name of file from which cluster data will be read
        
        Output
        ------
            meta_dict (dict): dictionary containing sub dictionaries that each contain the multimer cluster components

        """

        def get_lines(FILE):
            """
            Performs a read lines method on a file of title "filename" and returns
            the result.
            
            Output
            ------
            lines : list of strings
                list containing strings of each line from file
            """
            try:
                type(FILE) == str
            except TypeError:
                print('Load clusters expected data file of type string, got {} filename'.format(type(FILE)))
            try:
                with open('{}'.format(FILE),'r') as f1:
                    lines = f1.readlines()
            except FileNotFoundError:
                print('Load clusters could not find file "{}", please choose a valid file name.'.format(FILE))
            return lines
        
        loadlines = get_lines(fileIN)

        meta_dict = {}

        for lineID, line in enumerate(loadlines):
            if "mers" in line:
                sub_dict = line.split()[0]
                meta_dict[sub_dict] = {'clusters': [],
                                       'formed': [],
                                       'broken': []}
            
            elif "Atom Groups" in line:
                break
            
            elif "Formed" in line:
                linesplit = line.split()
                meta_dict[sub_dict]['formed'].append(linesplit[1])
                meta_dict[sub_dict]['broken'].append(linesplit[3])

                prev_linesplit = loadlines[lineID-1].split(']')
                cluster = prev_linesplit[0] + ']'
                meta_dict[sub_dict]['clusters'].append(cluster)

            

        return meta_dict



if __name__ == "__main__":

    start_time = time.time()

    parser = argparse.ArgumentParser(
                prog='aggregate-networker',
                description='Calculates the connections between molecules within a static frame of a simulation (i.e. pdb file) \
                             using the networkx package. The script will print and return "True" if the maximum number of molecules \
                             in any subgraph equals or exceeds the given threshold number.',
                epilog='')

    parser.add_argument('-f', '--fileName', type=str, help='Name of the structure file to be analysed (.pdb or .gro allowed)')
    parser.add_argument('-g', '--atomGroups', nargs='*', help='List of atom selections to define the group of molecules that will make up the network (MDAnalysis atom selection)')
    parser.add_argument('-tv','--threshVal', type=float, help='Distance between nodes (atom groups / molecules) below which they are considered bound')

    args = parser.parse_args()

    

    # tprFile = TPRreader('../{}.tpr'.format(args.fileName))
    # trajectory_dt = tprFile.trajectory_timestep()

    trajectory_dt = 2000
    analysis_dt = 2000
    sim_time = 1000000

    n_steps = int(sim_time / analysis_dt)

    data_array = np.empty(n_steps, dtype=object)

    init_time = 0

    for i in range(n_steps):

        data_array[i] = {}

        time_curr = init_time + i*analysis_dt

        system("echo '2' | gmx trjconv -f ../{0}.xtc -s ../{0}.tpr -dump {1} -o {0}-{1}.pdb -b {2}".format(args.fileName, time_curr, time_curr-trajectory_dt))
        
        print(args.fileName)
        print("{0}-{1}.pdb".format(args.fileName, time_curr))

        network = molecular_network("{}-{}.pdb".format(args.fileName, time_curr), args.atomGroups, args.threshVal)

        cluster_sizes = network.calc_cluster_sizes()

        for size in cluster_sizes:
            data_array[i]['{}'.format(size)] = data_array[i].get('{}'.format(size), 0) + 1
        

    # Determine the number of dictionaries and the range of keys
    num_dicts = len(data_array)
    keys = list(map(str, range(1, 21)))
    num_keys = len(keys)

    # Create a 2D numpy array filled with np.nan (or use another fill value if preferred)
    array = np.full((num_dicts, num_keys), 0)

    # Populate the array with the values from the dictionaries
    for dataindex, datadict in enumerate(data_array):
        for clustersize, numclusters in datadict.items():
            array[dataindex, int(clustersize) - 1] = numclusters

    colors = ["#ffffff", "#004488"]
    colormap = LinearSegmentedColormap.from_list("custom_cmap", colors)

    # Plot the 2D color map
    plt.figure(figsize=(10, 6))
    plt.imshow(array.T, aspect='auto', cmap=colormap, interpolation='none')
    plt.colorbar(label='Number of Clusters')

    # Set x and y axis labels
    plt.xlabel('Analysis Step')
    plt.ylabel('Cluster Size')

    # Set the y-axis ticks and labels
    plt.yticks(ticks=np.arange(num_keys), labels=keys)

    plt.gca().invert_yaxis()

    plt.savefig('test.png')
    plt.show()
    


    print("--- %s seconds ---" % (time.time() - start_time))

    # fig = plt.figure(figsize=(8, 6))

    # nx.draw(
    #     contact_network,
    #     node_size=[contact_network.degree()[node] for node in contact_network],
    #     node_color='#994455',
    #     with_labels=True,
    # )

    # plt.show()


    


