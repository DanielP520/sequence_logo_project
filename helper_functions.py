import pdb_parser
import pythreejs as p3js
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def extract_info_pdb(pdb_file, chain_id) -> list:
    all_Ca = []
    with open(pdb_file, "r") as file:
        found_chain = False
        for line in file:
            dict_of_atoms = {}
            if "ATOM " not in line:
                continue
            if "TERM" in line:
                break
            parsed_line = pdb_parser.PDBLineParser(line)
            parsed_line.parse_line()
            if parsed_line.chain_identifier != chain_id and found_chain:
                break
            if parsed_line.chain_identifier == chain_id and parsed_line.atom_name == 'CA':
                dict_of_atoms['X'] = parsed_line.x_cord
                dict_of_atoms['Y'] = parsed_line.y_cord
                dict_of_atoms['Z'] = parsed_line.z_cord
                dict_of_atoms['chain'] = parsed_line.chain_identifier
                dict_of_atoms['residue_index'] = parsed_line.residue_sequence_number
                dict_of_atoms['AA'] = parsed_line.residue_name
                dict_of_atoms['atom_type'] = parsed_line.atom_name
                dict_of_atoms['file'] = pdb_file
                found_chain = True
                all_Ca.append(dict_of_atoms)

    return all_Ca





def plot_3d_scatter_matplotlib(df):
    # Create a new figure and a 3D axis
    points_data = df[['X',"Y","Z"]].values.tolist()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract x, y, and z coordinates from the points data
    x = df.iloc[:, 0]
    y = df.iloc[:, 1]
    z = df.iloc[:, 2]
    # Create the 3D scatter plot
    ax.scatter(x, y, z, c='r', marker='o')

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.ion()
    # Show the plot
    plt.show()