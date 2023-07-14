import pdb_parser
import pythreejs as p3js

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


def plot_3d_points(df):
    # Extract the x, y, z coordinates from the DataFrame
    points = df[['X', 'Y', 'Z']].values

    # Create BufferGeometry
    geometry = p3js.BufferGeometry(
        attributes={
            'position': p3js.BufferAttribute(array=points, normalized=False)
        }
    )

    # Create Material
    material = p3js.PointsMaterial(size=0.05, color='red')

    # Create Points object
    points_object = p3js.Points(geometry=geometry, material=material)

    # Create Scene
    scene = p3js.Scene(children=[points_object])

    # Create Renderer
    renderer = p3js.Renderer(scene=scene, controls=[p3js.OrbitControls()])

    # Display the Renderer
    renderer