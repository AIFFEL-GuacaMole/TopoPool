from copy import deepcopy
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from torch_geometric.data import Data
import torch
import numpy as np
import networkx as nx

# allowable multiple choice node and edge features
allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)) + ['misc'],
    'possible_chirality_list': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_SQUAREPLANAR',
        'CHI_TRIGONALBIPYRAMIDAL',
        'CHI_OCTAHEDRAL',
        'CHI_OTHER'
    ],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list': [
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'misc'
    ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring_list': [False, True],
    'possible_bond_type_list': [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
        'misc'
    ],
    'possible_bond_stereo_list': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'possible_is_conjugated_list': [False, True],
}


def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except Exception:
        return len(l) - 1


def atom_to_feature_vector(atom):
    """
    Converts rdkit atom object to feature list of indices
    :param mol: rdkit atom object
    :return: list
    """
    atom_feature = [
        safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
        allowable_features['possible_chirality_list'].index(str(atom.GetChiralTag())),
        safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
        safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
        safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
        safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
        safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
        allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
        allowable_features['possible_is_in_ring_list'].index(atom.IsInRing()),
    ]
    return atom_feature


def get_atom_feature_dims():
    return list(map(len, [
        allowable_features['possible_atomic_num_list'],
        allowable_features['possible_chirality_list'],
        allowable_features['possible_degree_list'],
        allowable_features['possible_formal_charge_list'],
        allowable_features['possible_numH_list'],
        allowable_features['possible_number_radical_e_list'],
        allowable_features['possible_hybridization_list'],
        allowable_features['possible_is_aromatic_list'],
        allowable_features['possible_is_in_ring_list']
    ]))


def bond_to_feature_vector(bond):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    bond_feature = [
        safe_index(allowable_features['possible_bond_type_list'], str(bond.GetBondType())),
        allowable_features['possible_bond_stereo_list'].index(str(bond.GetStereo())),
        allowable_features['possible_is_conjugated_list'].index(bond.GetIsConjugated()),
    ]
    return bond_feature


def get_bond_feature_dims():
    return list(map(len, [
        allowable_features['possible_bond_type_list'],
        allowable_features['possible_bond_stereo_list'],
        allowable_features['possible_is_conjugated_list']
    ]))


def atom_feature_vector_to_dict(atom_feature):
    [atomic_num_idx,
     chirality_idx,
     degree_idx,
     formal_charge_idx,
     num_h_idx,
     number_radical_e_idx,
     hybridization_idx,
     is_aromatic_idx,
     is_in_ring_idx] = atom_feature

    feature_dict = {
        'atomic_num': allowable_features['possible_atomic_num_list'][atomic_num_idx],
        'chirality': allowable_features['possible_chirality_list'][chirality_idx],
        'degree': allowable_features['possible_degree_list'][degree_idx],
        'formal_charge': allowable_features['possible_formal_charge_list'][formal_charge_idx],
        'num_h': allowable_features['possible_numH_list'][num_h_idx],
        'num_rad_e': allowable_features['possible_number_radical_e_list'][number_radical_e_idx],
        'hybridization': allowable_features['possible_hybridization_list'][hybridization_idx],
        'is_aromatic': allowable_features['possible_is_aromatic_list'][is_aromatic_idx],
        'is_in_ring': allowable_features['possible_is_in_ring_list'][is_in_ring_idx]
    }

    return feature_dict


def bond_feature_vector_to_dict(bond_feature):
    [bond_type_idx,
     bond_stereo_idx,
     is_conjugated_idx] = bond_feature

    feature_dict = {
        'bond_type': allowable_features['possible_bond_type_list'][bond_type_idx],
        'bond_stereo': allowable_features['possible_bond_stereo_list'][bond_stereo_idx],
        'is_conjugated': allowable_features['possible_is_conjugated_list'][is_conjugated_idx]
    }

    return feature_dict


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def boltzmann_average(quantities, energies, k=8.617333262145e-5, temperature=298):
    # k: Boltzmann constant in eV/K
    assert len(quantities) == len(energies)
    if len(energies) == 1:
        return quantities[0]

    return np.sum(softmax(-np.asarray(energies) / k / temperature) * np.asarray(quantities))


def reorder_molecule_idx(molecule_idx):
    previous_idx = molecule_idx[0].item()
    cursor = 0
    new_molecule_idx = torch.zeros_like(molecule_idx).long()

    for i, idx in enumerate(molecule_idx[1:], 1):
        if idx.item() != previous_idx:
            cursor += 1
            previous_idx = idx.item()
        new_molecule_idx[i] = cursor

    return new_molecule_idx


def canonicalize_3d_mol(mol_smiles, mol_3d):
    def get_node_features(atomic_numbers):
        node_features = np.zeros((len(atomic_numbers), 100))
        for node_index, node in enumerate(atomic_numbers):
            features = np.zeros(100)  # one-hot atomic numbers
            features[node] = 1.
            node_features[node_index, :] = features
        return np.array(node_features, dtype=np.float32)

    def get_reindexing_map(g1, g2, use_node_features=True):
        if use_node_features:
            nm = nx.algorithms.isomorphism.generic_node_match(['Z'], [None], [np.allclose])
            gm = nx.algorithms.isomorphism.GraphMatcher(g1, g2, node_match=nm)
        else:
            gm = nx.algorithms.isomorphism.GraphMatcher(g1, g2)
        assert gm.is_isomorphic()  # THIS NEEDS TO BE CALLED FOR gm.mapping to be initiated
        idx_map = gm.mapping

        return idx_map

    def mol_to_nx(mol):
        m_atoms = mol.GetAtoms()
        m_atom_numbers = [a.GetAtomicNum() for a in m_atoms]
        adj = np.array(Chem.rdmolops.GetAdjacencyMatrix(mol), dtype=int)

        node_feats = get_node_features(m_atom_numbers)
        node_feats_dict = {j: node_feats[j] for j in range(node_feats.shape[0])}

        g = nx.Graph(adj)
        nx.set_node_attributes(g, node_feats_dict, 'Z')

        return g

    try:
        mol_smiles = Chem.AddHs(mol_smiles)
        Chem.AllChem.EmbedMolecule(mol_smiles)
        mol_smiles.GetConformer()
        # check to make sure a conformer was actually generated
        # sometime conformer generation fails
    except Exception:
        print('Failed to embed conformer')
        return None

    g_smiles = mol_to_nx(mol_smiles)
    g_3d = mol_to_nx(mol_3d)

    idx_map = get_reindexing_map(g_smiles, g_3d, use_node_features=True)

    new_3d_mol = deepcopy(mol_smiles)
    xyz_coordinates = mol_3d.GetConformer().GetPositions()
    for k in range(new_3d_mol.GetNumAtoms()):
        x, y, z = xyz_coordinates[idx_map[k]]
        new_3d_mol.GetConformer().SetAtomPosition(k, Point3D(x, y, z))

    return new_3d_mol


def get_chosen_descriptors():
    """Get list of RDKit descriptors to calculate"""
    chosen_descriptors = [
        'MolWt', 'MolLogP', 'MolMR', 'TPSA', 
        'HeavyAtomCount', 'NumRotatableBonds',
        'NumHAcceptors', 'NumHDonors', 'NumAromaticRings',
        'FractionCSP3', 'NumAliphaticRings', 'BertzCT'
    ]  # Using a subset of descriptors for efficiency
    return chosen_descriptors

def calculate_rdkit_features(mol):
    """Calculate RDKit molecular descriptors"""
    calculator = MolecularDescriptorCalculator(get_chosen_descriptors())
    features = calculator.CalcDescriptors(mol)
    return torch.tensor(features, dtype=torch.float)


def mol_to_data_obj(mol):
    # atoms
    atom_features_list = []
    atom_features_list.extend(
        atom_to_feature_vector(atom) for atom in mol.GetAtoms()
    )
    x = torch.tensor(np.asarray(atom_features_list), dtype=torch.long)

    # bonds
    if len(mol.GetBonds()) <= 0:
        num_bond_features = 2
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
    else:
        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.extend(([i, j], [j, i]))
            edge_feature = bond_to_feature_vector(bond)
            edge_attr.extend((edge_feature, edge_feature))
        edge_index = torch.tensor(np.asarray(edge_index), dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.asarray(edge_attr), dtype=torch.long)

    # # coordinates
    # pos = mol.GetConformer().GetPositions()
    # pos = torch.from_numpy(pos).float()

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data

class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
        Matching original format:
        g: networkx graph
        label: integer graph label
        node_tags: list of integer node tags
        node_features: torch float tensor (will be created from node_tags)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0
        self.max_neighbor = 0

# def mol_to_s2v_graph(mol, label):
#     """Convert molecular graph to original S2VGraph format"""
#     try:
#         # Create networkx graph
#         g = nx.Graph()
        
#         # Process atoms (nodes)
#         node_tags = []
#         feat_dict = {}  # Maps atom features to tag indices
        
#         for atom_idx, atom in enumerate(mol.GetAtoms()):
#             g.add_node(atom_idx)
            
#             # Create atom feature tuple
#             feat = tuple(atom_to_feature_vector(atom))
            
#             # Get or create tag for this feature
#             if feat not in feat_dict:
#                 feat_dict[feat] = len(feat_dict)
#             node_tags.append(feat_dict[feat])
        
#         # Process bonds (edges)
#         for bond in mol.GetBonds():
#             i = bond.GetBeginAtomIdx()
#             j = bond.GetEndAtomIdx()
#             g.add_edge(i, j)
        
#         # Create initial S2VGraph
#         graph = S2VGraph(g=g, label=label, node_tags=node_tags)
        
#         # Add neighbors
#         graph.neighbors = [[] for _ in range(len(g))]
#         for i, j in g.edges():
#             graph.neighbors[i].append(j)
#             graph.neighbors[j].append(i)
        
#         # Calculate max_neighbor
#         degree_list = [len(neigh) for neigh in graph.neighbors]
#         graph.max_neighbor = max(degree_list)
        
#         # Create edge_mat
#         edges = [[i, j] for i, j in g.edges()]
#         edges.extend([[j, i] for i, j in g.edges()])  # Add reverse edges
#         graph.edge_mat = torch.LongTensor(edges).transpose(0, 1)
        
#         # Create one-hot node features from tags
#         unique_tags = len(feat_dict)
#         graph.node_features = torch.zeros(len(node_tags), unique_tags)
#         graph.node_features[range(len(node_tags)), node_tags] = 1
#     except:
#         import pdb; pdb.set_trace()
#     return graph


def mol_to_s2v_graph(mol, label):
    """Convert molecular graph to original S2VGraph format"""
    try:
        if len(mol.GetBonds()) == 0:
            print(f"Skipped molecule with no edges: {Chem.MolToSmiles(mol)}")
            return None

        g = nx.Graph()
        
        # Store raw atom features directly
        node_features = []
        
        for atom_idx, atom in enumerate(mol.GetAtoms()):
            g.add_node(atom_idx)
            # Get raw atom features
            node_features.append(atom_to_feature_vector(atom))
        
        # Process bonds (edges)
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            g.add_edge(i, j)
        
        # Create initial S2VGraph
        graph = S2VGraph(g=g, label=label)
        
        # Add neighbors
        graph.neighbors = [[] for _ in range(len(g))]
        for i, j in g.edges():
            graph.neighbors[i].append(j)
            graph.neighbors[j].append(i)
        
        graph.max_neighbor = max(len(neigh) for neigh in graph.neighbors)
        
        edges = [[i, j] for i, j in g.edges()]
        edges.extend([[j, i] for i, j in g.edges()])
        graph.edge_mat = torch.LongTensor(edges).transpose(0, 1)
        
        # Store raw features directly
        graph.node_features = torch.tensor(node_features, dtype=torch.float)

        mol_features = calculate_rdkit_features(mol)
        graph.mol_features = mol_features 
        
        return graph
    except Exception as e:
        print(f"Error processing molecule: {e}")
        return None
    
def prepare_molecular_dataset(train_df, test_df=None):
    """
    Prepare molecular dataset from SMILES strings.
    
    Args:
        train_df: Training data DataFrame
        test_df: Test data DataFrame (optional)
    
    Returns:
        Tuple of (train_graphs, test_graphs)
    """
    train_graphs = []
    test_graphs = []
    skipped_train = 0
    
    if train_df is not None:
        for idx, row in train_df.iterrows():
            mol = Chem.MolFromSmiles(row['Drug'])
            if mol is None:
                skipped_train += 1
                continue
            try:
                graph = mol_to_s2v_graph(mol, row['Y'])
                if graph is not None:
                    train_graphs.append(graph)
            except Exception as e:
                skipped_train += 1

    if test_df is not None:
        skipped_test = 0

        for idx, row in test_df.iterrows():
            mol = Chem.MolFromSmiles(row['Drug'])
            if mol is None:
                skipped_test += 1
                continue
            try:
                graph = mol_to_s2v_graph(mol, row['Y'])
                if graph is not None:
                    test_graphs.append(graph)
            except Exception as e:
                skipped_test += 1

    # Only print summary if molecules were processed
    # if train_df is not None:
    #     print(f"Processed {len(train_graphs)} training graphs" + 
    #           (f", skipped {skipped_train} molecules" if skipped_train > 0 else ""))

    return train_graphs, test_graphs

# Example usage
if __name__ == "__main__":
    # Create a test molecule
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
    mol = Chem.MolFromSmiles(smiles)
    label = 1
    
    # Convert to graph
    graph = mol_to_s2v_graph(mol, label)
    
    print("\nGraph details:")
    print(f"Number of nodes: {len(graph.g)}")
    print(f"Number of node tags: {len(graph.node_tags)}")
    print(f"Node features shape: {graph.node_features.shape}")
    print(f"Edge matrix shape: {graph.edge_mat.shape}")
    print(f"Maximum neighbors: {graph.max_neighbor}")