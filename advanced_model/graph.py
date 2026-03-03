import numpy as np
import networkx as nx

class Graph:
    """
    The Graph to model the skeletons extracted by MediaPipe.
    Args:
        strategy (string): must be one of the followings:
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points
    """
    def __init__(self, strategy='spatial', max_hop=1, dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation
        
        self.get_edge()
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self):
        # -----------------------------------------------------
        # Node Layout (119 nodes)
        # 00-32: Pose (33)
        # 33-53: Left Hand (21)
        # 54-74: Right Hand (21)
        # 75-118: Face (44)
        # -----------------------------------------------------
        self.num_node = 119
        self_link = [(i, i) for i in range(self.num_node)]
        neighbor_link = []

        # =====================================================
        # 1. POSE EDGES (MediaPipe Pose Topology)
        # =====================================================
        # Standard MP Body connections
        pose_connections = [
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5),
            (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
            (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
            (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
            (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
            (29, 31), (30, 32), (27, 31), (28, 32)
        ]
        # Offset is 0 for Pose
        neighbor_link += pose_connections

        # =====================================================
        # 2. HAND EDGES
        # =====================================================
        # Standard MP Hand connections (same for Left and Right)
        hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),         # Index
            (0, 9), (9, 10), (10, 11), (11, 12),    # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20)   # Pinky
        ]
        
        # Left Hand (Offset 33)
        lh_offset = 33
        neighbor_link += [(i + lh_offset, j + lh_offset) for i, j in hand_connections]

        # Right Hand (Offset 33+21 = 54)
        rh_offset = 54
        neighbor_link += [(i + rh_offset, j + rh_offset) for i, j in hand_connections]

        # =====================================================
        # 3. FACE EDGES (Reduced 44)
        # =====================================================
        # Since we have a custom subset (Lips + Eyes), we don't have a standard topology.
        # We will create a local connectivity:
        # - Lips (0-39 in face subset): Connect sequentially as a loop? 
        #   Indices 0-19 are range(61,81) (Inner lip?), 20-39 are range(291,311).
        # - Eyes (40-43): Connect to each other or neighbors?
        
        # Strategy: Connect them sequentially within their blocks to simulate a structure
        face_offset = 75
        
        # Block 1: Lips 1 (0-19)
        for i in range(19):
            neighbor_link.append((face_offset + i, face_offset + i + 1))
        # Close loop
        neighbor_link.append((face_offset + 19, face_offset + 0))
        
        # Block 2: Lips 2 (20-39)
        for i in range(19):
            neighbor_link.append((face_offset + 20 + i, face_offset + 20 + i + 1))
        # Close loop
        neighbor_link.append((face_offset + 39, face_offset + 20))

        # Block 3: Eyes (40-43) - [33, 133, 362, 263]
        # Just connect them to create a small "eye box" or line
        neighbor_link.append((face_offset + 40, face_offset + 41))
        neighbor_link.append((face_offset + 42, face_offset + 43))
        
        # =====================================================
        # 4. INTER-PART CONNECTIONS (Bridging the gap)
        # =====================================================
        # Pose Wrist -> Hand Wrist
        # Left Wrist (Pose 15) -> Left Hand Root (33 + 0)
        neighbor_link.append((15, 33))
        # Right Wrist (Pose 16) -> Right Hand Root (54 + 0)
        neighbor_link.append((16, 54))

        # Pose Nose (0) -> Face
        # Connect Nose to Lip start and Eye connection points to anchor the face
        neighbor_link.append((0, face_offset + 0))  # Nose -> Lip1
        neighbor_link.append((0, face_offset + 20)) # Nose -> Lip2
        neighbor_link.append((0, face_offset + 40)) # Nose -> Eye L
        neighbor_link.append((0, face_offset + 42)) # Nose -> Eye R

        self.edge = self_link + neighbor_link
        self.center = 0 # Center gravity (0 = Nose, or 11/12 Shoulders are often better)
        # Let's use 0 (Nose) as it's central to Face/Hands in sign language, 
        # or stick to ST-GCN standard 1 (Neck/Spine top) if available? 
        # MP Pose 0 is Nose. 11/12 are Shoulders. 
        # Let's set Center = 0 (Nose/Head) for now as it anchors the face.

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do not support this strategy.")

def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # Compute hop distance
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)

    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis

def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD
