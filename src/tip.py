import numpy as np
import heapq
from collections import deque
from scipy.optimize import curve_fit


class TIPEngine:
    """Core simulation logic for Trapping Invasion Percolation (TIP)."""

    def __init__(self, field, mode='site'):
        self.field = field          # 2D Array of capillary pressures/radii
        self.L = field.shape[0]     # Assuming a square L x L grid for 2D
        self.mode = mode            # 'site' (imbibition) or 'bond' (drainage)

        # State tracking: 0=Defender (Uninvaded), 1=Invader, 2=Trapped
        self.state = np.zeros((self.L, self.L), dtype=int)

        # Priority queue for the fluid-fluid interface (min-heap in Python)
        self.interface = []

        # Tracking if breakthrough has occurred
        self.breakthrough = False

    def _initialize_invasion(self):
        """Sets up a point inlet at the center bottom edge (L//2, 0)."""
        start_x, start_y = self.L // 2, 0
        self.state[start_x, start_y] = 1
        self._add_neighbors_to_interface(start_x, start_y)

    def _add_neighbors_to_interface(self, x, y):
        """Finds defending neighbors and adds them to the priority queue."""
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.L and 0 <= ny < self.L:
                if self.state[nx, ny] == 0:
                    # Python's heapq is a min-heap.
                    # For Bond TIP (Drainage), we invade smallest constrictions first -> push positive.
                    # For Site TIP (Imbibition), we invade largest pores first -> push negative.
                    val = self.field[nx, ny] if self.mode == 'bond' else -self.field[nx, ny]
                    heapq.heappush(self.interface, (val, nx, ny))

    def _local_trapping_check(self, x, y):
        """
        The 'Simultaneous Forest-Fire' Algorithm.
        Searches outward from the uninvaded neighbors of the newly invaded site (x, y).
        Uses reflecting boundary conditions on x-edges: only the top edge (y = L-1)
        is the outlet/escape route, not the side walls.
        """
        # Find all uninvaded neighbors of the new invasion site
        uninvaded = [(x+dx, y+dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                     if 0 <= x+dx < self.L and 0 <= y+dy < self.L and self.state[x+dx, y+dy] == 0]

        if not uninvaded:
            return # No neighbors to check

        # Setup independent BFS searches for each neighbor
        queues = {i: [n] for i, n in enumerate(uninvaded)}
        visited = {i: {n} for i, n in enumerate(uninvaded)}
        active = {i: True for i in range(len(uninvaded))}

        # Global tracker to see if two searches collide
        global_visited = {n: i for i, n in enumerate(uninvaded)}

        # Advance all active searches one step at a time (interleaved BFS)
        while any(active.values()):
            for i in list(active.keys()):
                if not active[i]:
                    continue

                # If queue is empty and it never hit the boundary, it is TRAPPED!
                if not queues[i]:
                    active[i] = False
                    for tx, ty in visited[i]:
                        self.state[tx, ty] = 2  # Mark cluster as trapped
                    continue

                cx, cy = queues[i].pop(0)

                # Reflecting BCs on x-edges: only the top edge (y = L-1) is the outlet.
                # Side walls (x=0, x=L-1) are NOT escape routes.
                if cy == self.L - 1:
                    active[i] = False # Component 'i' is NOT trapped
                    continue

                # Expand to uninvaded neighbors
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < self.L and 0 <= ny < self.L and self.state[nx, ny] == 0:
                        if (nx, ny) not in global_visited:
                            global_visited[(nx, ny)] = i
                            visited[i].add((nx, ny))
                            queues[i].append((nx, ny))
                        elif global_visited[(nx, ny)] != i:
                            # Search 'i' collided with Search 'j'
                            j = global_visited[(nx, ny)]
                            if not active[j]:
                                # Search 'j' already reached the outlet; 'i' is therefore also safe
                                active[i] = False
                                break
                            else:
                                # Merge search 'i' into search 'j' to avoid redundant work
                                visited[j].update(visited[i])
                                queues[j].extend(queues[i])
                                for node in visited[i]:
                                    global_visited[node] = j
                                active[i] = False
                                del queues[i]
                                del visited[i]
                                break

    def step(self):
        """Executes a single invasion percolation step."""
        if not self.interface:
            return False

        # 1. Pop the most favorable site/bond from the interface
        val, x, y = heapq.heappop(self.interface)

        # If the site was already invaded or trapped previously, skip it
        # (Sites can be added to the queue multiple times, or trapped after being queued)
        if self.state[x, y] != 0:
            return True

        # 2. Invade the site
        self.state[x, y] = 1

        # Check for breakthrough (reached the top edge)
        if y == self.L - 1:
            self.breakthrough = True
            return False

        # 3. Add new neighbors to the interface
        self._add_neighbors_to_interface(x, y)

        # 4. Local Trapping Check
        self._local_trapping_check(x, y)

        return True

    def run_until_breakthrough(self):
        """Runs the step() process in a loop until the invading fluid reaches the outlet."""
        self._initialize_invasion()

        print("Starting Invasion Percolation...")
        steps = 0
        while not self.breakthrough:
            continue_sim = self.step()
            steps += 1
            if not continue_sim and not self.breakthrough:
                print("Simulation stalled. No uninvaded, untrapped sites left.")
                break

        print(f"Breakthrough reached after {steps} invasion steps.")
        return self.state


class BackboneExtractor:
    """Module to identify elastic and transport backbones from the cluster."""

    def __init__(self, state_grid):
        self.state = state_grid
        self.L = state_grid.shape[0]
        self.inlet = (self.L // 2, 0)
        self.distances = np.full((self.L, self.L), np.inf)
        self.elastic_bb = set()
        self.transport_bb = set()

    def _get_invaded_neighbors(self, x, y):
        """Returns neighbors that are part of the invaded cluster (state == 1)."""
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.L and 0 <= ny < self.L:
                if self.state[nx, ny] == 1:
                    neighbors.append((nx, ny))
        return neighbors

    def _calculate_chemical_distances(self):
        """
        Step 1: Breadth-First Search (BFS) from the inlet to label every
        invaded site with its shortest distance along the cluster.
        """
        queue = deque([self.inlet])
        self.distances[self.inlet[0], self.inlet[1]] = 0

        while queue:
            cx, cy = queue.popleft()
            current_dist = self.distances[cx, cy]

            for nx, ny in self._get_invaded_neighbors(cx, cy):
                if self.distances[nx, ny] == np.inf:
                    self.distances[nx, ny] = current_dist + 1
                    queue.append((nx, ny))

    def extract_elastic_backbone(self):
        """
        Step 2: 'Burning' algorithm. Starts from the breakthrough point and traces
        the steepest descent of distance labels back to the inlet.
        """
        self._calculate_chemical_distances()

        # Find the breakthrough point on the outlet boundary (y = L - 1)
        # We pick the one with the shortest distance to the inlet
        breakthrough_sites = [(x, self.L - 1) for x in range(self.L) if self.state[x, self.L - 1] == 1]
        if not breakthrough_sites:
            print("Warning: No breakthrough detected. Cannot extract backbone.")
            return set()

        current = min(breakthrough_sites, key=lambda p: self.distances[p[0], p[1]])
        self.elastic_bb.add(current)

        # Trace back to inlet
        while current != self.inlet:
            neighbors = self._get_invaded_neighbors(current[0], current[1])
            # Move to the neighbor with the strictly smaller distance label
            next_step = min(neighbors, key=lambda p: self.distances[p[0], p[1]])
            self.elastic_bb.add(next_step)
            current = next_step

        return self.elastic_bb

    def extract_transport_backbone(self):
        """
        Step 3: DFS to find all loops. It starts from branch points on the backbone,
        prioritizes paths moving closer to the inlet, and identifies loops.
        Dead ends are marked and discarded to avoid redundant searches.
        """
        if not self.elastic_bb:
            self.extract_elastic_backbone()

        self.transport_bb = set(self.elastic_bb)
        dead_ends = set()

        # Sort current backbone nodes by distance so we process them systematically
        branch_points = sorted(list(self.transport_bb), key=lambda p: self.distances[p[0], p[1]])

        for bp in branch_points:
            # Look for branches starting from this backbone node
            neighbors = self._get_invaded_neighbors(bp[0], bp[1])
            starts = [n for n in neighbors if n not in self.transport_bb and n not in dead_ends]

            for start_node in starts:
                # DFS Stack: stores (current_node, path_taken_to_get_here)
                stack = [(start_node, [start_node])]
                local_visited = {start_node}
                loop_found = False

                while stack:
                    curr, path = stack.pop()

                    curr_neighbors = self._get_invaded_neighbors(curr[0], curr[1])
                    valid_neighbors = [n for n in curr_neighbors if n != bp and n not in dead_ends]

                    # CRITICAL OPTIMIZATION from the paper:
                    # "explore the branch which is labeled closest to the inlet face first"
                    # We sort descending so the smallest distance is at the end of the list (popped first)
                    valid_neighbors.sort(key=lambda p: self.distances[p[0], p[1]], reverse=True)

                    hit_backbone = False
                    for n in valid_neighbors:
                        if n in self.transport_bb:
                            # Loop found! It reconnected to the backbone at a different site.
                            self.transport_bb.update(path)
                            hit_backbone = True
                            loop_found = True
                            break

                    if hit_backbone:
                        # If this branch forms a loop, the path is now part of the backbone.
                        # We stop exploring deeper on this specific trajectory and move to the next.
                        continue

                    for n in valid_neighbors:
                        if n not in local_visited and n not in self.transport_bb:
                            local_visited.add(n)
                            stack.append((n, path + [n]))

                # If the entire DFS completes without hitting the backbone, it's a dangling end.
                if not loop_found:
                    dead_ends.update(local_visited)

        return self.transport_bb

    def get_backbone_grids(self):
        """Returns 2D numpy arrays suitable for visualization."""
        elastic_grid = np.zeros((self.L, self.L), dtype=int)
        for x, y in self.elastic_bb:
            elastic_grid[x, y] = 1

        transport_grid = np.zeros((self.L, self.L), dtype=int)
        for x, y in self.transport_bb:
            transport_grid[x, y] = 1

        return elastic_grid, transport_grid


class DataAnalyzer:
    """Module for scaling analysis and fractal dimension calculation."""

    def __init__(self, final_state, elastic_bb, transport_bb):
        self.state = final_state
        self.elastic_bb = elastic_bb
        self.transport_bb = transport_bb

    def calculate_mass(self, target='cluster'):
        """
        Counts the number of sites belonging to the target structure.

        Parameters:
            target (str): 'cluster' (invading fluid), 'trapped' (defending),
                          'elastic' (shortest path), or 'transport' (loops).
        """
        if target == 'cluster':
            # The Sample-Spanning Cluster (SSC) is the invading fluid (state == 1)
            return np.sum(self.state == 1)
        elif target == 'trapped':
            # The trapped defending fluid (state == 2)
            return np.sum(self.state == 2)
        elif target == 'elastic':
            return len(self.elastic_bb)
        elif target == 'transport':
            return len(self.transport_bb)
        else:
            raise ValueError("Target must be 'cluster', 'trapped', 'elastic', or 'transport'.")

    def calculate_minimal_path_length(self):
        """Length of the elastic backbone (shortest path through SSC)."""
        return len(self.elastic_bb)

    def calculate_radius_of_gyration(self, target='cluster'):
        """
        Calculates the Radius of Gyration (Rg), which measures how the
        mass of the cluster is distributed around its center of mass.
        """
        if target == 'cluster':
            coords = np.argwhere(self.state == 1)
        elif target == 'elastic':
            coords = np.array(list(self.elastic_bb))
        elif target == 'transport':
            coords = np.array(list(self.transport_bb))
        else:
            raise ValueError("Invalid target for Radius of Gyration.")

        if len(coords) == 0:
            return 0.0

        # Center of Mass (CM)
        center_of_mass = np.mean(coords, axis=0)

        # Mean squared distance from the CM
        sq_distances = np.sum((coords - center_of_mass)**2, axis=1)
        rg = np.sqrt(np.mean(sq_distances))

        return rg

    def fit_standard_scaling(self, L_array, mass_array):
        """
        Standard power-law scaling fit: M ~ L^Df.
        Uses a simple log-log linear regression. Highly stable.

        Returns:
            Df (float): The estimated fractal dimension.
        """
        log_L = np.log(L_array)
        log_M = np.log(mass_array)

        # Polyfit returns [slope, intercept] for a 1st degree polynomial
        coeffs = np.polyfit(log_L, log_M, 1)
        Df = coeffs[0]

        return Df

    def fit_advanced_finite_size_scaling(self, L_array, mass_array):
        """
        Fits data to the paper's specific correction-to-scaling equation:
        c1 + Df * M^w = c2 * L^(w * Df)

        Rearranged to solve for Mass (M) to allow standard curve fitting:
        M = [ (c2 * L^(w * Df) - c1) / Df ] ^ (1 / w)

        Returns:
            Df (float): The corrected fractal dimension.
            w (float): The correction-to-scaling exponent.
        """
        # Ensure inputs are numpy arrays
        L = np.array(L_array, dtype=float)
        M = np.array(mass_array, dtype=float)

        # Define the target function for scipy's curve_fit
        def scaling_func(L, c1, c2, Df, w):
            # Protect against fractional powers of negative numbers during optimization
            inner_term = (c2 * (L ** (w * Df)) - c1) / Df
            # np.maximum prevents negative values from throwing math errors
            return np.maximum(inner_term, 1e-10) ** (1.0 / w)

        try:
            # Provide initial guesses [c1, c2, Df, w]
            # Standard percolation expects w ~ 1.0 to 2.0, Df ~ 1.89 (in 2D)
            initial_guesses = [1.0, 1.0, 1.89, 1.0]

            # Bound Df between 1 and d(2), and w strictly positive
            bounds = ([-np.inf, -np.inf, 1.0, 0.01], [np.inf, np.inf, 2.0, 5.0])

            popt, _ = curve_fit(scaling_func, L, M, p0=initial_guesses, bounds=bounds, maxfev=10000)

            c1, c2, Df, w = popt
            return Df, w

        except RuntimeError:
            print("Advanced scaling fit failed to converge. Falling back to standard scaling.")
            return self.fit_standard_scaling(L_array, mass_array), None
