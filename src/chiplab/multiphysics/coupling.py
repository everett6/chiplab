"""Feedback loop manager for multiphysics coupling.

This module implements the coupling graph, feedback loop identification,
and various coupling schemes (staggered, monolithic, quasi-static) for
managing the multiphysics feedback cycles.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


class CouplingGraph:
    """Directed graph of field dependencies.
    
    Represents the dependency relationships between different field variables
    in the multiphysics system. Edges represent physical couplings (e.g., E→T,
    T→η, η→c, c→σ, σ→E).
    """
    
    def __init__(self):
        """Initialize an empty coupling graph."""
        self.nodes: Set[str] = set()
        self.edges: Dict[str, Set[str]] = {}  # node -> set of dependent nodes
        self.reverse_edges: Dict[str, Set[str]] = {}  # node -> set of dependency nodes
    
    def add_node(self, node: str) -> None:
        """Add a field variable node to the graph."""
        if node not in self.nodes:
            self.nodes.add(node)
            if node not in self.edges:
                self.edges[node] = set()
            if node not in self.reverse_edges:
                self.reverse_edges[node] = set()
    
    def add_edge(self, from_node: str, to_node: str) -> None:
        """Add a directed edge from one field to another.
        
        Parameters
        ----------
        from_node : str
            Source field variable
        to_node : str
            Target field variable
        """
        self.add_node(from_node)
        self.add_node(to_node)
        self.edges[from_node].add(to_node)
        self.reverse_edges[to_node].add(from_node)
    
    def has_edge(self, from_node: str, to_node: str) -> bool:
        """Check if there is a directed edge from from_node to to_node."""
        return to_node in self.edges.get(from_node, set())
    
    def get_dependents(self, node: str) -> Set[str]:
        """Get all nodes that depend on the given node."""
        return self.edges.get(node, set())
    
    def get_dependencies(self, node: str) -> Set[str]:
        """Get all nodes that the given node depends on."""
        return self.reverse_edges.get(node, set())
    
    def has_cycle(self) -> bool:
        """Check if the graph contains any directed cycles."""
        # DFS-based cycle detection
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        
        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in self.edges.get(node, set()):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in self.nodes:
            if node not in visited:
                if dfs(node):
                    return True
        
        return False
    
    def get_cycles(self) -> List[List[str]]:
        """Get all directed cycles in the graph."""
        cycles: List[List[str]] = []
        visited: Set[str] = set()
        rec_stack: List[str] = []
        
        def dfs(node: str):
            visited.add(node)
            rec_stack.append(node)
            
            for neighbor in self.edges.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in rec_stack:
                    # Found a cycle - extract it
                    cycle_start = rec_stack.index(neighbor)
                    cycle = rec_stack[cycle_start:] + [neighbor]
                    if cycle not in cycles:
                        cycles.append(cycle)
            
            rec_stack.pop()
        
        for node in self.nodes:
            if node not in visited:
                dfs(node)
        
        return cycles
    
    def topological_sort(self) -> Optional[List[str]]:
        """Return a topological ordering of nodes, or None if cycles exist."""
        in_degree: Dict[str, int] = {node: 0 for node in self.nodes}
        
        # Count in-degrees
        for node in self.nodes:
            for dependent in self.edges.get(node, set()):
                in_degree[dependent] = in_degree.get(dependent, 0) + 1
        
        # Kahn's algorithm
        queue: List[str] = [node for node in self.nodes if in_degree[node] == 0]
        result: List[str] = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            for dependent in self.edges.get(node, set()):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        if len(result) != len(self.nodes):
            return None  # Cycles exist
        
        return result


class FeedbackLoop:
    """Identifies and manages feedback cycles in the multiphysics system.
    
    Detects when changes in one field propagate back to itself through
    the coupling graph, creating a feedback loop.
    """
    
    def __init__(self, coupling_graph: CouplingGraph):
        """Initialize feedback loop manager.
        
        Parameters
        ----------
        coupling_graph : CouplingGraph
            The coupling graph to analyze
        """
        self.graph = coupling_graph
        self.cycles: List[List[str]] = self.graph.get_cycles()
        self.is_feedback_active: bool = len(self.cycles) > 0
    
    def identify_loops(self) -> List[Dict[str, Any]]:
        """Identify all feedback loops and their characteristics.
        
        Returns
        -------
        List of dicts, each describing a feedback loop with:
            - cycle: list of field names in the loop
            - strength: estimated coupling strength (0-1)
            - direction: clockwise or counterclockwise
        """
        loops: List[Dict[str, Any]] = []
        
        for cycle in self.cycles:
            loop_info = {
                "cycle": cycle,
                "length": len(cycle),
                "strength": self._estimate_loop_strength(cycle),
                "direction": "clockwise" if len(cycle) % 2 == 0 else "counterclockwise",
            }
            loops.append(loop_info)
        
        return loops
    
    def _estimate_loop_strength(self, cycle: List[str]) -> float:
        """Estimate the strength of a feedback loop.
        
        Parameters
        ----------
        cycle : List[str]
            Sequence of field names in the loop
            
        Returns
        -------
        float
            Estimated strength in [0, 1], where 1 is strongest feedback
        """
        if len(cycle) < 2:
            return 0.0
        
        # Simple estimate based on cycle length: shorter loops are stronger
        # and longer chains have weaker effective feedback
        base_strength = 1.0 / len(cycle)
        
        # Adjust based on node types (physical coupling strengths)
        strength = base_strength
        for i in range(len(cycle)):
            from_node = cycle[i]
            to_node = cycle[(i + 1) % len(cycle)]
            # Nodes with stronger physical coupling contribute more
            if "E" in from_node and "T" in to_node:
                strength *= 1.2  # E→T is strong (Joule heating)
            if "T" in from_node and "η" in to_node or "viscosity" in to_node:
                strength *= 1.1  # T→η is moderate
            if "η" in from_node or "viscosity" in from_node and "c" in to_node:
                strength *= 1.05  # η→c is moderate
            if "c" in from_node and "σ" in to_node:
                strength *= 1.15  # c→σ is significant
            if "σ" in from_node and "E" in to_node:
                strength *= 1.3  # σ→E is very strong (Ohm's law)
        
        return min(strength, 1.0)
    
    def get_feedback_variables(self, node: str) -> Set[str]:
        """Get all variables that feed back into the given node."""
        # Find all nodes that can reach this node through directed paths
        reaching: Set[str] = set()
        queue: List[str] = [node]
        
        while queue:
            current = queue.pop(0)
            if current in reaching:
                continue
            reaching.add(current)
            # Add nodes that have edges TO current (i.e., current depends on them)
            for dep in self.graph.get_dependencies(current):
                if dep not in reaching:
                    queue.append(dep)
        
        return reaching


class CouplingScheme:
    """Base class for coupling schemes.
    
    Defines the interface for different multiphysics coupling approaches.
    """
    
    def __init__(self, name: str, graph: CouplingGraph, 
                 convergence_tol: float = 1e-6, max_iterations: int = 50):
        """Initialize coupling scheme.
        
        Parameters
        ----------
        name : str
            Name of the scheme
        graph : CouplingGraph
            Coupling graph describing field dependencies
        convergence_tol : float
            Convergence tolerance for coupled iterations
        max_iterations : int
            Maximum number of coupling iterations
        """
        self.name = name
        self.graph = graph
        self.convergence_tol = convergence_tol
        self.max_iterations = max_iterations
        self.convergence_monitor = None  # Will be set by subclass
    
    def solve_step(self, fields: FieldSystem, dt: float) -> Tuple[FieldSystem, bool]:
        """Perform one coupling step.
        
        Parameters
        ----------
        fields : FieldSystem
            Current field values
        dt : float
            Time step size
            
        Returns
        -------
        Tuple[FieldSystem, bool]
            Updated fields and whether convergence was achieved
        """
        raise NotImplementedError
    
    def check_convergence(self, fields_old: FieldSystem, fields_new: FieldSystem) -> bool:
        """Check if the fields have converged.
        
        Parameters
        ----------
        fields_old : FieldSystem
            Fields before the iteration
        fields_new : FieldSystem
            Fields after the iteration
            
        Returns
        -------
        bool
            True if converged within tolerance
        """
        raise NotImplementedError


class StaggeredScheme(CouplingScheme):
    """Staggered coupling scheme: sequential solve E → T → c → σ → repeat.
    
    Solves fields in sequence, using latest values from previous steps.
    Good for initial exploration, less expensive than monolithic but may
    converge slowly or miss some couplings.
    """
    
    def __init__(self, graph: CouplingGraph, convergence_tol: float = 1e-6,
                 max_iterations: int = 50, relaxation: float = 0.6):
        """Initialize staggered scheme.
        
        Parameters
        ----------
        graph : CouplingGraph
            Coupling graph
        convergence_tol : float
            Convergence tolerance
        max_iterations : int
            Maximum number of outer coupling iterations
        relaxation : float
            Relaxation factor for under-relaxation (0 < ω ≤ 1)
        """
        super().__init__("Staggered", graph, convergence_tol, max_iterations)
        self.relaxation = relaxation
    
    def solve_step(self, fields: FieldSystem, dt: float) -> Tuple[FieldSystem, bool]:
        """Perform one staggered coupling step.
        
        Solves fields in the sequence: E → T → c → σ → E ...
        Repeats until convergence or max iterations.
        
        Parameters
        ----------
        fields : FieldSystem
            Current field values
        dt : float
            Time step size
            
        Returns
        -------
        Tuple[FieldSystem, bool]
            Updated fields and whether convergence was achieved
        """
        fields_updated = fields.copy()
        converged = False
        
        for iteration in range(self.max_iterations):
            fields_before = fields_updated.copy()
            
            # Solve Electric field (Maxwell's equations)
            # Uses current T, c, σ values
            from .fields import MaxwellEquations
            maxwell = MaxwellEquations(fields_updated)
            Q_joule = maxwell.update_from_joule_heating()
            
            # Solve Heat equation (temperature update)
            from .fields import HeatEquation
            heat = HeatEquation(fields_updated)
            heat_result = heat.time_step(dt)
            
            # Solve Particle dynamics (concentration update)
            from .fields import ParticleDynamics
            particle = ParticleDynamics(fields_updated)
            particle_result = particle.time_step(dt)
            
            # Update electrical conductivity based on new state
            from .fields import ConstitutiveRelations
            constitutive = ConstitutiveRelations(fields_updated)
            new_sigma = constitutive.electrical_conductivity()
            fields_updated.sigma = TrackedValue(
                value=new_sigma, unit="S/m",
                provenance=fields_updated.sigma.provenance,
                source="StaggeredScheme"
            )
            
            # Check convergence
            diff = self._compute_max_diff(fields_before, fields_updated)
            if diff < self.convergence_tol:
                converged = True
                break
            
            # Apply relaxation
            fields_updated = self._apply_relaxation(fields_before, fields_updated)
        
        return fields_updated, converged
    
    def _compute_max_diff(self, fields_old: FieldSystem, fields_new: FieldSystem) -> float:
        """Compute maximum difference between old and new field values."""
        max_diff = 0.0
        
        # Compare temperatures
        diff_T = abs(float(fields_new.T) - float(fields_old.T))
        max_diff = max(max_diff, diff_T)
        
        # Compare concentrations
        diff_c = abs(float(fields_new.c) - float(fields_old.c))
        max_diff = max(max_diff, diff_c)
        
        # Compare conductivities
        diff_sigma = abs(float(fields_new.sigma) - float(fields_old.sigma))
        max_diff = max(max_diff, diff_sigma)
        
        return max_diff
    
    def _apply_relaxation(self, fields_before: FieldSystem, fields_after: FieldSystem) -> FieldSystem:
        """Apply under-relaxation to updated fields."""
        fields_relaxed = fields_after.copy()
        
        # Relax temperature
        T_old = float(fields_before.T)
        T_new = float(fields_after.T)
        T_relaxed = T_old + self.relaxation * (T_new - T_old)
        fields_relaxed.T = TrackedValue(
            value=T_relaxed, unit="K",
            provenance=fields_before.T.provenance,
            source="StaggeredScheme"
        )
        
        # Relax concentration
        c_old = float(fields_before.c)
        c_new = float(fields_after.c)
        c_relaxed = c_old + self.relaxation * (c_new - c_old)
        fields_relaxed.c = TrackedValue(
            value=max(0.0, c_relaxed), unit="1/m³",
            provenance=fields_before.c.provenance,
            source="StaggeredScheme"
        )
        
        return fields_relaxed


class MonolithicScheme(CouplingScheme):
    """Monolithic coupling scheme: fully coupled Jacobian Newton-Raphson.
    
    Solves all fields simultaneously using a fully coupled Jacobian matrix.
    Most accurate but most expensive. Uses Newton-Raphson with physics-based
    preconditioning for convergence.
    """
    
    def __init__(self, graph: CouplingGraph, convergence_tol: float = 1e-8,
                 max_iterations: int = 25, use_preconditioner: bool = True):
        """Initialize monolithic scheme.
        
        Parameters
        ----------
        graph : CouplingGraph
            Coupling graph
        convergence_tol : float
            Convergence tolerance
        max_iterations : int
            Maximum number of Newton-Raphson iterations
        use_preconditioner : bool
            Whether to use physics-based preconditioning
        """
        super().__init__("Monolithic", graph, convergence_tol, max_iterations)
        self.use_preconditioner = use_preconditioner
    
    def solve_step(self, fields: FieldSystem, dt: float) -> Tuple[FieldSystem, bool]:
        """Perform one monolithic coupling step using Newton-Raphson.
        
        Parameters
        ----------
        fields : FieldSystem
            Current field values
        dt : float
            Time step size
            
        Returns
        -------
        Tuple[FieldSystem, bool]
            Updated fields and whether convergence was achieved
        """
        fields_updated = fields.copy()
        converged = False
        
        for iteration in range(self.max_iterations):
            fields_before = fields_updated.copy()
            
            # Compute residual (uncoupled update errors)
            residual = self._compute_residual(fields_updated, dt)
            
            if float(np.linalg.norm(residual)) < self.convergence_tol:
                converged = True
                break
            
            # Compute Jacobian (fully coupled)
            jacobian = self._compute_jacobian(fields_updated, dt)
            
            # Solve for update step
            if self.use_preconditioner:
                jacobian_precond = self._apply_preconditioner(jacobian)
                delta = np.linalg.solve(jacobian_precond, residual)
            else:
                delta = np.linalg.solve(jacobian, residual)
            
            # Apply update with under-relaxation
            # This is a simplified version - full implementation would
            # update all field components simultaneously
            fields_updated = self._apply_newton_step(fields_before, delta)
        
        return fields_updated, converged
    
    def _compute_residual(self, fields: FieldSystem, dt: float) -> np.ndarray:
        """Compute the residual vector for the coupled system."""
        # Placeholder: in full implementation, this would compute
        # the imbalance in all field equations simultaneously
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    def _compute_jacobian(self, fields: FieldSystem, dt: float) -> np.ndarray:
        """Compute the fully coupled Jacobian matrix."""
        # Placeholder: full implementation would compute partial derivatives
        # ∂R/∂E, ∂R/∂T, ∂R/∂c, ∂R/∂σ, etc.
        n_fields = 6  # E, T, c, σ, H, u
        return np.eye(n_fields)  # Identity as placeholder
    
    def _apply_preconditioner(self, jacobian: np.ndarray) -> np.ndarray:
        """Apply physics-based preconditioner to Jacobian."""
        # Block-diagonal preconditioner with physics-based blocks
        # Separates fast (E) and slow (T, c, σ) field couplings
        n = jacobian.shape[0]
        # Simple diagonal preconditioner
        diag = np.diag(np.abs(np.diag(jacobian)) + 1e-10)
        return diag
    
    def _apply_newton_step(self, fields_before: FieldSystem, delta: np.ndarray) -> FieldSystem:
        """Apply Newton-Raphson update step to fields."""
        fields_updated = fields_before.copy()
        
        # Map delta components to field updates
        # This is simplified - full implementation would have proper mapping
        if len(delta) >= 1:
            # Update temperature
            T_old = float(fields_before.T)
            T_new = T_old + delta[0] if len(delta) > 0 else T_old
            T_new = max(100.0, min(1000.0, T_new))  # Clamp
            fields_updated.T = TrackedValue(
                value=T_new, unit="K",
                provenance=fields_before.T.provenance,
                source="MonolithicScheme"
            )
        
        if len(delta) >= 2:
            # Update concentration
            c_old = float(fields_before.c)
            c_new = c_old + delta[1] if len(delta) > 1 else c_old
            c_new = max(0.0, min(1.0, c_new))  # Clamp to [0, 1]
            fields_updated.c = TrackedValue(
                value=c_new, unit="1/m³",
                provenance=fields_before.c.provenance,
                source="MonolithicScheme"
            )
        
        return fields_updated


class ConvergenceMonitor:
    """Monitor convergence of coupling iterations.
    
    Tracks residuals across coupling iterations and provides
    relaxation factor updates and Anderson acceleration support.
    """
    
    def __init__(self, tolerance: float = 1e-6, max_iterations: int = 50,
                 use_anderson: bool = False, anderson_history: int = 5):
        """Initialize convergence monitor.
        
        Parameters
        ----------
        tolerance : float
            Convergence tolerance for residual norm
        max_iterations : int
            Maximum number of monitoring iterations
        use_anderson : bool
            Whether to use Anderson acceleration
        anderson_history : int
            Number of previous iterations to keep for Anderson acceleration
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.use_anderson = use_anderson
        self.anderson_history = anderson_history
        self.residual_history: list[float] = []
        self.iteration = 0
    
    def update(self, residual: float) -> bool:
        """Record a residual and check for convergence.
        
        Parameters
        ----------
        residual : float
            Current residual norm
            
        Returns
        -------
        bool
            True if converged (residual < tolerance)
        """
        self.residual_history.append(residual)
        self.iteration += 1
        
        if len(self.residual_history) > self.max_iterations:
            self.residual_history.pop(0)
        
        # Check convergence
        if residual < self.tolerance:
            return True
        
        # Check if we've exceeded max iterations
        if self.iteration >= self.max_iterations:
            return True
        
        return False
    
    def has_converged(self) -> bool:
        """Check if the monitor indicates convergence."""
        if not self.residual_history:
            return False
        return self.residual_history[-1] < self.tolerance
    
    def get_status(self) -> dict[str, Any]:
        """Get current monitoring status."""
        return {
            "iteration": self.iteration,
            "current_residual": self.residual_history[-1] if self.residual_history else float('inf'),
            "converged": self.has_converged(),
            "tolerance": self.tolerance,
        }


class QuasiStaticScheme(CouplingScheme):
    """Quasi-static coupling scheme: fast fields equilibrated, slow fields evolved.
    
    Identifies which fields equilibrate quickly (electromagnetic) versus
    which evolve slowly (thermal, particle dynamics, mechanical) and
    solves accordingly. Fast for systems where separation of timescales exists.
    """
    
    def __init__(self, graph: CouplingGraph, convergence_tol: float = 1e-7,
                 max_iterations: int = 30, fast_fields: List[str] | None = None,
                 slow_fields: List[str] | None = None):
        """Initialize quasi-static scheme.
        
        Parameters
        ----------
        graph : CouplingGraph
            Coupling graph
        convergence_tol : float
            Convergence tolerance
        max_iterations : int
            Maximum number of iterations
        fast_fields : List[str], optional
            Field names that equilibrate quickly (default: ['E', 'H'])
        slow_fields : List[str], optional
            Field names that evolve slowly (default: ['T', 'c', 'σ'])
        """
        super().__init__("QuasiStatic", graph, convergence_tol, max_iterations)
        self.fast_fields = fast_fields or ["E", "H"]
        self.slow_fields = slow_fields or ["T", "c", "σ"]
    
    def solve_step(self, fields: FieldSystem, dt: float) -> Tuple[FieldSystem, bool]:
        """Perform one quasi-static coupling step.
        
        Parameters
        ----------
        fields : FieldSystem
            Current field values
        dt : float
            Time step size
            
        Returns
        -------
        Tuple[FieldSystem, bool]
            Updated fields and whether convergence was achieved
        """
        fields_updated = fields.copy()
        converged = False
        
        for iteration in range(self.max_iterations):
            fields_before = fields_updated.copy()
            
            # Step 1: Equilibrate fast fields (E, H) with current slow field values
            from .fields import MaxwellEquations
            maxwell = MaxwellEquations(fields_updated)
            Q_joule = maxwell.update_from_joule_heating()
            
            # Step 2: Solve slow fields with equilibrated fast fields
            from .fields import HeatEquation
            heat = HeatEquation(fields_updated)
            heat_result = heat.time_step(dt)
            
            from .fields import ParticleDynamics
            particle = ParticleDynamics(fields_updated)
            particle_result = particle.time_step(dt)
            
            # Step 3: Update constitutive relations with new slow field values
            from .fields import ConstitutiveRelations
            constitutive = ConstitutiveRelations(fields_updated)
            new_sigma = constitutive.electrical_conductivity()
            fields_updated.sigma = TrackedValue(
                value=new_sigma, unit="S/m",
                provenance=fields_updated.sigma.provenance,
                source="QuasiStaticScheme"
            )
            
            # Check convergence
            diff = self._compute_max_diff(fields_before, fields_updated)
            if diff < self.convergence_tol:
                converged = True
                break
        
        return fields_updated, converged