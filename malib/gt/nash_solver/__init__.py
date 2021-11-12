from .fictitious_play import fictitious_play
from .nash_solver import DefaultSolver, SelfPlaySolver, FictitiousSelfPlaySolver

SOLVER_LIB = {
    "nash": DefaultSolver,
    "naive_self_play": SelfPlaySolver,
    "fictitious_self_play": FictitiousSelfPlaySolver,
}


def build_solver(solver):
    if isinstance(solver, str):
        assert (
            solver in SOLVER_LIB
        ), f"Invalid solver type, you can only choose from {list(SOLVER_LIB.keys())}"
        return SOLVER_LIB[solver]
    else:
        return solver
