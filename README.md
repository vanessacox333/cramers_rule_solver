# cramers_rule_solver

Solves a linear system with 3 unknowns using Cramer's rule. Implements logging to track events throughout the code. The level is set to DEBUG but can be changed when running the file using argparser. If the system is solvable, the solver method will return an array with the solution. The solver method will return a message telling the user if it doesn't have a unique solution.