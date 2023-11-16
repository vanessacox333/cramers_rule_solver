import numpy as np
import logging
import argparse


# create a logger for this file :
logger1 = logging.getLogger("logger1")

# set logging level
logger1.setLevel(logging.DEBUG)

# create file handler to write logs to file
fh = logging.FileHandler('linearSystem.log', 'w')

# format for our output
formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(lineno)d:%(message)s")

# file handler so it knows how to write our output
fh.setFormatter(formatter)

# give logger to file handler
logger1.addHandler(fh)

# create another logger 
logger2 = logging.getLogger("logger2")

# set logging level
logger2.setLevel(logging.DEBUG)

# create file handler to write logs to file
fh = logging.FileHandler('solution.log', 'w')

# #and add it to our file handler so it knows how to write our output:
fh.setFormatter(formatter)

# add to file handler
logger2.addHandler(fh)


parser = argparse.ArgumentParser(description='analyze Auto MPG data set')
parser.add_argument('-l', '--logging', metavar='<set level>', choices=['NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                        help='Choose a logging level.', type=str.upper)
args = parser.parse_args()

if args.logging == 'NOTSET':
  logger1.setLevel(logging.NOTSET)
  logger2.setLevel(logging.NOTSET)
elif args.logging == 'DEBUG':
  logger1.setLevel(logging.DEBUG)
  logger2.setLevel(logging.DEBUG)
elif args.logging == 'INFO':
  logger1.setLevel(logging.INFO)
  logger2.setLevel(logging.INFO)
elif args.logging == 'WARNING':
  logger1.setLevel(logging.WARNING)
  logger2.setLevel(logging.WARNING)
elif args.logging == 'ERROR':
  logger1.setLevel(logging.ERROR)
  logger2.setLevel(logging.ERROR)
elif args.logging == 'CRITICAL':
  logger1.setLevel(logging.CRITICAL)
  logger2.setLevel(logging.CRITICAL)
else:
  pass

class LinearSolver:
   """
   Solves a linear equation with 3 unknowns using Cramer's rule.

    Attributes:
    ----------------------------------------
        system (np.array): 3x3 numpy array
        constants (np.array): 3x1 numpy array

    Methods:
    ----------------------------------------
        __check_system__: ensures system array is 3x3 and constant array is 3x1
        __solver__: solves linear system using Cramer's rule
   """
   def __init__(self, system, constants):
      """
      Initializes LinearSolver object. Note that constants is casted to a 3x1 which
      allows for (3,) arrays to be used to create objects. However, an error will arise
      if constants can't be formatted as a (3,1) array.
      """
      self.system = system
      self.constants = constants.reshape(3,1)
   def __str__(self):
      return f"Linear System: {self.system}, Constants: {self.constants}"
   def check_system(self):
      """
      Checks shape of system and constants. Shape of system must be 3x3
      and shape of constants must be 3x1 or errors will arise.
      """
      logger1.info("Checking system and constants shape")
      if self.system.shape == (3, 3) and self.constants.shape == (3, 1):
         logger1.info("Correct system and constants shape")
         return f"Shape of linear system and constants conducive to solving linear equation with 3 unknowns"
      else:
         logger1.warning("Incorrect system or constants shape")
         return f"Shape of linear system and/or constants are incorrectly shaped to solve linear equation with 3 unknowns"
   def solver(self):
      """
      Checks 3x3 matrix determinant to see if system is solvable. If it is, Cramer's rule is used
      to solve system with 3 unknowns. If system has no solution or no unique solution, a 
      message is returned.
      """
      # get determinant of 3x3 matrix
      D = np.linalg.det(self.system)
      if D != 0:
        logger1.info("Determinant does not equal 0")
        print("System has a unique solution")
        Dx = self.system.copy()
        # this format is an option for if the constants are supposed to be (3,)
        # Dx[:, 0] = self.constants.reshape(3,)
        # this format is for if the constants are supposed to be (3,1)
        Dx[:, 0][:, None] = self.constants
        # get determinant for Dx
        Dx = np.linalg.det(Dx)

        Dy = self.system.copy()
        # [:, None] ensures shape Dy[:, 1] matches self.constants
        Dy[:, 1][:, None] = self.constants
        # get determinant for Dy
        Dy = np.linalg.det(Dy)

        Dz = self.system.copy()
        Dz[:, 2][:, None] = self.constants
        # get determinant for Dz
        Dz = np.linalg.det(Dz)

        solution = np.array([(Dx/D), (Dy/D), (Dz/D)])
        logger2.info(f"Solution: {solution}")
        return solution
   
      else:
         logger1.warning("Determinant=0")
         return f"System does not have a unique solution"


if __name__ == "__main__":
   system = np.array([[5, -14, -3], [1, 2, 2], [-7, 4, 5]])
   constants = np.array([[-39, -2, -29]])
   ex1 = LinearSolver(system, constants)
   # ex1.solver()
   # print(ex1.solver())

   system_no_sol = np.array([[2, -4, 1],[8, -2, 4],[-4, 1, -2]])
   constants_no_sol = np.array([3, 7, -14])
   ex3 = LinearSolver(system_no_sol, constants_no_sol)
   # print(ex3.solver())

# Logging examples:
   # testing loggers
   # logger1.warning("testing...testing")
   # logger1.info("here's some information")
   # logger1.critical("critical thingy")
   # logger2.info("testing2....testing2!!")
   # logger2.critical("something super serious")
   # logger2.debug("debugging")

