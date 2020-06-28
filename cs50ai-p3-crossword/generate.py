import sys
import copy

from crossword import *
from collections import deque, Counter


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        for var, domain in self.domains.items():
            self.domains[var] = {d for d in domain if len(d) == var.length}

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """

        overlap = self.crossword.overlaps[x, y]
        o_v1 = overlap[0]
        o_v2 = overlap[1]

        keep = set()

        # Algorithm as outlined in the lecture.
        revised = False
        if overlap is not None:
            for dx in self.domains[x]:
                dx_c = dx[o_v1]
                for dy in self.domains[y]:
                    dy_c = dy[o_v2]

                    # Check that the two values for the vars x and y fulfill their constraints
                    # and keep them if they do.
                    if dx_c == dy_c and dx != dy:
                        keep.add(dx)
                        continue

            if keep:
                if keep != self.domains[x]:
                    revised = True
                self.domains[x] = keep

        return revised

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """

        # AC3 algorithm as outlined in the lecture
        if arcs is None:
            arcs = [(var, n) for var in self.domains for n in self.crossword.neighbors(var)]

        queue = deque(arcs)

        while queue:
            x, y = queue.popleft()
            if self.revise(x, y):
                if len(self.domains[x]) == 0:
                    return False
                for z in self.crossword.neighbors(x) - {y}:
                    queue.append((z, x))

        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """

        # Check if all variables are contained in the assignment
        if set(assignment.keys()) != self.crossword.variables:
            return False

        # Check if all variables have a value
        for value in assignment.values():
            if value is None:
                return False

        return True

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """

        # Check if all of the arcs in assignment are arc consistent, meaning that
        # the overlaps of all assigned variables should have correct letters.
        for x, x_value in assignment.items():
            neighbors = self.crossword.neighbors(x)

            # Check the neighbors of x that are contained in the assignment
            for neighbor in neighbors.intersection(assignment.keys()):
                ox, on = self.crossword.overlaps[x, neighbor]
                if x_value[ox] != assignment[neighbor][on] or x_value == assignment[neighbor]:
                    return False

        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """

        # Check the amount of neighbors that contain the value dx for var ->
        # if we assign the value dx for var, it gets removed from those neighbors
        neighbors = self.crossword.neighbors(var)
        c = []
        for dx in self.domains[var]:
            n = 0
            # Go through all neighbors minus the ones already assigned.
            for neighbor in neighbors - assignment.keys():
                ox, on = self.crossword.overlaps[var, neighbor]

                # Go through every possible value and check if its eliminated
                # from the neighbors domain if dx is chosen.
                for value in self.domains[neighbor]:
                    if dx[ox] != value[on]:
                        n += 1

            c.append((dx, n))

        # Sort by the amount of neighbors that a value rules out.
        c.sort(key=lambda pair: pair[1])
        return [var for var, value in c]

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        remaining = [(var, len(self.domains[var])) for var in self.crossword.variables - assignment.keys()]

        # Sort by domain size
        remaining.sort(key=lambda pair: pair[1])

        # The first item in the sorted remaining variables set is the variable with the smallest domain.
        min_domain = remaining[0]

        # Count occurrences of domains sizes to check whether there are multiple variables with the smallest domain
        # value.
        c = Counter(elem[1] for elem in remaining)

        if c[min_domain[1]] == 1:
            # If there's only one element with the minimum domain size, return it.
            return remaining[0][0]
        else:
            # Sort tied variables by number of neighbors.
            tied = remaining[:c[min_domain[1]]]
            tied.sort(key=lambda pair: self.crossword.neighbors(pair[0]))
            return tied[0][0]

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        if self.assignment_complete(assignment):
            return assignment

        var = self.select_unassigned_variable(assignment)
        for value in self.order_domain_values(var, assignment):
            new_assignment = assignment.copy()
            new_assignment[var] = value
            if self.consistent(new_assignment):
                result = self.backtrack(new_assignment)
                if result is not None:
                    return result
        return None


def main():
    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
