import pdb
from collections import defaultdict

def num_to_str(num):
    str_rep = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    return str_rep[num]

def create_references(max_dist = 2, verbose = False):
    references = defaultdict(lambda: [])
    references[( (-2,0), (-1,0) )].append('reach <GOAL> between <OBJ1> and <OBJ2>')
    references[( (2, 0), (1, 0) )].append('reach <GOAL> between <OBJ1> and <OBJ2>')
    references[( (0,-2), (0,-1) )].append('reach <GOAL> between <OBJ1> and <OBJ2>')
    references[( (0, 2), (0, 1) )].append('reach <GOAL> between <OBJ1> and <OBJ2>')

    for i in range(-2, 3):
        for j in range(-2, 3):
            for goal_i in range(-2, 3):
                for goal_j in range(-2, 3):
                    if goal_i != i and goal_j != j and goal_i != 0 and goal_j != 0:
                        
                        if abs(goal_j - j) > max_dist or abs(goal_i - i) > max_dist:
                            continue
                        ## i with reference to obj 1
                        ## j with reference to obj 2
                        row = goal_i
                        col = goal_j - j
                        if row > 0:
                            vertical = '{} below <OBJ1>'.format( num_to_str(row) )
                        elif row < 0:
                            vertical = '{} above <OBJ1>'.format( num_to_str(abs(row)) )
                        else:
                            raise RuntimeError('goal_i should not be in line with obj 1')
                            
                        if col > 0:
                            horizontal = '{} to the right of <OBJ2>'.format( num_to_str(col) )
                        elif col < 0:
                            horizontal = '{} to the left of <OBJ2>'.format( num_to_str(abs(col)) )
                        else:
                            raise RuntimeError('goal_j should not be in line with obj 2')

                        if verbose:
                            print 'OBJ2: ', i, j, '    Goal: ', goal_i, goal_j

                        instructions = 'reach <GOAL> ' + vertical + ' and ' + horizontal 
                        references[( (i,j), (goal_i,goal_j) )].append(instructions)
                        
                        if verbose:
                            print '    ', instructions

                        ## i with reference to obj 2
                        ## j with reference to obj 1
                        row = goal_i - i
                        col = goal_j
                        if row > 0:
                            vertical = '{} below <OBJ2>'.format( num_to_str(row) )
                        elif row < 0:
                            vertical = '{} above <OBJ2>'.format( num_to_str(abs(row)) )
                        else:
                            raise RuntimeError('goal_i should not be in line with obj 2')
                            
                        if col > 0:
                            horizontal = '{} to the right of <OBJ1>'.format( num_to_str(col) )
                        elif col < 0:
                            horizontal = '{} to the left of <OBJ1>'.format( num_to_str(abs(col)) )
                        else:
                            raise RuntimeError('goal_j should not be in line with obj 1')

                        instructions = 'reach <GOAL> ' + vertical + ' and ' + horizontal 
                        
                        if verbose:
                            print '    ', instructions

                        references[( (i,j), (goal_i,goal_j) )].append(instructions)

    return references

                          
if __name__ == '__main__':
    references = create_references()
    pdb.set_trace()






