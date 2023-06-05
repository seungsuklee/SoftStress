# SoftStress
For detailed explanations of the software, why we need it, what it can do etc., 
please read the pdf in this repo (will be available soon). 
SoftStress can learn a stress pattern with a Maximum Entropy learner (Goldwater and Johnson 2003) using the Gradient Descent as its optimization algorithm, combined with Expectation Maximization (Dempster et al. 1977) to deal with the presence of hidden structure (Tesar and Smolensky 2000, Jarosz 2013). The learner is first presented in Pater et al. 2012, and the source code for the learner came from https://github.com/blprickett/Hidden-Structure-MaxEnt. 
It can also compute the minimal set of weights that represent a stress pattern via Linear Programming, introduced in Potts et al 2010 and implemented in OT-Help. The current software can handle larger and more complicated problems. 

It can learn any of the 61 attested stress patterns (https://docs.google.com/spreadsheets/d/1S6ZATuLHsgWLTHFUazIvdQwL-Gkr5H2XmsrDtYzQ8qI/edit#gid=1787957068) using one of the five predefined constraint sets:\
Foot: OriginalTS, RevisedTS, RevisedTS_nonfinmain\
Grid: OriginalGordon, RevisedGordon

If you want to add more constraints or want to learn a stress pattern that's not one of the 61 patterns, please let me know: seungsuklee[at]umass[dot]edu

To see the patterns (winners for each UR), 
```
read_winners(filename, QI_or_QS)
```
To learn a pattern
```
final_weights, learned_when = learn_language(filename, QI_or_QS, Foot_or_Grid, Constraint_set)
print_result_pretty(filename, QI_or_QS, Foot_or_Grid, Constraint_set, final_weights, learned_when)
```
To solve a pattern (to see if a pattern is representable with the constraint set)
```
solutions = solve_language(filename, QI_or_QS, Foot_or_Grid, Constraint_set)
print_solutions_pretty(filename, QI_or_QS, Foot_or_Grid, Constraint_set, solutions)
```
# Example
```
read_winners('hz112', 'QI')
final_weights, learned_when = learn_language('hz112', 'QI', 'Foot', OriginalTS)
print_result_pretty('hz112', 'QI', 'Foot', OriginalTS, final_weights, learned_when)
solutions = solve_language('hz112', 'QI', 'Foot', OriginalTS)
print_solutions_pretty('hz112', 'QI', 'Foot', OriginalTS, Constraint_set, solutions)
```
# Acknowledgement
This research and software development was supported by NSF BCS-2140826 awarded to the University of Massachusetts Amherst.
