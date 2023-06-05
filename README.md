# SoftStress
# Acknowledgement
This research and software development was supported by NSF BCS-2140826 awarded to the University of Massachusetts Amherst.

For detailed explanations of the software, the assumptions, why we need it, what it can do etc., 
please read the pdf in this repo (will be available soon). 

5 constraint sets:
Foot: OriginalTS, RevisedTS, RevisedTS_nonfinmain
Grid: OriginalGordon, RevisedGordon

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
# example
```
read_winners('hz112', 'QI')
final_weights, learned_when = learn_language('hz112', 'QI', 'Foot', OriginalTS)
print_result_pretty('hz112', 'QI', 'Foot', OriginalTS, final_weights, learned_when)
solutions = solve_language('hz112', 'QI', 'Foot', OriginalTS)
print_solutions_pretty('hz112', 'QI', 'Foot', OriginalTS, Constraint_set, solutions)
```
