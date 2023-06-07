# SoftStress
For detailed explanations of the software, why we need it, what it can do etc., please read the pdf in this repo (will be available soon). \
SoftStress can learn a stress pattern with a Maximum Entropy learner (Goldwater and Johnson 2003) using the Gradient Descent as its optimization algorithm, combined with Expectation Maximization (Dempster et al. 1977) to deal with the presence of hidden structure (Tesar and Smolensky 2000, Jarosz 2013). The learner is first presented in Pater et al. 2012, and the source code for the learner came from [Brandon Prickett's github](https://github.com/blprickett/Hidden-Structure-MaxEnt). \
It can also compute the minimal set of weights that represent a stress pattern via Linear Programming, introduced in Potts et al 2010 and implemented in OT-Help. The current software can handle larger and more complicated problems. 

You can investigate learnability/representability of any of the 61 attested stress patterns [[click to see these patterns]](https://docs.google.com/spreadsheets/d/1S6ZATuLHsgWLTHFUazIvdQwL-Gkr5H2XmsrDtYzQ8qI/edit#gid=1787957068) using five predefined constraint sets:\
Foot: OriginalTS, RevisedTS, RevisedTS_nonfinmain\
Grid: OriginalGordon, RevisedGordon
For the definitions of these constraints, see the pdf manual.

To see the pattern (winners for each UR), 
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
# [Demo in Google Colab](https://colab.research.google.com/drive/10kKmw0Eeb4F-8F99WxRzdu31Tlcnp6ff?usp=sharing)
```
# Run these lines of code to use SoftStress from a google colab notebook
import os
from subprocess import getoutput
getoutput("git clone -l -s https://github.com/seungsuklee/SoftStress.git cloned-repo")
os.chdir('cloned-repo')
%pip install pulp
from SoftStress import *
```
```
# prints the winner SRs for this pattern
read_winners('hz112', 'QI')
# learns the pattern and prints the learning result
final_weights, learned_when = learn_language('hz112', 'QI', 'Foot', OriginalTS)
print_result_pretty('hz112', 'QI', 'Foot', OriginalTS, final_weights, learned_when)
# solves the pattern and prints the solution(s)
solutions = solve_language('hz112', 'QI', 'Foot', OriginalTS)
print_solutions_pretty('hz112', 'QI', 'Foot', OriginalTS, Constraint_set, solutions)
```

# Constraint sets
```
OriginalTS = [
  FtBin
  , Parse	
  , Iamb
  , FootNonfin
  # , Trochee
  , Nonfin_ft
  # , Nonfin_main
  , WordFootRight
  , WordFootLeft
  , AllFeetRight
  , AllFeetLeft
  , MainRight
  , MainLeft
  # , MainRightSyll
  # , MainLeftSyll
  , WSP_ft
]

RevisedTS = [
  FtBin
  , Parse	
  , Iamb
  # , FootNonfin
  , Trochee
  , Nonfin_ft
  # , Nonfin_main
  , WordFootRight
  , WordFootLeft
  , AllFeetRight
  , AllFeetLeft
  # , MainRight
  # , MainLeft
  , MainRightSyll
  , MainLeftSyll
  , WSP_ft
]

RevisedTS_nonfinmain = [
  FtBin
  , Parse	
  , Iamb
  # , FootNonfin
  , Trochee
  , Nonfin_ft
  , Nonfin_main
  , WordFootRight
  , WordFootLeft
  , AllFeetRight
  , AllFeetLeft
  # , MainRight
  # , MainLeft
  , MainRightSyll
  , MainLeftSyll
  , WSP_ft
]

OriginalGordon = [
  Align1LPrWd
  , Align1RPrWd	
  , Align1Edges
  , Nonfin
  , Lapse
  , ExtendedLapse
  , LapseRight
  , LapseLeft
  , ExtendedLapseRight
  , Clash
  , Align2LPrWd
  , Align2RPrWd
  # , Align2LPrWdSyll
  # , Align2RPrWdSyll
  , WSP
]

RevisedGordon = [
  Align1LPrWd
  , Align1RPrWd	
  , Align1Edges
  , Nonfin
  , Lapse
  , ExtendedLapse
  , LapseRight
  , LapseLeft
  , ExtendedLapseRight
  , Clash
  # , Align2LPrWd
  # , Align2RPrWd
  , Align2LPrWdSyll
  , Align2RPrWdSyll
  , WSP
]
```
If you want to add more constraints or want to learn a stress pattern that's not one of the 61 patterns, please let me know: seungsuklee[at]umass[dot]edu
# Acknowledgement
This research and software development was supported by NSF BCS-2140826 awarded to the University of Massachusetts Amherst.
