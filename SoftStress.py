# -*- coding: utf-8 -*-
"""
Seung Suk Lee 081623 update

# Acknowledgment
This research and software development was supported by NSF BCS-2140826 
awarded to the University of Massachusetts Amherst.

Special thanks to Joe Pater, Brandon Prickett.
"""

####################################################
# pkgs
####################################################

# necessary pkgs
from pulp import *
import numpy as np
import pandas as pd
import mpmath
from itertools import product, combinations_with_replacement
from more_itertools import partitions
from collections import defaultdict, Counter
from re import sub, search
from tqdm.notebook import tqdm
from os import listdir, path, mkdir
import pickle
from google.colab import files
import urllib.request

####################################################
# representation fns
####################################################

def convert2brandon(representation, mode= 'SR'):
  # convert a np.array representation of data into Brandon's format:
  # e.g. [[0, 0], [0, 2]] --> [L L1]
  form = '['

  def weight2brandon(digit):
    if digit == 0:
      return 'L'
    elif digit == 1:
      return 'H'
    elif digit == 2:
      return 'S'

  def stress2brandon(digit):
    if digit == 0:
      return ''
    elif digit == 1:
      return '2'
    elif digit == 2:
      return '1'
    return form

  if mode=='SR':
    for w, s in zip(representation[0], representation[1]):
      form += weight2brandon(w)
      form += stress2brandon(s)
      form += ' '
    form = form[:-1]
    form+=']'
    return form

  elif mode=='UR':
    for w in representation:
      form += weight2brandon(w)
      form += ' '
    form = form[:-1]
    form+=']'
    return form

def brandon2st2(SR):
  # convert a Brandon's representation of data into np.array format:
  # e.g. [L L1]  --> [[0, 0], [0, 2]]

  SR = SR.split(" ")
  SR = [form.strip(r'[,]') for form in SR]
  weight = []
  stress = []
  for form in SR:
    s = search('\d', form)
    if s==None:
      stress.append(0)
    else:
      s = s.group(0)
      if s == '2': # Brandon's secondary stress notation
        stress.append(1)
      elif s == '1': # Brandon's primary stress notation
        stress.append(2)
    w = search('\w', form).group(0)
    if w == 'L':
      weight.append(0)
    elif w == 'H':
      weight.append(1)
    elif w == 'S':
      weight.append(2)

  weight = tuple(weight)
  stress = tuple(stress)
  return weight, stress


####################################################
# GEN
####################################################

def gen_UR(syll_len, weight=2, TS_style=True):
  '''
  given a syllable length, and a weight param
  Generates all combination of light and heavy (and superheavy) syllables
  weight = 1 is all light syllables
  weight = 2 has 2 levels of weight: heavy, light = 1, 0
  weight = 3 has 3 levels of weight: superheavy, heavy, light = 2, 1, 0
  if brandonian=True (default), return Ls and Hs not a np arary
  if TS_style=True (default), then only return LLLLLL and LLLLLLL (6 & 7 syll len)
  '''
  if syll_len<6:
    weight = [np.array(i, dtype=int) for i in product(range(weight), repeat=syll_len)]
  else:
    weight = [np.zeros(syll_len, dtype=int)]
  return weight

def gen_URlist(QI_or_QS):
  '''
  given QI or QS,
  create a list of URs
  for QI: 6 URs, light syllables of length 2-7
  for QS: 62 URs, light/heavy for length 2-5 and light of length 6 & 7
  '''
  if QI_or_QS=='QI':
    return [gen_UR(i,1)[0] for i in range(2,8)]
  elif QI_or_QS=='QS':
    UDLs = []
    for i in range(2,8):
      UDLs.extend(gen_UR(i))
    return UDLs

def getURfromSR(sfc):
  '''
  given a surface form (np array),
  return its UR form
  '''
  sfc = sfc.split(" ")
  sfc = [form.strip(r'[,]') for form in sfc]
  weight = []
  for form in sfc:
    w = search('\w', form).group(0)
    if w == 'L':
      weight.append(0)
    elif w == 'H':
      weight.append(1)
    elif w == 'S':
      weight.append(2)
  return convert2brandon(np.array(weight), 'UR')

def gen_SR(udl): # udl means underlying
  '''
  given a UR (np array)
  return a list of SRs for that UR
  '''
  stress = [i for i in product(range(3), repeat=len(udl))]
  stress = [np.array(i, dtype=int) for i in stress if i.count(2)==1]
  surface = []
  for sfc in stress:
    surface.append(np.stack((udl, sfc)))
  return surface

def gen_HR(sfc): # sfc means surface
  '''
  gien a surface form,
  return a list of tuples:
  containing all possible ways in which
  syllable indices can be parsed into feet
  for each tuple,
  first: SR
  second: feet parsing
  '''

  udl_length = len(sfc[0])
  all_prosodifications = []
  for parts in partitions(range(udl_length)): # more_itertools pkg
    tooLong = False # prune out any prosodifications grouping more than 2 sylls
    for part in parts:
      if len(part)>2:
        tooLong = True
    if tooLong == False:
      all_prosodifications.append(parts)

  # first change the surface form so that there's no 2
  new_sfc = []
  for x in sfc[1]:
    if x == 2:
      new_sfc.append(1)
    else:
      new_sfc.append(x)

  # based on the surface form, only include the footings that make sense
  # i.e. only include the footings that has the form of (1/2, 0), (0, 1/2), (1/2)
  HRs = []
  for footings in all_prosodifications:
    has_stressed = []
    isGood = True # flag to prune out bad partition options

    for ft in footings: # indicies for where ft starts and ends in the surface form
      start = ft[0]
      end = ft[-1]+1
      ft_structure = sum(new_sfc[start:end]) # see what the foot looks like
      # print(ft_structure)
      if ft_structure>1: # no (1,1), (1,2), (2,1)
        isGood = False
      if ft_structure>0: # no (0,0), (0)
        has_stressed.append(ft)
    if isGood ==True:
      if (sfc, has_stressed) not in HRs: # sometimes they end up looking the same, so prune'em out
        HRs.append((sfc, has_stressed))
  return HRs

def full_cands(udl):
  '''
  given a UR,
  return a tuple object:
  first item: a list of tuples (HR list)
  second item: list of indexes that indicate HR-SR mapping
  '''
  sfcs = gen_SR(udl)
  cands = []
  for i in range(len(sfcs)):
    cur_sfc = sfcs[i]
    cands.extend(gen_HR(cur_sfc))

  hrs = [convert2brandon(hr[0]) for hr in cands]
  sfcs_brandon = [convert2brandon(sfc) for sfc in sfcs]
  return cands

####################################################
# reading data files
####################################################

# mannual checking of qi/qs
qi_fsa=["hz111",
"hz112",
"hz113",
"hz115",
"hz117",
"hz118",
"hz119",
"hz128",
"hz129",
"hz131",
"hz133",
"hz134",
"hz136",
"hz144",
"hz145",
"hz150",
"hz157",
"hz158",
"hz169",
"hz170",
"hz171",
"hz179",
"hz191",
"hz208",
"hz211",
"hz212",
"hz213",
"hz215"]

qs_fsa = [
"hz114",
"hz116",
"hz127",
"hz132",
"hz139",
"hz146",
"hz148",
"hz149",
"hz153",
"hz156",
"hz159",
"hz162",
"hz165",
"hz167",
"hz172",
"hz176",
"hz177",
"hz178",
"hz182",
"hz184",
"hz192",
"hz194",
"hz196",
"hz199",
"hz200",
"hz203",
"hz204",
"hz205",
"hz209",
"hz210",
"hz214",
"hz218",
"hz219"
]

def read_input_files(filename):
  '''
  given a filename,
  return all the SR candidates and the probability assigned to them
  '''
  surface = []
  probs = []

  # read from my website
  file_path = "https://people.umass.edu/seungsuklee/files/FSA/" + filename + ".csv"
  inputfile = pd.read_csv(file_path)
  surface = inputfile.loc[inputfile['SR']==inputfile['SR']]['SR'].tolist()
  probs = inputfile.loc[inputfile['p']==inputfile['p']]['p'].tolist()

  return surface, probs

def read_input_list(langname):
  '''
  given a language name, return the winner SRs
  Palestinian_Arabic, Najrani_Arabic, Moroccan_Arabic, Jordanian_Arabic,
  Iraqi_Arabic, Classical_Arabic_McCarthy, Classical_Arabic_Abdo, Algerian_Arabic
  '''
  file_path = "https://people.umass.edu/seungsuklee/files/otherLangs/" + langname + ".txt"
  winners=[]
  for line in urllib.request.urlopen(file_path):
    winners.append(line.decode('utf-8').strip())
  return winners

read_input_list('Moroccan_Arabic')

def filter_list(l, ids):
  '''
  filter a list using a list of ids
  return a list with only the items that are of interest
  '''
  return [f for i, f in enumerate(l) if i in ids]

def get_winners(surface, probs):
  '''
  given a list of SRs and a probability vector,
  return a list of SRs that have the probability of 1,
  i.e. the winners
  '''
  def get_winner_indices(ListOfProbs):
    ids = []
    for i, f in enumerate(ListOfProbs):
      if f == 1:
        ids.append(i)
    return ids

  winner_indices = get_winner_indices(probs)
  winner_sr = filter_list(surface, winner_indices)
  return winner_sr

def read_winners(filename, QI_or_QS):
  '''
  given a filename and QI/QS,
  return a list of winners (either for 6 urs or 62 urs)
  '''
  if filename in qi_fsa or filename in qs_fsa:
    long_surface, long_probs = read_input_files(filename)
    winners = get_winners(long_surface, long_probs)
  else:
    winners = read_input_list(filename)

  if QI_or_QS=='QS':
    return winners
  elif QI_or_QS=='QI':
    winners_qi = []
    for w in winners:
      if 'H' not in w:
        winners_qi.append(w)
    return winners_qi

####################################################
# CON
####################################################


####################################################
# Grid Constraints
####################################################

# Align1LPrWd, Align1RPrWd
def Align1LPrWd(form):
  s = form[1]
  return sum(np.where(s>0)[0])

def Align1RPrWd(form):
  s = form[1]
  s = np.flip(s, axis=0)
  return sum(np.where(s>0)[0])

# Align1Edges
def Align1Edges(form):
  s = form[1]
  if s[0]<1 and s[-1]<1:
    return 2
  elif s[0]<1 or s[-1]<1:
    return 1
  else:
    return 0

# Align2LPrWD Align2RPrWd
def Align2LPrWd(form):
  s = form[1]
  # identify the location of primary stress
  pri = np.where(s==2)[0][0]

  # clip the stress string
  s = s[0:pri]
  # count the intervening secondary stresses
  return len(np.where(s==1)[0])

# Align2LPrWD Align2RPrWd
def Align2RPrWd(form):
  s = form[1]
  s = np.flip(s, axis=0)

  # identify the location of primary stress
  pri = np.where(s==2)[0][0]

  # clip the stress string
  s = s[0:pri]
  # count the intervening secondary stresses
  return len(np.where(s==1)[0])

# Align2LPrwDSyll Align2RPrWdSyll
def Align2LPrWdSyll(form):
  s = form[1]
  # identify the location of primary stress
  pri = np.where(s==2)[0][0]

  # clip the stress string
  s = s[0:pri]

  # count the intervening syllables
  return len(s)

# Align2LPrwDSyll Align2RPrWdSyll
def Align2RPrWdSyll(form):
  s = form[1]
  s = np.flip(s, axis=0)

  # identify the location of primary stress
  pri = np.where(s==2)[0][0]

  # clip the stress string
  s = s[0:pri]

  # count the intervening syllables
  return len(s)

# Nonfin
def Nonfin(form):
  s = form[1]
  if s[-1]>0:
    return 1
  else:
    return 0

def Lapse(form):
  s = form[1]
  seq = np.array([0,0])

  # solution from https://stackoverflow.com/a/36535397
  # Store sizes of input array and sequence
  Na, Nseq = s.size, seq.size

  # Range of sequence
  r_seq = np.arange(Nseq)

  # Create a 2D array of sliding indices across the entire length of input array.
  # Match up with the input sequence & get the matching starting indices.
  M = (s[np.arange(Na-Nseq+1)[:,None] + r_seq] == seq).all(1)

  return sum(M)

def ExtendedLapse(form):
  s = form[1]
  seq = np.array([0,0,0])


  # solution from https://stackoverflow.com/a/36535397
  # Store sizes of input array and sequence
  Na, Nseq = s.size, seq.size

  # Range of sequence
  r_seq = np.arange(Nseq)

  # Create a 2D array of sliding indices across the entire length of input array.
  # Match up with the input sequence & get the matching starting indices.
  M = (s[np.arange(Na-Nseq+1)[:,None] + r_seq] == seq).all(1)

  return sum(M)

def Clash(form):
  # same as lapse, but seq = np.array([1,1])
  s = form[1]

  # change primary stress (2) to secondary (1)
  # so that all you need to find is the sequence [1,1]
  new_s = []
  for x in s:
    if x == 2:
      new_s.append(1)
    else:
      new_s.append(x)
  s = np.array(new_s)

  seq = np.array([1,1])

  # solution from https://stackoverflow.com/a/36535397
  # Store sizes of input array and sequence
  Na, Nseq = s.size, seq.size

  # Range of sequence
  r_seq = np.arange(Nseq)

  # Create a 2D array of sliding indices across the entire length of input array.
  # Match up with the input sequence & get the matching starting indices.
  M = (s[np.arange(Na-Nseq+1)[:,None] + r_seq] == seq).all(1)

  return sum(M)

def LapseLeft(form):
  s = form[1]
  lapses = (s[0]==0 and s[1]==0)
  if lapses:
    return 1
  else:
    return 0

def LapseRight(form):
  s = form[1]
  s = np.flip(s, axis=0)

  lapses = (s[0]==0 and s[1]==0)
  if lapses:
    return 1
  else:
    return 0

def ExtendedLapseRight(form):
  s = form[1]
  s = np.flip(s, axis=0)

  lapses = (s[0]==0 and s[1]==0 and s[2]==0)

  if lapses:
    return 1
  else:
    return 0

def AlignL1Left(form):
  weight = np.array(form[0])
  stress = np.array(form[1])
  # checking if there's L1
  main_id = np.where(stress==2)
  if weight[main_id]==0:
    return main_id[0][0]
  else:
    return 0

def AlignL1Right(form):
  weight = np.array(form[0])
  stress = np.array(form[1])
  # checking if there's L1
  main_id = np.where(stress==2)
  if weight[main_id]==0:
    return len(stress)-1-main_id[0][0]
  else:
    return 0

# WSP
def WSP(form):
  # count the stressless heavy
  ww = form[0]
  ss = form[1]
  heavys = np.where(ww>0)[0]
  violations = 0
  for i in heavys:
    if ss[i]==0:
      violations +=1
  return violations

def WSP_superheavy(form):
  # count the stressless superheavy
  ww = form[0]
  ss = form[1]
  superheavys = np.where(ww>1)[0]
  violations = 0
  for i in superheavys:
    if ss[i]==0:
      violations +=1
  return violations

####################################################
# Foot Constraints
####################################################

def FtBin(form):
  weight = form[0][0]
  stress = form[0][1]
  feet = form[1]
  counter = 0
  for ft in feet:
    if len(ft)<2 and weight[ft]<1:
      counter+=1
  return counter

def Parse(form):
  stress = form[0][1]
  feet = form[1]
  counter = 0
  for ft in feet:
    counter+=len(ft) #count parsed sylls
  return len(stress)-counter

def Iamb(form):
  stress = form[0][1]
  feet = form[1]
  counter = 0
  for ft in feet:
    if len(ft)==2:
      if stress[ft[-1]] == 0:
        counter+=1
  return counter

def FootNonfin(form):
  '''
  Assign a violation for each foot
  if its final syllable is stressed
  (every monosyllabic foot gets one violation).
  '''
  stress = form[0][1]
  feet = form[1]
  counter = 0
  for ft in feet:
    if stress[ft[-1]] > 0:
      counter+=1
  return counter

def Trochee(form):
  stress = form[0][1]
  feet = form[1]
  counter = 0
  for ft in feet:
    if len(ft)==2:
      if stress[ft[-1]] > 0:
        counter+=1
  return counter

def Nonfin_ft(form):
  '''
  Assign a violation if the final syllable of a word is footed.
  '''
  stress = form[0][1]
  feet = form[1]
  fin = len(stress)-1
  finalft = feet[-1]
  if fin in finalft:
    return 1
  else:
    return 0

def Nonfin_main(form):
  '''
  a violation if the final syllable has main (i.e., primary) stress
  and no violations if the main stress is anywhere else
  '''
  stress = form[0][1]
  fin = len(stress)-1
  if stress[fin]==2:
    return 1
  else:
    return 0

def WordFootRight(form):
  stress = form[0][1]
  feet = form[1]
  edgesyll = len(stress)-1
  edgeft = feet[-1]
  if edgesyll in edgeft:
    return 0
  else:
    return 1

def WordFootLeft(form):
  stress = form[0][1]
  feet = form[1]
  edgesyll = 0
  edgeft = feet[0]
  if edgesyll in edgeft:
    return 0
  else:
    return 1

def AllFeetRight(form):
  stress = form[0][1]
  feet = form[1]
  counter = 0
  edgesyll = len(stress)-1
  for ft in feet:
    counter+=edgesyll-ft[-1]
  return counter

def AllFeetLeft(form):
  stress = form[0][1]
  feet = form[1]
  counter = 0
  for ft in feet:
    counter+=ft[0]
  return counter

def MainRight(form):
  stress = form[0][1]
  feet = form[1]
  prisyll = np.where(stress==2)[0][0]
  for ft in feet:
    if prisyll in ft:
      headft = ft
  return len(stress) - 1 - headft[-1]

def MainLeft(form):
  stress = form[0][1]
  feet = form[1]
  prisyll = np.where(stress==2)[0][0]
  for ft in feet:
    if prisyll in ft:
      headft = ft
  return headft[0]

def MainRightSyll(form):
  # same as Align2RPrWdSyll
  s = form[0][1] # except this line
  s = np.flip(s, axis=0)

  # identify the location of primary stress
  pri = np.where(s==2)[0][0]

  # clip the stress string
  s = s[0:pri]

  # count the intervening syllables
  return len(s)

def MainLeftSyll(form):
  # same as Align2LPrWdSyll
  s = form[0][1] # except this line
  # identify the location of primary stress
  pri = np.where(s==2)[0][0]

  # clip the stress string
  s = s[0:pri]

  # count the intervening syllables
  return len(s)

def AlignL1Left_ft(form):
  stress = form[0][1]
  weight = form[0][0]
  # checking if there's L1
  main_id = np.where(stress==2)
  if weight[main_id]==0:
    return main_id[0][0]
  else:
    return 0

def AlignL1Right_ft(form):
  stress = form[0][1]
  weight = form[0][0]
  # checking if there's L1
  main_id = np.where(stress==2)
  if weight[main_id]==0:
    return len(stress)-1-main_id[0][0]
  else:
    return 0

def WSP_ft(form):
  ww = form[0][0]
  ss = form[0][1]
  heavys = np.where(ww>0)[0]
  violations = 0
  for i in heavys:
    if ss[i]==0:
      violations +=1
  return violations

def WSP_superheavy_ft(form):
  ww = form[0][0]
  ss = form[0][1]
  superheavys = np.where(ww>1)[0]
  violations = 0
  for i in superheavys:
    if ss[i]==0:
      violations +=1
  return violations

####################################################
# Eval
####################################################


# violation matrix = Candidate X Constraint
def evaluate(ListOfCands, ListOfCons):
  violations = []
  for i in range(len(ListOfCons)):
    violations.append(list(map(ListOfCons[i], ListOfCands)))
  return np.array(violations).T

####################################################
# Learning
####################################################
"""
Excel demo:
https://docs.google.com/spreadsheets/d/1ahyTertC-ibLyalhpmEf9vhtafZ1X5rJ/edit#gid=1008380569

Math note: https://drive.google.com/drive/u/0/folders/1L_wsyNyTekWVJv3LwfGOMV9WUAJVLzWE
""" 

def prepare_data_for_learning(QI_or_QS, Foot_or_Grid):
  '''
  given QI/QS and Foot/Grid
  , ur = unique_underlying (list of all URs in brandonian form)
  , sr = unique_surface (list of all SRs in brandonian form)
  , cands = ListOfCandidates (list of all candidates in st2 form to be assigned viols)
  , data = ListOfSurfaces (list of SRs in the length of HRs (i.e., with repetitions), in brandonian form
      later to be used in gradient descent function)
  , ur2datum = underlying2data (dictionary for each UR (brandonian), how many HRs it has, as ranges)
  , sr2datum = surface2data (dictionary for each SR (brandonian), how many HRs it has, as ranges)
  '''
  UDLs = gen_URlist(QI_or_QS)

  ListOfSurfaces = []
  ListOfCandidates = []

  for udl in UDLs:
    ListOfSurfaces.extend(gen_SR(udl)) # list of all SR
    ListOfCandidates.extend(full_cands(udl)) # list of all HR - Foot
  unique_underlying = [convert2brandon(form, 'UR') for form in UDLs] #turn UDLs to brandon form
  unique_surface = [convert2brandon(form) for form in ListOfSurfaces] #turn ListOfSurfaces to brandon form

  # initiate dictionaries {ur:[]} and {sr:[]} in the list of candidates
  underlying2data = {form:[] for form in unique_underlying}
  surface2data = {form:[] for form in unique_surface}

  if Foot_or_Grid =='Grid': # list of candidates is unique surface
    for datum_index, form in enumerate(unique_surface):
      underlying2data[getURfromSR(form)].append(datum_index)
      surface2data[form].append(datum_index) # this is for assigning violations, each sr has 1 index
    return unique_underlying, unique_surface, ListOfSurfaces, unique_surface, underlying2data, surface2data

  elif Foot_or_Grid=='Foot': # list of candidates is list of HRs
    for datum_index, form in enumerate(ListOfCandidates):
      underlying2data[convert2brandon(form[0][0], 'UR')].append(datum_index)
      surface2data[convert2brandon(form[0])].append(datum_index) # this is for assigning violations

    ListOfSurfaces = [convert2brandon(form[0]) for form in ListOfCandidates] # this is later to loop through/SRs in length of HRs
    return unique_underlying, unique_surface, ListOfCandidates, ListOfSurfaces, underlying2data, surface2data

def get_normalized_probs(winners, surface2data, unique_underlying, data):
  '''
  Note: 'data' is unique_surface (Grid) or ListOfSurfaces (Foot)
  '''
  probs=[]
  for sfc in data:
    if sfc in winners:
      probs.append(1)
    else:
      probs.append(0)
  probs = np.array(probs) # length = Number of HRs

  new_probs = []
  for datum_index, this_prob in enumerate(probs):
    new_prob = this_prob/(len(surface2data[data[datum_index]])*len(unique_underlying))
    # default: 1/NumberOfUR, but also weighted by how many HRs exist for that SR
    new_probs.append(new_prob)
  return np.array(new_probs)

def initialize_weights(constraint_number, init_weight, rand_weights):
  if rand_weights:
    initial_weights = list(np.random.uniform(low=0.0, high=10.0, size=constraint_number))
    #Init constraint weights = rand 1-10
    print("Initial weights: ", initial_weights)
  else:
    initial_weights = [init_weight] * constraint_number
  return initial_weights

def get_predicted_probs(weights, viols, unique_underlying, underlying2data):
  # maybe later check: unique_underlying just ur2datum keys
  harmonies = viols.dot(weights)
  eharmonies = np.exp(harmonies)
  #Calculate denominators to convert eharmonies to predicted probs:
  Zs = np.array([mpmath.mpf(0.0) for z in range(viols.shape[0])])
  for underlying_form in unique_underlying:     #Sum of eharmonies for this UR (converts to probs)
    this_Z = sum(eharmonies[underlying2data[underlying_form]])\
                                    *float(len(unique_underlying)) #Number of UR's (normalizes the updates)
    if this_Z == 0:
      eharmonies = np.array([mpmath.exp(h) for h in harmonies])
      this_Z = sum(eharmonies[underlying2data[underlying_form]])\
                                  *float(len(unique_underlying))
      Zs[underlying2data[underlying_form]] = this_Z
    else:
      Zs[underlying2data[underlying_form]] = mpmath.mpf(this_Z)

  #Calculate prob for each datum:
  probs = []
  for datum_index, eharm in enumerate(eharmonies):
    if Zs[datum_index] == 0:
        #error_f = open("_weights.csv", "w")
        #error_f.write("\n".join([str(w) for w in weights]))
        #error_f.close()
        #print("\n\n"+remember_me+"\n\n")
      raise Exception("Rounding error! (Z=0 for "+unique_underlying[datum_index]+")")
    else:
      probs.append(float(eharm/Zs[datum_index]))

  return np.array(probs)

def check_learning(QI_or_QS, Foot_or_Grid, cur_weights, ListOfConFns, winners):
  '''
  given a weight vector and a list of observed winnerSR for each UR,
  if the weight vector allows the observed winnerSR to have .90 prob,
  return True else False
  '''
  UDLs = gen_URlist(QI_or_QS)
  for i in range(len(UDLs)):
    cur_UR = UDLs[i] # loop through each UR
    if Foot_or_Grid == 'Grid':
      candidates = gen_SR(cur_UR)
      surfaces = [convert2brandon(form) for form in candidates]
    elif Foot_or_Grid == 'Foot':
      candidates = full_cands(cur_UR)
      surfaces = [convert2brandon(form[0]) for form in candidates]

    winner_ids = []
    cur_winner = winners[i]
    for id, sfc in enumerate(surfaces):
      if sfc==cur_winner:
        winner_ids.append(id)

    viol_vec = -1*evaluate(candidates, ListOfConFns)
    harmonies = viol_vec.dot(cur_weights)
    eharmonies = np.exp(harmonies)
    Z = sum(eharmonies)
    probs = eharmonies/Z
    if sum(probs[winner_ids])<.90:
      return False
  return True

def learn_language(filename, QI_or_QS, Foot_or_Grid, my_cons, rand_weights = False, init_weights = 1, neg_weights = False, epochs = 1000, eta=4.):
  CON_num = len(my_cons)

  ur, sr, cands, DATA, ur2datum, sr2datum = prepare_data_for_learning(QI_or_QS, Foot_or_Grid)
  v = evaluate(cands, my_cons) * -1
  weights = initialize_weights(CON_num, init_weights, rand_weights)

  cur_winners = read_winners(filename, QI_or_QS)
  td_probs = get_normalized_probs(cur_winners, sr2datum, ur, DATA)

  if epochs==0:
    return weights

  learned_when=-1
  for epoch in tqdm(range(epochs)):
    if epoch!=0:
      weights = np.copy(new_weights)

    #Forward pass:
    le_probs = get_predicted_probs(weights, v, ur, ur2datum)

    #Weight the td_probs, based on what we know about the
    #different hidden structures:
    sr2totalLEProb = {form:sum(le_probs[sr2datum[form]]) for form in sr2datum.keys()} #Sums expected SR probs (merging different HR's)
    sr2totalTDProb = {form:sum(td_probs[sr2datum[form]]) for form in sr2datum.keys()} #Sums remaining data SR probs (merging different HR's)
    weighted_tdProbs = []
    for datum_index, le_p in enumerate(le_probs):
      if sr2totalLEProb[DATA[datum_index]]==0.0:
            #exit("Got a zero when you didn't want one!")
        HR_givenSR = 0.0
      else:
        HR_givenSR = le_p/sr2totalLEProb[DATA[datum_index]] #How likely is the HRdata, given our current grammar
      weighted_tdProbs.append(HR_givenSR * sr2totalTDProb[DATA[datum_index]]) #Weight the HR probs in the training data by our current estimation of HR probs

    #Backward pass:
    TD = v.T.dot(weighted_tdProbs) #Violations present in the training data
    LE = v.T.dot(le_probs) #Violations expected by the learner
    gradients = (TD - LE)

    #Update weights:
    updates = gradients * eta
    new_weights = weights + updates

    #Police negative weights:
    if not neg_weights:
      new_weights = np.maximum(new_weights, 0)

    # # check learned yet?
    if check_learning(QI_or_QS, Foot_or_Grid, new_weights, my_cons, cur_winners):
      learned_when = epoch+1 # epoch starts from 0 so, add 1 to be non-pythonic
      break # stop learning once learned
    else: # continue with learning if not yet learned
      pass

  return new_weights, learned_when

####################################################
# Solver
####################################################

"""
SoftStress manual: https://docs.google.com/document/d/1EIIdUaD7rKdoNN7TuV5Kcmxve9g11dLQIZaGQ3_e_r4/edit
""" 

def add_tableau(Foot_or_Grid, LP, udl, winner, ListOfConFns, DictOfCons, alpha):
  # generate candidates based on the UR given
  if Foot_or_Grid=='Foot':
    cands = full_cands(udl)
  elif Foot_or_Grid=='Grid':
    cands = gen_SR(udl)

  # find the index of the winner
  for i in range(len(cands)):
    # print(i, '\n', str(cands[i]), '?== \n', str(winner))
    if str(cands[i]) == str(winner):
      # print("I found my winner", i)
      winner_id = i
      break

  # constraint names
  ListOfConNames = [fn.__name__ for fn in ListOfConFns]
  # make the violvec here using the cands
  violvec = evaluate(cands, ListOfConFns)

  # loop through the candidates
  for loser_id in range(len(cands)):
    # skipping the winner id, put loser candidate on the left side
    # put winner candidate on the right side for each loser candidate
    if loser_id != winner_id:
      LP += (
          # losing side: lpSum does sum product
          lpSum([violvec[loser_id][ListOfConNames.index(i)] * DictOfCons[i] for i in ListOfConNames])
          >=
          # winning side (margin of separation (alpha) = 1 by default)
          alpha + lpSum([violvec[winner_id][ListOfConNames.index(i)] * DictOfCons[i] for i in ListOfConNames])

          # , convert2brandon(winner) + " vs " + convert2brandon(cands[loser_id])
      )
  return LP

def solve_language(filename, QI_or_QS, Foot_or_Grid, ListOfConFns):
  UDLs = gen_URlist(QI_or_QS)
  winners = [np.array(brandon2st2(s)) for s in read_winners(filename, QI_or_QS)]
  ListOfConNames = [fn.__name__ for fn in ListOfConFns]
  DictOfCons = LpVariable.dicts("con", ListOfConNames, lowBound=0 # no neg weights
                            # , cat='Continuous' # float weights
                            , cat='Integer' # only integer weights allowed
                            )

  if Foot_or_Grid=='Foot': # then do the branching stuff as below:
    # document how this branching is done:
    FIRST = True
    Combo = []
    solutions = []

    for i in tqdm(range(len(UDLs))):
      if FIRST: # adding tableau for UR1 (First)
        cur_UR = UDLs[i]
        # print('UR: ', convert2brandon(cur_UR, 'UR'))

        winner_SR = winners[i]
        # print(f'Trying to make {convert2brandon(winner_SR)} optimal')

        consistent_HRs = gen_HR(winner_SR)
        # print(f'There are {len(consistent_HRs)} HRs consistent with that winner SR')

        # go through the consistent HRs
        for hid in consistent_HRs:
          prob = LpProblem('',LpMinimize)
          prob = add_tableau(Foot_or_Grid, prob, cur_UR, hid, ListOfConFns, DictOfCons, alpha=1)
          # for constr in constraints:
          #   prob += (con_vars[constr]>=0, constr)
          prob += lpSum(DictOfCons)
          if prob.solve()==1:
            Combo.append([hid])
        # print('current number of Combos:', len(Combo))
        if len(Combo)==0:
          # print('Not representable')
          break
        # for branch in Combo:
        #   print(branch)
        FIRST = False

      else:
        # print('moving on to the next tableau')

        cur_UR = UDLs[i]
        # print('adding UR: ', convert2brandon(cur_UR, 'UR'))

        winner_SR = winners[i]
        # print(f'Trying to make {convert2brandon(winner_SR)} jointly optimal w/ the previous winner(s)')

        consistent_HRs = gen_HR(winner_SR)
        # print(f'There are {len(consistent_HRs)} HRs consistent with that winner SR')

        # take each of the stored branch and extend it by trying all the combinations:
        Updated = []
        for branch in Combo:
          for hid in consistent_HRs:
            # print('trying to add', hid)
            prob = LpProblem('',LpMinimize)
            prob = add_tableau(Foot_or_Grid, prob, cur_UR, hid, ListOfConFns, DictOfCons, alpha=1)
            for stored_hr in branch:
              # print('to', stored_hr) stored_hr[0][0] = ur
              prob = add_tableau(Foot_or_Grid, prob, stored_hr[0][0], stored_hr, ListOfConFns, DictOfCons, alpha=1)

            # for constr in constraints:
            #   prob += (con_vars[constr]>=0, constr)
            prob += lpSum(DictOfCons)

            if prob.solve()==1:
              # print("succeeded, storing")
              extended_branch = branch + [hid]
              Updated.append(extended_branch)
              if i == len(UDLs)-1:
                w_vec = []
                for var in DictOfCons.values():
                  w_vec.append(var.value())
                solutions.append(w_vec)
          Combo = Updated
        # print('current number of Combos:', len(Combo))
        if len(Combo)==0:
          # print('Not representable')
          break
        # for branch in Combo:
        #   print(branch)
    print(f"{filename}: number of solutions: {len(Combo)}")
    return solutions

  elif Foot_or_Grid == 'Grid':
    solutions= []
    prob = LpProblem('', LpMinimize)

    for tab in range(len(UDLs)):
      prob = add_tableau(Foot_or_Grid, prob, UDLs[tab], winners[tab], ListOfConFns, DictOfCons, alpha=1)
    prob += lpSum(DictOfCons)

    if prob.solve() == 1:
      w_vec = []
      for var in DictOfCons.values():
        w_vec.append(var.value())
      solutions.append(w_vec)
    else:
      print("no solution :(")
    print(f"{filename}: number of solutions: {len(Combo)}")
    return solutions

# add solver to check one specific parsing

####################################################
# Checking
####################################################

def check_found_weights(cands, w_vec, ListOfConFns):
  viol_vec = -1*evaluate(cands, ListOfConFns)
  winner_id = viol_vec.dot(w_vec).argmax()
  return cands[winner_id]

def check_solution(filename, QI_or_QS, Foot_or_Grid, ListOfConFns, w_vec):
  UDLs = gen_URlist(QI_or_QS)
  observed = read_winners(filename, QI_or_QS)

  CONSISTENT = True
  winners_by_found_weights = []
  if Foot_or_Grid=='Foot':
    for i in range(len(UDLs)):
      cands = full_cands(UDLs[i])
      winner_by_found_weights = check_found_weights(cands, w_vec, ListOfConFns)
      winners_by_found_weights.append(winner_by_found_weights)
      winner_by_found_weights = convert2brandon(winner_by_found_weights[0])

      # print(winner_by_found_weights, observed[i])

      if winner_by_found_weights != observed[i]:
        CONSISTENT = False
        print(f"Wrong winner by found weights for {convert2brandon(UDLs[i], 'UR')}")
        print(f"Observed: {observed[i]}")
        print(f"Found: {winner_by_found_weights}")
        return 'Something Wrong :('
    if CONSISTENT:
      # print("all correct!")
      return [print_foot_pretty(i) for i in winners_by_found_weights]
  elif Foot_or_Grid == 'Grid':
    for i in range(len(UDLs)):
      cands = gen_SR(UDLs[i])
      winner_by_found_weights = check_found_weights(cands, w_vec, ListOfConFns)
      winner_by_found_weights = convert2brandon(winner_by_found_weights)
      winners_by_found_weights.append(winner_by_found_weights)

      # print(winner_by_found_weights, observed[i])

      if winner_by_found_weights != observed[i]:
        CONSISTENT = False
        print(f"Wrong winner by found weights for {convert2brandon(UDLs[i], 'UR')}")
        print(f"Observed: {observed[i]}")
        print(f"Wrong: {winner_by_found_weights}")
        return 'Something Wrong :('
    if CONSISTENT:
      return winners_by_found_weights

def check_learned_weights(filename, QI_or_QS, Foot_or_Grid, ListOfConFns, w_vec, learned_when):
  '''
  given a language and a list of weight vectors, wheter the learning is successful (learned when),
  (and URlist, and CONSTRAINTS)
  for each UR
  if successful,
  then return the top candidates with their probs,
    if the top candidate doesn't have .90 prob,
    then return the second best together
  if not successful,
    then return the observed winner and the top 1 candidate with their probs
  '''
  UDLs = gen_URlist(QI_or_QS)
  observed = read_winners(filename, QI_or_QS)

  result = []
  for i in range(len(UDLs)):
    cur_UR = UDLs[i] # loop through each UR
    if Foot_or_Grid == 'Grid':
      candidates = gen_SR(cur_UR)
      surfaces = [convert2brandon(form) for form in candidates]
    elif Foot_or_Grid == 'Foot':
      candidates = full_cands(cur_UR)
      surfaces = [convert2brandon(form[0]) for form in candidates]

    viol_vec = -1*evaluate(candidates, ListOfConFns)
    harmonies = viol_vec.dot(w_vec)
    eharmonies = np.exp(harmonies)
    Z = sum(eharmonies)
    probs = eharmonies/Z
    probs = np.array(probs)

    # choosing three best
    # ref: https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
    best_ids = np.argpartition(probs, -2)[-2:]
    best_ids = best_ids[np.argsort(probs[best_ids])][::-1]

    winner_ids = []
    cur_winner = observed[i]
    for id, sfc in enumerate(surfaces):
      if sfc==cur_winner:
        winner_ids.append(id)
    winner_ids = np.array(winner_ids)
    # sort winner_ids by probs
    winner_ids = winner_ids[np.argsort(probs[winner_ids])][::-1]

    if learned_when < 0: #not learned
      result.append(
          (learned_when
          , cur_winner # what shouldve been the winner
          , sum(probs[winner_ids]) # how much probs does that winner SR got
          , [candidates[i] for i in best_ids] # what are the two best candidates with current weights
          , [probs[i] for i in best_ids]) # what are the probs of those two best candidates
      )
    elif learned_when > 0: # learned
      result.append(
          (learned_when
          , cur_winner # what shouldve been the winner
          , sum(probs[winner_ids]) # how much probs does that winner SR got
          , [candidates[i] for i in winner_ids[:2]] # what are the two best HRs for that winner SR
          , [probs[i] for i in winner_ids[:2]]) # what are the probs of those two best HRs
      )
  return result

####################################################
# Printing fns
####################################################

def print_foot_pretty(representation):
  sfc, ft=representation
  sfc = convert2brandon(sfc)
  sfc = sfc.split()

  left_bds = [i[0] for i in ft]
  right_bds = [i[-1] for i in ft]

  left_placed=[]
  for id in range(len(sfc)):
    syll = sfc[id]
    if id in left_bds:
      if syll[0]=='[':
        left_placed.append('[('+syll[1:])
      else:
        left_placed.append('('+syll)
    else:
      left_placed.append(syll)

  right_placed=[]
  for id in range(len(left_placed)):
    syll = left_placed[id]
    if id in right_bds:
      if syll[-1]==']':
        right_placed.append(syll[:-1]+')]')
      else:
        right_placed.append(syll+')')
    else:
      right_placed.append(syll)
  return " ".join(right_placed)

def print_result_pretty(filename, QI_or_QS, Foot_or_Grid, ListOfConFns, w_vec, learned_when, con_suffix):
  RES = check_learned_weights(filename, QI_or_QS, Foot_or_Grid, ListOfConFns, w_vec, learned_when)
  con_suffix = '_'+con_suffix
  ListOfConNames = [fn.__name__ for fn in ListOfConFns]

  output_file_name = filename +con_suffix+"_BriefOutput.txt"
  output_file = open(path.join("./", output_file_name), "w")

  w_vec_sorted, CON_names_sorted = (list(t) for t in zip(*sorted(zip(w_vec, ListOfConNames), reverse=True)))
  for i in range(len(CON_names_sorted)):
    line = CON_names_sorted[i] + ': ' + '%.3f'%w_vec_sorted[i]
    output_file.write(line)
    output_file.write('\n')
  output_file.write('\n')
  output_file.write('------------------------------------------')
  output_file.write('\n')

  success = RES[0][0]

  if Foot_or_Grid=='Foot':
    if success>0:
      line = f"learning was successful! language {filename} learned in {success} epoch(s)"
      output_file.write(line)
      output_file.write('\n')
      for tab in RES:
        output_file.write('------------------------------------------')
        output_file.write('\n')
        line = f"observerd form {tab[1]}"+': %.2f'%tab[2]
        output_file.write(line)
        output_file.write('\n')

        output_file.write("for this SR...")
        output_file.write('\n')
        line = f"Best HR {print_foot_pretty(tab[3][0])}"+': %.2f'%tab[4][0]
        output_file.write(line)
        output_file.write('\n')

    else:
      output_file.write("learning was not successful :(")
      output_file.write('\n')
      for tab in RES:
        output_file.write('------------------------------------------')
        output_file.write('\n')
        line = f"observerd form {tab[1]}"+': %.2f'%tab[2]
        output_file.write(line)
        output_file.write('\n')

        line = f"Best HR {print_foot_pretty(tab[3][0])}"+': %.2f'%tab[4][0]
        output_file.write(line)
        output_file.write('\n')

  elif Foot_or_Grid=='Grid':
    if success>0:
      line = f"learning was successful! language {filename} learned in {success} epoch(s)"
      output_file.write(line)
      output_file.write('\n')
      for tab in RES:
        output_file.write('------------------------------------------')
        output_file.write('\n')
        line = f"observerd form {tab[1]}"+': %.2f'%tab[2]
        output_file.write(line)
        output_file.write('\n')
    else:
      output_file.write(f"learning was not successful :(")
      output_file.write('\n')

      for tab in RES:
        output_file.write('------------------------------------------')
        output_file.write('\n')
        line = f"observerd form {tab[1]}"+': %.2f'%tab[2]
        output_file.write(line)
        output_file.write('\n')

        output_file.write("Instead...")
        output_file.write('\n')
        line=f"Best candidate {convert2brandon(tab[3][0])}"+': %.2f'%tab[4][0]
        output_file.write(line)
        output_file.write('\n')
  files.download(output_file_name)
  output_file.close()

def print_solutions_pretty(filename, QI_or_QS, Foot_or_Grid, ListOfConFns, ListOfSolutions, con_suffix):
  ListOfConNames = [fn.__name__ for fn in ListOfConFns]
  UDLs = gen_URlist(QI_or_QS)
  con_suffix = '_'+con_suffix
  output_file_name = filename +con_suffix+"_all_solutions.txt"
  output_file = open(path.join("./", output_file_name), "w")
  output_file.write(f'There are {len(ListOfSolutions)} solution(s) found for {filename}')
  output_file.write('\n')
  for solution in ListOfSolutions:
    output_file.write('------------------------------------------')
    output_file.write('\n')

    w_vec_sorted, CON_names_sorted = (list(t) for t in zip(*sorted(zip(solution, ListOfConNames), reverse=True)))
    for i in range(len(CON_names_sorted)):
      line = CON_names_sorted[i] + ': ' + '%.3f'%w_vec_sorted[i]
      output_file.write(line)
      output_file.write('\n')
    output_file.write('\n')

    found_winners = check_solution(filename, QI_or_QS, Foot_or_Grid, ListOfConFns, solution)
    for winner_candidate in found_winners:
      output_file.write(winner_candidate)
      output_file.write('\n')
  files.download(output_file_name)
  output_file.close()

def foot_tableau(filename, QI_or_QS, cur_udl, ListOfConFns, weights, comparative):
  header = ['Input', 'Output', 'Hidden', 'Target_p', 'H', 'p']
  ListOfConNames = [fn.__name__ for fn in ListOfConFns]
  header += ListOfConNames

  winners = read_winners(filename, QI_or_QS)
  candidates = full_cands(cur_udl)
  cur_udl = convert2brandon(cur_udl,'UR')
  viol_mat = -1*evaluate(candidates, ListOfConFns)
  harmonies = viol_mat.dot(weights)
  eharmonies = np.exp(harmonies)
  Z=sum(eharmonies)
  probs=eharmonies/Z

  TableauRows = []
  id = 0
  for cand in candidates:
    row = []
    row.append(cur_udl) # ur
    sr = convert2brandon(cand[0])
    row.append(sr) # sr
    row.append(print_foot_pretty(cand)) # hr
    if sr in winners:
      row.append(1) # observed
    else:
      row.append(0)
    row.append(harmonies[id])
    row.append(probs[id])
    row.extend(viol_mat[id])
    TableauRows.append(row)
    id+=1
  tableau = pd.DataFrame(TableauRows, columns = header)
  tableau.sort_values(by="p", ascending=False, inplace=True, ignore_index=True)
  tableau = tableau.round(2)
  if not comparative:
    return tableau
  else:
    TableauRows_comparative = []
    optimal_H = tableau.iloc[0]['H']
    optimal_p = tableau.iloc[0]['p']
    id = 0
    FIRST = True
    for r in range(len(tableau)):
      c_row = []
      c_row.append(tableau.iloc[r]['Input']) # ur
      c_row.append(tableau.iloc[r]['Output']) # sr
      c_row.append(tableau.iloc[r]['Hidden']) # hr
      c_row.append(tableau.iloc[r]['Target_p']) # obs
      if FIRST:
        c_row.append(tableau.iloc[r]['H'])
        c_row.append(tableau.iloc[r]['p'])
      else:
        c_row.append(tableau.iloc[r]['H']-optimal_H)
        c_row.append(tableau.iloc[r]['p']-optimal_p)
      v = tableau.iloc[r][ListOfConNames]*weights
      c_row.extend(v)
      TableauRows_comparative.append(c_row)
      FIRST=False
      id+=1
    c_tableau = pd.DataFrame(TableauRows_comparative, columns = header)
    c_tableau.sort_values(by="p", ascending=False, inplace=True, ignore_index=True)
    c_tableau = c_tableau.round(2)
    return c_tableau

def grid_tableau(filename, QI_or_QS, cur_udl, ListOfConFns, weights, comparative):
  header = ['Input', 'Output', 'Target_p', 'H', 'p']
  ListOfConNames = [fn.__name__ for fn in ListOfConFns]
  header += ListOfConNames

  winners = read_winners(filename, QI_or_QS)
  candidates = gen_SR(cur_udl)
  cur_udl = convert2brandon(cur_udl,'UR')
  viol_mat = -1*evaluate(candidates, ListOfConFns)
  harmonies = viol_mat.dot(weights)
  eharmonies = np.exp(harmonies)
  Z=sum(eharmonies)
  probs=eharmonies/Z

  TableauRows = []
  id = 0
  for cand in candidates:
    row = []
    row.append(cur_udl) # ur
    sr = convert2brandon(cand)
    row.append(sr) # sr
    # row.append(print_foot_pretty(cand)) # hr
    if sr in winners:
      row.append(1) # observed
    else:
      row.append(0)
    row.append(harmonies[id])
    row.append(probs[id])
    row.extend(viol_mat[id])
    TableauRows.append(row)
    id+=1
  tableau = pd.DataFrame(TableauRows, columns = header)
  tableau.sort_values(by="p", ascending=False, inplace=True, ignore_index=True)
  tableau = tableau.round(2)
  if not comparative:
    return tableau
  else:
    TableauRows_comparative = []
    optimal_H = tableau.iloc[0]['H']
    optimal_p = tableau.iloc[0]['p']
    id = 0
    FIRST = True
    for r in range(len(tableau)):
      c_row = []
      c_row.append(tableau.iloc[r]['Input']) # ur
      c_row.append(tableau.iloc[r]['Output']) # sr
      c_row.append(tableau.iloc[r]['Target_p']) # obs
      if FIRST:
        c_row.append(tableau.iloc[r]['H'])
        c_row.append(tableau.iloc[r]['p'])
      else:
        c_row.append(tableau.iloc[r]['H']-optimal_H)
        c_row.append(tableau.iloc[r]['p']-optimal_p)
      v = tableau.iloc[r][ListOfConNames]*weights
      c_row.extend(v)
      TableauRows_comparative.append(c_row)
      FIRST=False
      id+=1
    c_tableau = pd.DataFrame(TableauRows_comparative, columns = header)
    c_tableau.sort_values(by="p", ascending=False, inplace=True, ignore_index=True)
    c_tableau = c_tableau.round(2)
    return c_tableau

def print_tableaux_pretty(filename, QI_or_QS, Foot_or_Grid, ListOfConFns, weights, comparative, con_suffix):
  header = ['Input', 'Output', 'Hidden', 'Target_p', 'H', 'p']
  ListOfConNames = [fn.__name__ for fn in ListOfConFns]
  con_suffix = '_'+con_suffix
  header += ListOfConNames
  UDLs = gen_URlist(QI_or_QS)
  if comparative:
    isComparative = '_comparative'
  else:
    isComparative = ''

  FIRST = True
  for ur in UDLs:
    if Foot_or_Grid == 'Foot':
      cur_tab = foot_tableau(filename, QI_or_QS, ur, ListOfConFns, weights, comparative)
    elif Foot_or_Grid == 'Grid':
      cur_tab = grid_tableau(filename, QI_or_QS, ur, ListOfConFns, weights, comparative)

    if FIRST:
      tab = cur_tab
      FIRST = False
    else:
      tab = pd.concat([tab, cur_tab]).reset_index(drop=True)
  if Foot_or_Grid == 'Foot':
    weights = [None]*6+list(weights)
  elif Foot_or_Grid == 'Grid':
    weights = [None]*5+list(weights)
  tab.loc[-1] =  weights
  tab.index = tab.index + 1
  tab.sort_index(inplace=True)
  tab = tab.round(2)

  output_file_name = filename +con_suffix+isComparative+'.csv'
  tab.to_csv(output_file_name, index=False)
  return files.download(output_file_name)
