import numpy as np
import pandas as pd
data = pd.read_csv('titanic.csv', index_col='PassengerId')
# print data.head()
nMales = np.nonzero( data['Sex'] == 'male')[0].shape[0]
nFemales = np.nonzero( data['Sex'] == 'female')[0].shape[0]

nPassengers = len(data)
nSurvived = np.sum(data['Survived'])
nFirstClass = np.nonzero(data['Pclass'] == 1)[0].shape[0]
FirstAllRatio = round(float(nFirstClass) / nPassengers * 100., 2)
SurvivedRatio = round(float(nSurvived) / nPassengers * 100., 2)
meanAge = round(np.nanmean(data['Age']), 2)
medianAge = np.nanmedian(data['Age'])

fname1 = '1.txt'
fname2 = '2.txt'
fname3 = '3.txt'
fname4 = '4.txt'
fname5 = '5.txt'
fname6 = '6.txt'

f1 = open(fname1, 'w')
# print nMales 
f1.write(str(nMales) + ' ' + str(nFemales))
f1.close()

f2 = open(fname2, 'w')
f2.write(str(SurvivedRatio))
f2.close()


f3 = open(fname3, 'w')


f3.write(str(FirstAllRatio))
f3.close()

f4 = open(fname4, 'w')
f4.write(str(meanAge) + ' ' + str(medianAge))
f4.close()

nSibs = data['SibSp']
nParch = data['Parch']

X = np.vstack([nSibs, nParch])

corr = round(np.corrcoef(X)[0,1],2)
f5 = open(fname5, 'w')
f5.write(str(corr))
print corr

import re

ifFem = data.Sex == 'female'
femNames = data[ifFem]['Name']
count_Misses = 0
count_total = 0
countCheck = 0

firstNames = []
for fullName in femNames:
  # print fullName
  count_total += 1
  matchMrs = re.search(r'(Mrs|Miss|Mme|Ms|Dr|the Countess|Mlle)([\.\s]*)(.*)', fullName)
  if matchMrs:
    count_Misses += 1
  else:
    print fullName
  namestr = matchMrs.group(3) 
  matchNameBrackets = re.search(r'\(\s*(\w+)',namestr)
  if matchNameBrackets:
    name = matchNameBrackets.group(1)
  else:
    matchNoBrackets = re.search(r'\s*(\w+)',namestr)
    if matchNoBrackets:
      name = matchNoBrackets.group(1)
  firstNames.append(name)
if count_total == count_Misses:
  print 'Everyone is a Missis'
else:
  print '{0:2d} of {1:3d} Ladies are Misses'.format(count_Misses, count_total)
print firstNames

nameFreqs = {}
for name in firstNames:
  if name in nameFreqs:
    nameFreqs[name] +=1
  else:
    nameFreqs[name] = 1
sortNames = sorted(nameFreqs, key=nameFreqs.get, reverse=True)
for name in sortNames:
  print name + ': ' + str(nameFreqs[name]) 
# print len(nameFreqs)
f6 = open(fname6,'w')
f6.write(sortNames[0])
f6.close()