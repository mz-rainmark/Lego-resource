

# 3, 0, 3, '151, 122, 99', '10', 1, 0, 3

# newFile = open('log/myprocess.txt', 'w')
#
# with open('log/final_data.txt','r') as f:
#     while True:
#         line = f.readline()
#         if not line:
#             break
#         else:
#             l = line.find('(')
#             r = line.find(')')
#             newline = line[:l].replace(',', '\t') + line[l:r+1] + line[r+1:].replace(',', '\t')
#             newFile.write(newline)
#             # newFile.write('\n')
#
# newFile.close()

import numpy as np
import pandas as pd


txt = np.loadtxt('log/myprocess.txt')
txtDF = pd.DataFrame(txt)
txtDF.to_csv('log/file1.csv', index=False)

print(txtDF)
