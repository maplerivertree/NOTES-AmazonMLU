### Project Answer 4 ###
#CALLOUTS:
#use to enable a dot calculation of n x 1 and 1 x m, you np.matrix to turn both vectors into matrics to allow the right shape
#Hadamard product is np.multiply(a, b)

A = np.matrix(np.ones(numUser), dtype='float').T   # numUser is already only taking from Training set
F = train.groupby(['ASIN']).mean().reset_index()
#print(F)
F = np.matrix(df_groupby.values[:, -1], dtype='float')#.values is used to convert dataframe into a np array
P = A.dot(F)
#print(F)
print("The shape of A is " + str(A.shape))
print("The shape of F is " + str(F.shape))
print("The shape of P is " + str(P.shape))
print("The number of Users in the training set is " + str(numUser))
print("The shape of S_test is " + str(S_test.shape))

b= np.linalg.norm(np.multiply(P, R_test) - S_test, 2)
print(round(b,1))
