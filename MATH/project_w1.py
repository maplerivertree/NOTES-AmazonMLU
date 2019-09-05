### Importing the Data ###

# Load training data
eider.s3.download('INTERNAL_DATABASE')
data = pd.read_csv('tmp/Book_Ratings.csv', dtype = {'User': str, 'ASIN': str, 'Rating': np.int})

print("Sample Data")
print("-----------")
print(data.sample(20))

# Count number of unique users and number of unique ASINs in our dataset
uniqueUsers = data['User'].unique().tolist()
uniqueASINs = data['ASIN'].unique().tolist()
reviews = data['Rating'].tolist()
numUser = len(uniqueUsers)
numASIN = len(uniqueASINs)
numR = len(reviews)

# Split to train and test
train, test = train_test_split(data, random_state = 8675309, stratify = data['ASIN'])
num_train = train.shape[0]
num_test = test.shape[0]

print("Number of unique Users: {}".format(numUser))
print("Number of unique ASINs: {}".format(numASIN))
print("Number of reviews: {}".format(numR))



#print(b)



#PROJECT QUESTION 1
#'Counter' turn each entry's count into a disctionary format
from collections import Counter 
dic_num_review_byUser = Counter(train['User'])
dic_num_review_byASIN = Counter(train['ASIN'])


#------------- Histogram
#turn the number of reviews by each distinct users into a list for plotting

list_num_review_byUser = list(dic_num_review_byUser.values())
#plot histogram
plt.figure(1), plt.title('histogram: number of reviews a user gives'), plt.ylabel('number of users'), plt.xlabel('number of reviews')
n, bins, patches = plt.hist(x = list_num_review_byUser, bins = "auto")


#-------------- Scatter Graph
#sort the dictionary (via counter) based on 'ASIN'
#sort the dataframe (via groupby) based on 'ASIN'. 
#turn the values in the dictionary into a list
#add this list as a new column in the dataframe
#use the two columns in the dataframe to create the scatter graph
reviews_received=[]
sorted_dic_num_review_byASIN = sorted(dic_num_review_byASIN.items(), key = lambda item:item[0])
for k, v in sorted_dic_num_review_byASIN:
    reviews_received.append(v)
# take averge in data frame
#https://stackoverflow.com/questions/30328646/python-pandas-group-by-in-group-by-and-average
df_groupby = train.groupby(['ASIN']).mean().reset_index()
sorted_df_groupby = df_groupby.sort_values(by = ['ASIN'])

#add a column in the dataframe using a list
sorted_df_groupby['Reviews_Received'] = reviews_received
print(k[0:5])
plt.figure(2), plt.title('average rating vs. the number of reviews received'), plt.ylabel('number of review'), plt.xlabel('average rating')
plt.scatter(sorted_df_groupby.Rating,sorted_df_groupby.Reviews_Received)



from sklearn.preprocessing import Binarizer
 
binarizer = Binarizer().fit(X)

S_train = train.pivot_table(index='User', columns='ASIN', values='Rating', fill_value=0).as_matrix()
R_train = binarizer.transform(R_train)
S_test = test.pivot_table(index='User', columns='ASIN', values='Rating', fill_value=0).as_matrix()
R_test = binarizer.transform(R_test)

print(S_train.shape)
print(S_test.shape)
