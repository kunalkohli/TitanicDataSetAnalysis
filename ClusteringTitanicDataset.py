import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random 				#to generate initial random position of cluster centroid

f = open('titanic.json','r') #read json from the file
d = json.load(f) #read from file is typically what you want

#Passenger details array is not used in the code since we are only using the useful inputs hence MeaningfulInputvector inputs.
#If we want to plot all the inputs ie fare,age,gender,passengercount and embarked location,we will use this vector.
passengerDetails = np.zeros((len(d),5))

#MeaningfulInputVector basically contians those inputs which are considered meaningful ie age passenger count and gender and the target
MeaningfulInputVector = np.zeros((len(d),4))

Target = np.zeros(len(d))

medianAge = 0.0							 #medianOfAge initialised to 0


#Calculate the median values for missing values of age
def calculateMedianOfAge():
	ageList = list()						 #create a list to store all the age values to find medain for imputation
	for i in range(0,len(d)):
		if (d[i]['Age']) != '':
			ageList.append(d[i]['Age'])

	ageList = np.array(ageList).astype(np.float)	#convert array of ages form unsigned type to float 
	#print ageList.dtype							#verify that the array type is float
	medianAge =np.median(ageList)					#find median of the age array using numpy inbuilt median function


####################################################
'''
#code to test if location has a missing value.
for i in range(0,len(d)):
	if (d[i]['Embarked']) == '':
		print 'Missing Location for :',d[i]['PassengerName']

#code to test if age has a missing value.
for i in range(0,len(d)):
	if (d[i]['Age']) == '':
		print 'Missing Age for :',d[i]['PassengerName']
'''
####################################################

def populateDataAndTarget():

	calculateMedianOfAge()

	for i in range(0,len(d)):

		passengerDetails[i][0] = d[i]['Fare']
	
		#handling missing values while creating list and populating them with average age value
		if d[i]['Age'] != '':
			passengerDetails[i][1] = d[i]['Age']
		else:
			passengerDetails[i][1] = medianAge

		#store age in meaningful vector inputs
		MeaningfulInputVector[i][0] = passengerDetails[i][1]

		passengerDetails[i][2] = d[i]['ParentsAndChildren'] + d[i]['SiblingsAndSpouses']
		
		#store passenger count in meaningful vector inputs
		MeaningfulInputVector[i][1] = passengerDetails[i][2]

		#encoding the string variables in this part of the code	
		if d[i]['Embarked'] == 'C':
			passengerDetails[i][3] = 1
		elif d[i]['Embarked'] == 'Q':
			passengerDetails[i][3] = 2
		elif d[i]['Embarked'] == 'S': 	
			passengerDetails[i][3] = 3
		elif d[i]['Embarked'] == '':	
			passengerDetails[i][3] = 2		#replacing missing values of embarked location to 'Q' ie 2

		#encoding gender in this code
		if d[i]['Sex'] == 'male':
			passengerDetails[i][4] = 0
		elif d[i]['Sex'] == 'female':
			passengerDetails[i][4] = 1
		
		#store genderin meaningful vector inputs
		MeaningfulInputVector[i][2] = passengerDetails[i][4]
		
		Target[i] = d[i]['Survived'] 		#populating target list. 0 = died, 1 = survived	
		MeaningfulInputVector[i][3] = Target[i]	#Adding target to the list so as to check if gender played an important part in survival
	

def normaliseInputs():
	numberOfColumns = len(MeaningfulInputVector[1] -1 ) #since target is not normalised
	for j in range(0,numberOfColumns):
		minValue = min(MeaningfulInputVector[:,j]) 		#find minimum element in jth column
		maxValue = max(MeaningfulInputVector[:,j])		#find maximum element in jth column

		for i in range(0,len(MeaningfulInputVector)): 	#for each row element in jth column ie loops runs for number of rows in data array
			MeaningfulInputVector[i,j] = (MeaningfulInputVector[i,j] - minValue)/float(maxValue - minValue)


#perform hierarchial clustering
def plotDendogram():
	#Only use inputs for dendogram hence we subset the array ie [0:3] ie first three columns.Since 3rd column has target
	Z = linkage(MeaningfulInputVector[:,0:3], method='ward', metric='euclidean') # distance between clusters and metric
	plt.title('Hierarchical Clustering Dendrogram')
	plt.xlabel('Titanic Data Set')
	plt.ylabel('distance')
	plt.axhline(y=4.5)		#creates a horizintal line depicting the cluster threshold i.e. line where to cut dendogram

	#creates a dendrogram hierarchial plot
	dendrogram(
		Z,
		leaf_rotation=90.,  # rotates the x axis labels
		leaf_font_size=8.  # font size for the x axis labels
	)
	#display both figures
	plt.show()


def performKmeans():

	#create 3 initial random cluster centroid since I have selected 3 clusters.since we have 3 inputs hence each centroid will have 3 demensions ie 9 times random function [[cluster 1 ->3 points],[cluster 2 ->3 points],[cluster 3 ->3 points]

	#clusterCentroid gives the centroid of each feature for each of the cluster
	#clusterCentroid[0][0] - Age centroid for first cluster
	#clusterCentroid[0][1] - Siblings/Spouse count centroid for first cluster
	#clusterCentroid[0][2] - Gender centroid for first cluster
	
	ClusterCentroid = [[random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)],[random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)],[random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)]]
	
	
	#initialise distance list to 3 float values.ie distances from each cluster centroid
	Dist = [0.0,0.0,0.0]
	index = 0
	while(index < 10):
		#compute distance of each point from each centroid and then assign to the corresponding cluster from whose centroid dist is minimum
		
		#create 1 list of lists.initialise each list to 0. these internal lists will store points corresponding to a cluster
		cluster = [[0],[0],[0]]
		for i in range(0,len(MeaningfulInputVector)):
		
			computeDistance(MeaningfulInputVector[i],ClusterCentroid,Dist)
			ClusterToAssign = FindClusterIndex(Dist)
		
			#append the point to the cluster found earlier.this cluster number(ClusterToAssign) is used to access the list
			cluster[ClusterToAssign].append(MeaningfulInputVector[i])
	
		#since we initialised the cluster list to 0 initially,so each inside list will have 0 as the first element which has to be popped
		cluster[0].pop(0)
		cluster[1].pop(0)
		cluster[2].pop(0)
	
		#now plot clusters.Total 10 plots will be plotted with a centroid collection at each iteration
		plotCluster(cluster,ClusterCentroid)

		#Compute centroid location. i.e. new location after adding the points
		computeClusterCentroid(ClusterCentroid,cluster)

		index+=1		#increment the index for each iteration
		

	#Plot clusters based on the target values.
	plotClustersBasedOnTarget(cluster,ClusterCentroid)
	

def computeDistance(InputList,ClusterCentroid,Dist):
	#distance formula ie sqrt((x1-c1)^2). using it to calculate distance of input from centroid.since input has 3 useful features ie age,sibling and gender,hence 3 calculations

	#Syntatic meaning is not symantic meaning. Hence we only consider points for age,gender and Number of sibling count as the import feature while computing centroids
	#Dist[0] is distance of point from first cluster centroid
	#Dist[1] is distance of point from second cluster centroid
	#Dist[2] is distance of point from third cluster centroid

	Dist[0] = ((InputList[0] - ClusterCentroid[0][0]) **2) + ((InputList[1] - ClusterCentroid[0][1]) **2) + ((InputList[2] - ClusterCentroid[0][2]) **2) 

	Dist[1] = ((InputList[0] - ClusterCentroid[1][0]) **2) + ((InputList[1] - ClusterCentroid[1][1]) **2) + ((InputList[2] - ClusterCentroid[1][2]) **2)

	Dist[2] = ((InputList[0] - ClusterCentroid[2][0]) **2) + ((InputList[1] - ClusterCentroid[2][1]) **2) + ((InputList[2] - ClusterCentroid[2][2]) **2)


def FindClusterIndex(Dist):
	minDistance = min(Dist)
	clusterIndex = 0

	for i in range(0,len(Dist)):
		if minDistance == Dist[i]:
			return i	

def computeClusterCentroid(ClusterCentroid,cluster):

	ElementsInCluster = 0

	#iterate for 3 times since we have 3 clusters.Calculate centroid everytime

	for i in range(0,len(cluster)):
		ElementsInCluster = len(cluster[i])
		SumOfFeatureInTheCluster = sum(cluster[i])

		#Check if cluster is not empty to prevent divide by zero in next step for average calculation		
		if (ElementsInCluster != 0):
			ClusterCentroid[i] = SumOfFeatureInTheCluster/float(ElementsInCluster)
	
def plotCluster(cluster,ClusterCentroid):
	colorIndex = ['b','g','m']
	ax = Axes3D(plt.gcf())
	for i in range(0,len(cluster)):
		x_point = []
		y_point = []    #These store the x,y and z axises which have to be plotted for a particular point.Initialised for each cluster
		z_point = []
		for j in range(0,len(cluster[i])):
			
			x_point.append(cluster[i][j][0])
			y_point.append(cluster[i][j][1])
			z_point.append(cluster[i][j][2])

		if len(cluster[i]): 
			ax.scatter(x_point, y_point, z_point,c = colorIndex[i])
		ax.scatter(ClusterCentroid[i][0],ClusterCentroid[i][1],ClusterCentroid[i][2],marker='x',s=100,color = colorIndex[i])

	plt.title("Clustering of input vectors")
	ax.set_xlabel('X intercept - Age')
	ax.set_ylabel('Y intercept - Number of Members')
	ax.set_zlabel('Z intercept - Gender')
	plt.show()
		

def plotClustersBasedOnTarget(cluster,ClusterCentroid):
	#1.This function takes the clusters generated after performing k means and then plot the points(Inputs) based on the target value
	#2. If the input has target value as survived,it checks if input is for gender = female or male
	#3. if input is female then it adds it to survived women list else survived men list
	#4.	If the input has target value as dead,it checks if input is for gender = female or male
	#5. if input is female then it adds it to dead women list else dead men list
	#6. Next we plot the curves and color code the points based on gender and survival
	#7. women_survived - green,men_survived - blue
	#8. women_dead - red, men_dead - black
	#9 based on the color coding we can identify which gender was given more preference in the survival operations
	ax = Axes3D(plt.gcf())
	dead_women_x = []
	dead_women_y = []
	dead_women_z = []
	survived_women_x = []
	survived_women_y = []		#Store different lists for women and men based on their survival
	survived_women_z = []
	dead_men_x = []
	dead_men_y = []
	dead_men_z = []
	survived_men_x = []
	survived_men_y = []
	survived_men_z = []

	for i in range(0,len(cluster)):		

		for j in range(0,len(cluster[i])):
			if (int(cluster[i][j][3])) == 1:			#if target is 1 i.e. survived

				if(int(cluster[i][j][2]) == 1):		#if input value is female
					survived_women_x.append(cluster[i][j][0])
					survived_women_y.append(cluster[i][j][1])
					survived_women_z.append(cluster[i][j][2])
				else:							#if input is for male
					survived_men_x.append(cluster[i][j][0])
					survived_men_y.append(cluster[i][j][1])
					survived_men_z.append(cluster[i][j][2])

			else:								#if target is 0 i.e. dead
				if(int(cluster[i][j][2]) == 1):

					dead_women_x.append(cluster[i][j][0])
					dead_women_y.append(cluster[i][j][1])
					dead_women_z.append(cluster[i][j][2])

				else:

					dead_men_x.append(cluster[i][j][0])
					dead_men_y.append(cluster[i][j][1])
					dead_men_z.append(cluster[i][j][2])
		
		
		ax.scatter(survived_women_x, survived_women_y, survived_women_z,c = 'green') #Green for survived women
		ax.scatter(survived_men_x, survived_men_y, survived_men_z,c = 'blue')		 # Blue for survived men

		ax.scatter(dead_women_x, dead_women_y, dead_women_z,c = 'red') 				 #red for dead women
		ax.scatter(dead_men_x, dead_men_y, dead_men_z,c = 'black')		 			 #black for dead men
			
	plt.title('Gender bias in survival operations')
	plt.figtext(0.80, 0.09, 'Green = women survived',color = 'green')
	plt.figtext(0.80, 0.07, 'blue = men survived',color = 'blue')
	plt.figtext(0.80, 0.05, 'red = women dead',color = 'red')
	plt.figtext(0.80, 0.03, 'blue = men dead',color = 'black')
	ax.set_xlabel('X intercept - Age')
	ax.set_ylabel('Y intercept - Number of Members')
	ax.set_zlabel('Z intercept - Gender')
	plt.show()	

	print 'Number of men rescued : ', len(survived_men_x)
	print 'Number of men Dead : ', len(dead_men_x)
	print 'Number of women rescued : ', len(survived_women_x)
	print 'Number of women dead : ', len(dead_women_x)
	

#Code below is basically used to check if clusters have changed after each iteration.Not used now since distances never converge
'''
def CheckIfAnyChangeInCentroid(ClusterCentroidOld,ClusterCentroid):
	cluster1Same = False
	cluster2Same = False
	cluster3Same = False
	cluster4Same = False
	cluster5Same = False

	if(ClusterCentroid[0][1] == ClusterCentroidOld[0][1] and ClusterCentroid[0][2] == ClusterCentroidOld[0][2] and ClusterCentroid[0][4] == ClusterCentroidOld[0][4]):
		cluster1Same = True

	if (ClusterCentroid[1][1] == ClusterCentroidOld[1][1] and ClusterCentroid[1][2] == ClusterCentroidOld[1][2] and ClusterCentroid[1][4] == ClusterCentroidOld[1][4]):
		cluster2Same = True

	if (ClusterCentroid[2][1] == ClusterCentroidOld[2][1] and ClusterCentroid[2][2] == ClusterCentroidOld[2][2] and ClusterCentroid[2][4] == ClusterCentroidOld[2][4]):
		cluster3Same = True

	if(ClusterCentroid[3][1] == ClusterCentroidOld[3][1] and ClusterCentroid[3][2] == ClusterCentroidOld[3][2] and ClusterCentroid[3][4] == ClusterCentroidOld[3][4]):
		cluster3Same = True

	if (ClusterCentroid[4][1] == ClusterCentroidOld[4][1] and ClusterCentroid[4][2] == ClusterCentroidOld[4][2] and ClusterCentroid[4][4] == ClusterCentroidOld[4][4]):
		cluster4Same = True
	
	if(cluster1Same and cluster2Same and cluster3Same and cluster4Same and cluster5Same):
		print 'yes'		
		return True
	else:
		print 'no'
		return False
'''	
 
###################################################################
#below this comment, all the functions are called

#Step1: populate data into arrays
#step2: normalise the inputs
#step3: plot dendogram using inbuilt library.select a cutoff
#step4: do k means analysis
#	step4a:	initiate a cluster with 2 in it...Each array will store the point assigned to that cluster
#	step4b: initiate clustercentroid list which contains centroids generated randomly
#	step4c: calculate distance of each point in the dataset and assign it to corresponding cluster
#	step4d: plot cluster
#	step4e: recalculate cluster centroid
#step5: Plot the cluster and calculate the centroid's new locations after adding the points to the cluster.
#step6: calcualte distances of points from the new centroids and change the assignment based on distance
#step8: repeat this for 10 times and then plot for every iteration
#step9: after we have the cluster. check if the survival operation was more biased towards a specific gender
#step10: color code the points in the cluster based on gender and their survival to get a better idea. 

#####################################################################################################

#CALLING ALL THE FUNCTIONS BELOW THIS IN THE ORDER

#call populate function for populating dataset and Target
populateDataAndTarget()

#normalise inputs to have a range 0 to 1 in the input vectors
normaliseInputs()

#plot dendogram
plotDendogram()

#perfrom k means clustering
performKmeans()




























