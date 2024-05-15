import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import iris as iris
import iris.analysis as iris_aa
import glob as glob
import scipy.stats as sst
import matplotlib



def find_nearest(array, value):
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	return idx
 	

	
def load_nao_data(dates):

	nao_index_footprint = []

	data_file = np.genfromtxt('/Volumes/catto/users/mp671/OBS_DATA/NAO/norm.daily.nao.index.b500101.current.ascii.txt',dtype=float,invalid_raise=False,delimiter='')

	years = data_file[:,0].astype(int).astype(str)
	months = np.char.zfill(data_file[:,1].astype(int).astype(str),2)
	days = np.char.zfill(data_file[:,2].astype(int).astype(str),2)

	nao_index = data_file[:,-1]
	
	unique_dates = np.unique(dates)

	nao_dates = []

	for i in range(len(years)):
	

		nao_dates.append(''+years[i]+''+months[i]+''+days[i]+'')
		
	for date in dates:
	

		date_index = np.where(np.array(nao_dates)==date[0:8])[0]
		
		nao_value = nao_index[date_index[0]]
		
		nao_index_footprint.append(nao_value)
		
		
	return nao_index_footprint
	

def load_fit_data(dataset,year,rp,nao_predict,lat_idx,lon_idx):

	## Rate of footprints
	lamda = len(year)/len(np.unique(year))
	num_sigma = int(len(year)*(1-0.7))
	
	if dataset == 'wisc':
		start_year='1950'
		end_year='2015'
	if dataset == 'wwieur':
		start_year='1979'
		end_year='2021'

	## Load in model parameters
	beta0 = np.genfromtxt('/Volumes/catto/users/mp671/python_code/willis/wtw_ajgre/WINDSTORM_MODEL_NEW/data/parameters/'+dataset+'/beta/beta0_q0.7_'+start_year+'_'+end_year+'.txt',delimiter=',')[lat_idx,lon_idx]
	beta1 = np.genfromtxt('/Volumes/catto/users/mp671/python_code/willis/wtw_ajgre/WINDSTORM_MODEL_NEW/data/parameters/'+dataset+'/beta/beta1_q0.7_'+start_year+'_'+end_year+'.txt',delimiter=',')[lat_idx,lon_idx]
	u = beta0+beta1*nao_predict
	sigma = np.loadtxt('/Volumes/catto/users/mp671/python_code/willis/wtw_ajgre/WINDSTORM_MODEL_NEW/data/parameters/'+dataset+'/sigma/sigma_mean_excess_q0.7_'+start_year+'_'+end_year+'.txt',delimiter=',')[lat_idx,lon_idx]	
		
		
	all_rp = []
	all_lower_bound = []
	all_upper_bound = []	
	for rp in rp_array:
		c_t = np.log(1-0.7)+np.log(rp)+np.log(lamda)

		## Calculate predicted gusts based on return period, sigma, and predictions
		yp = u+sigma*c_t
	
		gamma_interval = sst.gamma.interval(0.95,num_sigma,scale=sigma/num_sigma)	
		yp_lower = u+(gamma_interval[0]*c_t)
		yp_upper = u+(gamma_interval[1]*c_t)
		
		all_rp.append(yp)
		all_lower_bound.append(yp_lower)
		all_upper_bound.append(yp_upper)
	return all_rp,all_lower_bound,all_upper_bound
	
print('Would you like to calcualte return levels using WISC (1950-2015) or WWIEUR (1979-2021)?')
data_input_choice = input('0 for WISC ... 1 for WWIEUR ... ')
if data_input_choice == '0':
	dataset='wisc'
if data_input_choice == '1':
	dataset='wwieur'
print(' ')
print('Calculations will be made using',dataset.upper(),'as input...')
print('')

## Generate footprint dates and get lat/lon information of WWIEUR data
footprint_dates = np.loadtxt('/Volumes/catto/users/mp671/python_code/willis/wtw_ajgre/WINDSTORM_MODEL_NEW/data/footprint_dates/footprint_dates_'+dataset+'.txt',dtype='str')
footprint_years = [i[0:4] for i in footprint_dates]

lons = np.loadtxt('/Volumes/catto/users/mp671/python_code/willis/wtw_ajgre/WINDSTORM_MODEL_NEW/data/lat_lon/'+dataset+'/'+dataset+'_lons.txt',dtype='float')
lats = np.loadtxt('/Volumes/catto/users/mp671/python_code/willis/wtw_ajgre/WINDSTORM_MODEL_NEW/data/lat_lon/'+dataset+'/'+dataset+'_lats.txt',dtype='float')

#### Specify latitude input 
lat_input = float(input('Specify latitude of location ... Range is '+str(round(np.nanmin(lats),1))+' - '+str(round(np.nanmax(lats),1))+'... '))
if lat_input >np.nanmin(lats) and lat_input<np.nanmax(lats):
	valid_lat=True
else:
	valid_lat=False
while valid_lat==False:
	if lat_input<np.nanmin(lats) or lat_input>np.nanmax(lats):
		print('Latitude input outside range of data. Range is '+str(round(np.nanmin(lats),1))+' - '+str(round(np.nanmax(lats),1))+'')
		lat_input = float(input('Enter new latitude input ... '))	
		
		if lat_input >np.nanmin(lats) and lat_input<np.nanmax(lats):
			valid_lat=True
		else:
			print('Latitude input still invalid ... enter again')
			print('')
print('Valid latitude input')
print('')

#### Specify longitude input
lon_input = float(input('Specify longitude of location ... Range is '+str(round(np.nanmin(lons),1))+' - '+str(round(np.nanmax(lons),1))+'... '))
if lon_input >np.nanmin(lons) and lon_input<np.nanmax(lons):
	valid_lon=True
else:
	valid_lon=False
while valid_lon==False:
	if lon_input<np.nanmin(lons) or lon_input>np.nanmax(lons):
		print('Longitude input outside range of data. Range is '+str(round(np.nanmin(lons),1))+' - '+str(round(np.nanmax(lons),1))+'')
		lon_input = float(input('Enter new longitude input ... '))	
		
		if lon_input >np.nanmin(lons) and lon_input<np.nanmax(lons):
			valid_lon=True
		else:
			print('Longitude input still invalid ... enter again')
			print('')
print('Valid longitude input')

print(' ')
print('Location to be plotted is',lat_input,'N,',lon_input,'E')

#### Specify location name
loc_name = input('Specify location name ... ')

## Find out where in data array the lat/lon point sits
lat_idx = find_nearest(lats,float(lat_input))
lon_idx = find_nearest(lons,float(lon_input))


## Load in NAO data
footprint_nao_values = load_nao_data(footprint_dates)
print('NAO data loaded')


### Specify NAO value to use in analysis
nao_choice = input('0 to use period mean NAO ... 1 for custom NAO ... ')
if nao_choice == '0':
	nao_value = np.nanmean(footprint_nao_values)
if nao_choice == '1':
	nao_value = float(input('Enter custom NAO state (-2 to 2) .... '))
print('NAO state is ...',str(nao_value))
print('')	

## Generate array of RP to estimate RL for
rp_array = np.linspace(2,200,199)
rp_array=np.hstack((rp_array,500))

print('Loading data and parameters for calculations')
## Calculate return levels
return_level_nao,return_level_nao_lower_bound,return_level_nao_upper_bound = load_fit_data(dataset,footprint_years,rp_array,nao_value,lat_idx,lon_idx)
print('Data loaded ... plotting.')


#######
#######
# PLOTTING
#######
#######

fig1, ax1 = plt.subplots()

plt.ion()
plt.semilogx(rp_array,return_level_nao,'k',label='Return level')
plt.semilogx(rp_array,return_level_nao_lower_bound,'k',linestyle='--',label='95% CI')
plt.semilogx(rp_array,return_level_nao_upper_bound,'k',linestyle='--')
plt.axhline(return_level_nao[np.where(rp_array==200)[0][0]],color='r',label='200yr')
plt.axhline(return_level_nao[np.where(rp_array==10)[0][0]],color='darkorange',label='10yr')
plt.legend(loc='upper left')
plt.title(''+str(lat_input)+'N,'+str(lon_input)+'E ('+loc_name+') - NAO='+str(round(nao_value,2))+' ('+dataset.upper()+')',fontsize=16)
ax1.set_xticks([2,10,20,50,100, 200,500])
plt.xlim(1,500)
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

plt.xlabel('Return Period (Years)',fontsize=14)
plt.ylabel('Return Level (m s$^{-1}$)',fontsize=14)

core_rp = [2,5,10,20,50,100,200]

for rp in core_rp:
	idx = np.where(rp_array==rp)[0][0]
	print(rp,'yr estimate ...',round(return_level_nao[idx],2),'- 95% CI ...',round(return_level_nao_lower_bound[idx],2),'-',round(return_level_nao_upper_bound[idx],2))
print('All estimates for NAO =',round(nao_value,2))


