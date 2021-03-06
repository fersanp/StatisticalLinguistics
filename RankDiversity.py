# Ultima version 26 Noviembre 2017

import numpy as np
from os import chdir, getcwd, listdir

################################################################################
# Esta parte es para la diversidad de rango. Inicio
################################################################################

def SearchRank(word,file_name):
    '''
    Devuelve el rango que tiene un objeto dado. Si no esta en file_name devuelve NaN
    SearchRank(objeto a buscar(str), archivo en donde buscar (str))
    '''
    rank_desire = np.NaN
    with open(file_name,'r') as file_open: 
        rank = 1
        for row in file_open:
            split_row = row.split()
            if split_row[0] == word:
                rank_desire = rank
                break
            rank += 1   
    return rank_desire

def find_traject(word,time_list):
    N = time_list.size
    trayectoria = np.empty(N)
    for i in xrange(N):
        trayectoria[i] = SearchRank(word,time_list[i])
    return trayectoria 

from scipy.stats import norm,cauchy
from scipy.optimize import curve_fit

# Ajuste con la distribucion normal, trabaja bien 
def funcion_erf(X,mu,sigma):
    return norm.cdf(X,loc=mu,scale=sigma)


def Data4RD(init_year,num_years,years_list,num_words,normalize=True):
    '''
    Regresa los datos de la diversidad de rango para un periodo de tiempo dado.

    Data4RD(init_year,num_years,years_list,num_words,normalize=True,dtype='u8')
    '''
        
    if init_year+num_years > years_list.size:
        num_years = years_list.size-init_year

    # Main program
    matrix_words = np.empty((num_years,num_words),dtype='a70')
    matrix_words[:,:] = ' '
    FilesSize = np.empty(num_years,dtype='u4')       

    # Matriz de words vs. years
    i = 0
    for year in years_list[init_year:init_year+num_years]:
        year_file = np.loadtxt(year,dtype='a50',usecols=(0,))

        FilesSize[i] = year_file.size   
        # Columnas son words y renglones son las years
        j = 0
        for player in year_file[:num_words]:
            matrix_words[i,j] = player
            j += 1
        i += 1

    # For normalize Data
    FilesSize.sort()
    MaxRank = FilesSize[-1]
    MinRank = FilesSize[0]


    # get the unique entries along the ranks
    different_words_per_rank = np.zeros((num_words,2),dtype='f4')
    different_words_per_rank[:,0] = np.arange(1,num_words+1)
    for j in xrange(num_words):
        different_words_per_rank[j][1] = np.unique(matrix_words[:,j]).size
    if MinRank < num_words:    
        different_words_per_rank[:,1][MinRank:] -= 1
    diversity = different_words_per_rank


    # Normalize data
    if normalize:
        if num_words <= MinRank:
            diversity[:,1] =  diversity[:,1]/np.float(num_years)   
        else:
            Array2Normalize = np.zeros(num_words)
            viejoMR = 0
            for i in xrange(num_years):
                nuevoMR = FilesSize[i]
                Array2Normalize[viejoMR:nuevoMR] = FilesSize[FilesSize>=nuevoMR].size
                viejoMR = nuevoMR
            if 0 in Array2Normalize:
                diversity = diversity[np.logical_not(Array2Normalize==0)]
            diversity[:,1] = diversity[:,1]/Array2Normalize[np.logical_not(Array2Normalize==0)]

    if MaxRank < num_words:
        num_words = MaxRank
        diversity = diversity[:num_words]

    # print 'NW %d\tNY %d\tIY %d' % (num_words,num_years,init_year)
    return diversity

def GetMeanData(diversity,size_bins=0.1,normal_scale=False):
    '''
    Regresa el promedio de datos.
    GetMeanData(diversity,bins size)
    '''
    if len(np.shape(diversity)) == 2:
        num_words = diversity.size/2
        diversity = diversity[:,1]
    else:
        num_words = np.size(diversity)

    # promedio de los datos

    nw_log_10 = np.log10(num_words)
    rank_logspace = np.log10(np.arange(1,num_words+1))
    limit = np.int(np.round((nw_log_10/size_bins)))
    bins = np.linspace(0,nw_log_10,num=limit+1)
    diversity_mean  = np.zeros((limit,2))
    diversity_mean[:,0] = bins[1:]
    k = 0

    nodata_between_points = []
    while k < limit:
        index = np.logical_and(rank_logspace>=bins[k],rank_logspace<=bins[k+1])
        if index.any():
            # diversity_mean[k][1] = np.mean(diversity[:,1][index]) 
            diversity_mean[k][1] = np.mean(diversity[index]) 
        else:
            nodata_between_points.append(k)
        k += 1    

    if len(nodata_between_points) != 0:
        for i in nodata_between_points:
            diversity_mean[i][1] = (diversity_mean[i+1][1] + diversity_mean[i-1][1])/2.

    # diversity_mean[1:3,1] = diversity[:2,1]
    diversity_mean[1:3,1] = diversity[:2]
    diversity_mean[1:3,0] = rank_logspace[:2]

    rd_mean = diversity_mean[1:]
    if normal_scale:
        rd_mean[:,0] = np.power(10.,rd_mean[:,0])
    return rd_mean


def GetExtrData(diversity,size_bins=0.1,normal_scale=False,Max=True):
    '''
    Regresa el promedio de datos.
    GetMeanData(diversity,bins size)
    option max/min
    '''
    if len(np.shape(diversity)) == 2:
        num_words = diversity.size/2
        diversity = diversity[:,1]
    else:
        num_words = np.size(diversity)

    # promedio de los datos

    nw_log_10 = np.log10(num_words)
    rank_logspace = np.log10(np.arange(1,num_words+1))
    limit = np.int(np.round((nw_log_10/size_bins)))
    bins = np.linspace(0,nw_log_10,num=limit+1)
    diversity_mean  = np.zeros((limit,2))
    diversity_mean[:,0] = bins[1:]
    k = 0

    nodata_between_points = []
    while k < limit:
        index = np.logical_and(rank_logspace>=bins[k],rank_logspace<=bins[k+1])
        if index.any():
            # diversity_mean[k][1] = np.mean(diversity[:,1][index])
            if Max:
                diversity_mean[k][1] = np.max(diversity[index]) 
            else:
                diversity_mean[k][1] = np.min(diversity[index]) 
        else:
            nodata_between_points.append(k)
        k += 1    

    if len(nodata_between_points) != 0:
        for i in nodata_between_points:
            diversity_mean[i][1] = (diversity_mean[i+1][1] + diversity_mean[i-1][1])/2.

    # diversity_mean[1:3,1] = diversity[:2,1]
    diversity_mean[1:3,1] = diversity[:2]
    diversity_mean[1:3,0] = rank_logspace[:2]

    rd_mean = diversity_mean[1:]
    if normal_scale:
        rd_mean[:,0] = np.power(10.,rd_mean[:,0])
    return rd_mean
    
def RankDiv(diversity_mean,witherror=False):
    parametros, error = curve_fit(funcion_erf,diversity_mean[:,0],diversity_mean[:,1])
    # parametros = media,desv
    if witherror:
#        parametros.append(error)
        np.append(parametros,error)
    return parametros

################################################################################
# Esta parte es para la diversidad de rango. Fin
################################################################################


################################################################################
# Esta parte es para la graficar la diversidad de rango. Inicio
################################################################################

from scipy.optimize import curve_fit

def get_data_to_fit(data):
    randint = np.random.randint
    join = np.concatenate

    # Valores para low-high de los indices 
    exp_mayor = 8
    limites = np.empty(exp_mayor,dtype='u4')
    for i in xrange(exp_mayor):
        limites[i] = 10**(i+1)

    #frequencies = np.loadtxt(anio+'_freq',dtype={'names': ('word','freq'),'formats': ('a50','f4')})
    num_words = data.size
    exponente = np.int(np.log10(num_words))

    indices = np.empty(0,'u4')
    for i in xrange(100):
        indices = join((indices,np.arange(10)))
        for j in xrange(exp_mayor-1):
            indices = join((indices,randint(limites[j],high=limites[j+1],size=10)))

    # set data
    indices.sort()
    indices = indices[indices<num_words]
    data_fit = np.empty((indices.size,2),dtype='u4')
    data_fit[:,0] = indices + 1
    data_fit[:,1] = data[indices]

    return data_fit

def model1(x,m,b):
    return m*x + b

def model1_bis(x,b):
    return x + b

def ajuste(data_fit,funcion):
    X = data_fit[:,0]
    Y = data_fit[:,1]
    X_log10 = np.log10(X)
    Y_log10 = np.log10(Y) 
    param_model, pcov1 = curve_fit(funcion,X_log10,Y_log10)

    return param_model

def deleteNaN(arreglo):
    return arreglo[~np.isnan(arreglo).any(1)]

def error_cuadratico_medio(y,f_x):
    return np.sqrt(np.power(y-f_x,2.).sum()/y.size)

import matplotlib.pyplot as plt
plt.rc('text',usetex=True)
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"] # solo para poder utlizar el entorno cases



def plot_rankDiv(axe_,diversity,ymax=1.0,xmax=10**5,size_bins=0.1,
                 showHT=False,showFit=True,showParam=False,windowing=False,
                 colour='blue',labels=""): 
    """
    Podria mejorarse anadiendo otras funciones a la que ajustar los datos y lmitando la cantidad de 
    k para hacer el fit
    """
    x_fit = np.logspace(0,5,num=100)
    diversity_mean = GetMeanData(diversity,size_bins)
    media,desv = RankDiv(diversity_mean, witherror=True)
    x = diversity[:,0]
    y = diversity[:,1]
    x_mean = diversity_mean[:,0]
    y_mean = diversity_mean[:,1]

    axe_.semilogx(x,y,marker='o',ms=5,mec='None',linestyle='None',alpha=0.4,color=colour,
                  rasterized=True)

    if showParam:
        axe_.text(0.7,0.05,latexarray([media,desv],r"\mu,\sigma"),
                  horizontalalignment = 'left',
                  verticalalignment = 'bottom',
                  fontsize = 16,
                  color = colour,
                  transform = axe_.transAxes)
                  # bbox = dict(boxstyle='square',ec='black',fc='white'))
    if showFit:
        axe_.semilogx(x_fit,funcion_erf(np.log10(x_fit),media,desv),color=colour,lw=3,label=labels)
#        f = funcion_erf(np.log10(x_fit),media,desv)
#        axe_.set_xscale("log")
#        axe_.set_yscale("log")
#        axe_.errorbar(x_fit, funcion_erf(np.log10(x_fit),media,desv),color=colour,lw=2,label=labels, yerr=media, fmt = 'b')
        
    if windowing:
        axe_.plot(10**x_mean,y_mean,'g-',lw=4)
    if showHT:
        cabeza_cola = np.r_[media-2*desv,media+2*desv]
        axe_.vlines(cabeza_cola[0],0,1,color='m',lw=5)
        axe_.vlines(cabeza_cola[1],0,1,color='m',lw=5)
        axe_.text(0.05,0.8, latexarray(10**cabeza_cola,r"k_-,k_+"),
                  horizontalalignment = 'left',
                  verticalalignment = 'top',
                  fontsize = 18,
                  bbox = dict(boxstyle='square',ec='black',fc='white'))

    axe_.set_xlim(0,xmax)
    axe_.set_ylim(0,ymax)

################################################################################
# Esta parte es para la graficar la diversidad de rango. Fin
################################################################################


################################################################################
# Esta parte es para las simulaciones. Inicio
################################################################################

class AlgorithWords:
    def __init__(self,mu,alpha,numwords):
        self.alpha = alpha
        self.mu = mu
        self.rand = norm.rvs
        self.ranks = np.arange(numwords)+1
        self.lista = np.empty(numwords,dtype={'names':('words','ranks'),'formats':('u8','f4')}) 
    def new_rank(self,numero):
        return numero + self.rand(self.mu,numero*self.alpha)
    def step(self,words):
        self.lista['words'] = words
        self.lista['ranks'] = map(self.new_rank,self.ranks)
        self.lista.sort(order='ranks')
        return self.lista['words']
    
class Simulation:
    def __init__(self,T,numwords):
        self.T, self.numwords = T,numwords
    def __repr__(self):
        return 'Diversity of '+ str(self.numwords) + ' during ' + str(self.T)
    def sim_data(self,alpha,mu):
        words = np.arange(self.numwords)+1
        tabla = np.empty((self.T,self.numwords),dtype='u4')
        self.alpha = alpha
        algoritmo = AlgorithWords(mu,alpha,self.numwords)
        for anio in xrange(self.T):
            tabla[anio] = words
            words = algoritmo.step(words)   
        tabla[anio] = words
        self.all_data = tabla
    def rd_byrank(self,Delta_t,delta_T=1,normed=True,fulldata=False):
        different_words_per_rank = np.empty(self.numwords)
        for i in xrange(self.numwords):
            different_words_per_rank[i] = np.unique(self.all_data[::delta_T,i][:Delta_t]).size
        if normed:
            new_Delta_t = self.all_data[::delta_T,0][:Delta_t].size
            different_words_per_rank =  np.divide(different_words_per_rank,np.float(new_Delta_t))
        self.dwpr = different_words_per_rank

        if fulldata:
            ranks = np.arange(self.numwords)+1
            ranks_diversity = np.empty((self.numwords,2))
            ranks_diversity[:,0],ranks_diversity[:,1] = ranks, different_words_per_rank
            output_diversity = ranks_diversity
        else:
            output_diversity = different_words_per_rank
        if Delta_t != new_Delta_t:
            print("Delta_t changes, Delta_t = %d" % new_Delta_t)
        return output_diversity
    def rd_bybins(self,binsize=0.1):
        nw_log_10 = np.log10(self.numwords)
        rank_logspace = np.log10(np.arange(self.numwords)+1)
        limit = np.int(np.round((nw_log_10/binsize)))
        bins = np.linspace(0,nw_log_10,num=limit+1)
        diversity_mean = np.empty((limit,2))
        diversity_mean[:,0] = bins[1:]
        k = 0

        nodata_between_points = []
        while k < limit:
            index = np.logical_and(rank_logspace>=bins[k],rank_logspace<=bins[k+1])
            if index.any():
                diversity_mean[k][1] = np.mean(self.dwpr[index]) 
            else:
                nodata_between_points.append(k)
                # diversity_mean[k][1] = 0.
            k += 1

        if len(nodata_between_points) != 0:
            for i in nodata_between_points:
                diversity_mean[i][1] = (diversity_mean[i+1][1] + diversity_mean[i-1][1])/2.

        diversity_mean[1:3,1] = self.dwpr[:2] #
        diversity_mean[1:3,0] = rank_logspace[:2]  #

        self.diversity = diversity_mean[1:]

        return diversity_mean[1:]

    def funcion_erf(self,X,mu,sigma):
        return norm.cdf(X,loc=mu,scale=sigma)

    def ajuste(self,maxwords=10000):
        mu,sigma = curve_fit(self.funcion_erf,self.diversity[:,0][:maxwords],self.diversity[:,1][:maxwords])[0]
        return mu, sigma
    def traject(self,rank):
        x,y = np.where(self.all_data == rank)
        return x,y+1
    def get_data(self):
        return self.all_data 

################################################################################
# Esta parte es para la graficar las simulaciones. Fin
################################################################################


################################################################################
# Esta parte es para graficar las trayectorias. Inicio
################################################################################

from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

# Topics: line, color, LineCollection, cmap, colorline, codex
'''
Defines a function colorline that draws a (multi-)colored 2D line with coordinates x and y.
The color is taken from optional data in z, and creates a LineCollection.

z can be:
- empty, in which case a default coloring will be used based on the position along the input arrays
- a single number, for a uniform color [this can also be accomplished with the usual plt.plot]
- an array of the length of at least the same length as x, to color according to this data
- an array of a smaller length, in which case the colors are repeated along the curve

The function colorline returns the LineCollection created, which can be modified afterwards.

See also: plt.streamplot
'''

# Data manipulation:
def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    return segments

#autumn cool hot spring summer winter
# Interface to LineCollection:
def colorline(x, y, ax, z=None, cmap=plt.get_cmap('summer'), norm=plt.Normalize(0.0, 1.0), linewidth=1.3, alpha=1.0):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''
    
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
           
    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])
        
    z = np.asarray(z)
    
    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    
    ax = plt.gca()
    ax.add_collection(lc)
    
    return lc

def clear_frame(ax=None): 
    # Taken from a post by Tony S Yu
    if ax is None: 
        ax = plt.gca() 
    ax.xaxis.set_visible(False) 
    ax.yaxis.set_visible(False) 
    for spine in ax.spines.itervalues(): 
        spine.set_visible(False) 


def ask4file(filename):
    files = listdir('.')
    if filename in files: 
        answer = True
    else:
        answer = False
    return answer

def plot_spaguettis(axe_,list_objects,dirFile,colormap='cool'):
    """
    De una lista de objetos dibuja sus trayectorias y los reparte 
    """
    here = getcwd()
    x = np.loadtxt(dirFile+'/time_separation')
    files = np.loadtxt(dirFile+'/time_list',dtype='a50')

    numObjetcs = len(list_objects)
    N = float(numObjetcs)
    plt.sca(axe_)

    for i in xrange(numObjetcs):
        color = 1-(i / N)
        chdir(dirFile+'/trajects')
        if not ask4file(list_objects[i]):
            chdir(dirFile+'/data/')
            y = find_traject(list_objects[i],files)
            np.savetxt(dirFile+'/trajects/'+list_objects[i],y,fmt='%.0f')
        else:
            y = np.loadtxt(list_objects[i])
        colorline(x, y, axe_, color,cmap=plt.get_cmap(colormap))
    chdir(here)


def plot_spaguettis_sim(axe,list_objects,simulacion,colormap='rainbown'):
    """
    Trayecotias simuladas
    """
    plt.sca(axe)
    numObjetcs = len(list_objects)
    N = float(numObjetcs)
    for i in xrange(numObjetcs):
        color = 1-(i / N)
        x,y = simulacion.traject(list_objects[i])
        colorline(x, y, axe, color,cmap=plt.get_cmap(colormap))


################################################################################
# Esta parte es para graficar las trayectorias. Fin
################################################################################


################################################################################
# Esta parte es para obtenrer y graficar la sigma vs.k. inicio
################################################################################


def get_st_from_differences(star_,num_words,init_year,num_years,dirFile,notNaN=True):
    data_file = np.loadtxt(dirFile+'/data/'+star_,dtype={'names': ('words','freq'),'formats': ('a50','f4')})
    words = data_file['words']
    n_words = words.size
    if num_words > n_words:
        num_words = n_words

    time_list = np.loadtxt(dirFile+'/time_list',dtype='a50')
    mu_sigma = np.empty((num_words,2))

    j=0
    for word in words[:num_words]:
        chdir(dirFile+'/trajects')
        if not ask4file(word):
            chdir(dirFile+'/data/')
            trajectory = find_traject(word,time_list)
            np.savetxt(dirFile+'/trajects/'+word,trajectory,fmt='%.0f')
        else:
            trajectory = np.loadtxt(word)
        trajectory = trajectory[init_year:init_year+num_years]
        diferencias = np.diff(trajectory)
        diferencias = diferencias[~np.isnan(diferencias)]
        mu_sigma[j] =  norm.fit(diferencias)
        j+=1
    datos_ajuste = np.empty((num_words,2))
    datos_ajuste[:,0], datos_ajuste[:,1] = np.arange(num_words)+1,mu_sigma[:,1]
    if notNaN:
        datos_ajuste = deleteNaN(datos_ajuste)
    return datos_ajuste


def get_st_from_differences_sim(simulacion,num_words,num_years):
    mu_sigma = np.empty((num_words,2))

    j=0
    for word in xrange(1,num_words+1):
        trajectory = simulacion.traject(word)
        diferencias = np.diff(trajectory)
        mu_sigma[j] = norm.fit(diferencias)
        j+=1
    datos_ajuste = np.empty((num_words,2))
    datos_ajuste[:,0], datos_ajuste[:,1] = np.arange(num_words)+1,mu_sigma[:,1]

    return datos_ajuste


def plot_sigma_hat(axe,datos_ajuste,params='one',showMeanData=False):
    ranks,sigma_hat = datos_ajuste[:,0],datos_ajuste[:,1]
    datos_ = GetMeanData(datos_ajuste)
    log_ranks, sigma_mean = datos_[:,0], datos_[:,1]
    ranks_mean,log_sigma = 10**log_ranks,np.log10(sigma_mean)

    n_words = ranks[-1]
    if params == 'two':
        m,b = curve_fit(model1,log_ranks,log_sigma)[0]
        y = 10**model1(log_ranks,m,b)
        axe.loglog(ranks_mean,y,'r-',lw=4)
        axe.text(0.05,0.95, r"\begin{eqnarray*} \hat{\sigma} & = & 10^b k^m \\ m & = & %.2f \\ b &= & %.2f \end{eqnarray*}" % (m,b), 
                horizontalalignment = 'left',
                verticalalignment = 'top',
                fontsize = 16,
                transform = axe.transAxes,
                bbox = dict(boxstyle='square',ec='black',fc='white'))
    elif params == 'one':
        b = curve_fit(model1_bis,log_ranks,log_sigma)[0]
        y = 10**model1_bis(log_ranks,b)
        axe.loglog(ranks_mean,y,'r-',lw=4)
        axe.text(0.05,0.95, r"\begin{eqnarray*} \hat{\sigma} & = & \alpha k \\ \alpha & = & %.2f \end{eqnarray*}" % 10**b, 
                horizontalalignment = 'left',
                verticalalignment = 'top',
                fontsize = 16,
                transform = axe.transAxes,
                bbox = dict(boxstyle='square',ec='black',fc='white'))
    elif params == 'both':
        m,b = curve_fit(model1,log_ranks,log_sigma)[0]
        b_ = curve_fit(model1_bis,log_ranks,log_sigma)[0]
        y = 10**model1(log_ranks,m,b)
        y_ = 10**model1_bis(log_ranks,b_)
        
        axe.loglog(ranks_mean,y,'r-',lw=4)
        axe.loglog(ranks_mean,y_,'m-',lw=4)

        axe.text(0.05,0.9,r"$\hat{\sigma} = 10^b k^m$",fontsize=14,color="red",transform = axe.transAxes)
        axe.text(0.05,0.8,r"$\hat{\sigma} = \alpha k$",fontsize=14,color="magenta",transform = axe.transAxes)
        axe.text(0.05,0.7,r"$m = %.2f$" % m,fontsize=14,transform = axe.transAxes)
        axe.text(0.05,0.6,r"$b = %.2f$" % b,fontsize=14,transform = axe.transAxes)
        axe.text(0.05,0.5,r"$\alpha = %.2f$" % 10**b_,fontsize=14,transform = axe.transAxes)

    axe.loglog(ranks,sigma_hat,'bo')
    if showMeanData:
        axe.loglog(ranks_mean,sigma_mean,'g-',lw=5)
    axe.set_ylim(y[0],n_words*2)
    axe.set_xlim(0,n_words)
    axe.set_ylabel(r"$\hat{\sigma}$")
    axe.set_xlabel(r"$k$")    
    

################################################################################
# Esta parte es para obtenrer y graficar la sigma vs.k. fin
################################################################################


################################################################################
# Esta parte es para las miscelaneo. inicio
################################################################################


def words2plot(ranks_list,dirFile,filename):
    chdir(dirFile+'/data') 
    list_words = []
    stop = len(ranks_list)
    i = 0
    
    with open(filename,'r') as file_open: 
        rank = 1
        for row in file_open:
            if rank == ranks_list[i]:
                list_words.append(row.split()[0])
                i += 1
                if i == stop:
                    break
            rank += 1   
    return list_words 

def removelabel(axe_,eje='x'):
    if eje == 'x':
        axe_.set_xticklabels([])
    elif eje == 'y':
        axe_.set_yticklabels([])
    elif eje == 'both':
        axe_.set_xticklabels([])
        axe_.set_yticklabels([])
    else:
        print('Invalid axis')

################################################################################
# Esta parte es para las miscelaneo. fin
################################################################################


################################################################################
# Esta parte es de los histogramas. Inicio
################################################################################


def plot_histogram_several(axe,init_rank,num_ranks,init_year,num_years,dirFile,normed=True,tipo='both'):
    time_list = np.loadtxt(dirFile+'/time_list',dtype='a50')
    freqList = np.loadtxt(dirFile+'/data/'+time_list[init_year], 
                          dtype={'names': ('word','freq'),'formats': ('a50','f4')})
    all_steps = []
    for word in freqList['word'][init_rank:init_rank+num_ranks]: 
        chdir(dirFile+'/trajects/')
        if not ask4file(word):
            chdir(dirFile+'/data/')
            word_traject = find_traject(word,time_list)
            np.savetxt(dirFile+'/trajects/'+word,word_traject,fmt='%.0f')
        else:
            word_traject = np.loadtxt(dirFile+'/trajects/'+word)

        word_traject = np.loadtxt(dirFile+'/trajects/'+word)
        word_steps = np.diff(word_traject[init_year:init_year+num_years])
        if normed:
            word_steps = word_steps/word_traject[init_year:init_year+num_years-1]
        word_steps_clean  = word_steps[~np.isnan(word_steps)]
        for step in word_steps_clean:
            all_steps.append(step)

    word_steps_clean = np.array(all_steps)
    if word_steps_clean.size > 10:
        # pdf, bins, patches = axe.hist(word_steps_clean,bins=40,normed=1,color='cyan',alpha=0.5,histtype='stepfilled') 
        pdf, bins, patches = axe.hist(word_steps_clean,bins=40,normed=1,color='cyan',alpha=0.5)        
        area = np.sum(pdf*np.diff(bins))       
 
        if normed :
            x = np.linspace(-0.75,0.75,num=200)
        else:
            x = np.linspace(bins[0],bins[-1],num=200)

        if tipo == 'gauss':
            mu,sigma =  norm.fit(word_steps_clean)
            y = norm.pdf(x,loc=mu,scale=sigma)*area
            axe.plot(x,y,'r-',lw=4,label='Gaussiana')
        elif tipo == 'lorentz':
            mu, sigma = cauchy.fit(word_steps_clean)
            y = cauchy.pdf(x,loc=mu,scale=sigma)*area
            axe.plot(x,y,'m--',lw=4,label='Lorentziana')
        elif tipo == 'both':
            mu,sigma =  norm.fit(word_steps_clean)
            y = norm.pdf(x,loc=mu,scale=sigma)*area
            axe.plot(x,y,'r-',lw=2,label='Gaussiana')
            mu, sigma = cauchy.fit(word_steps_clean)
            y = cauchy.pdf(x,loc=mu,scale=sigma)*area
            axe.plot(x,y,'m--',lw=2,label='Lorentziana')
            
        #axe.legend(loc='center left',prop={'size':16})        
        label_ = 'IniRank: '+ str(init_rank)+ '\nNumRanks: ' + str(num_ranks)
        axe.text(0.05,0.95, label_, 
                horizontalalignment = 'left',
                verticalalignment = 'top',
                fontsize = 16,
                transform = axe.transAxes,
                bbox = dict(boxstyle='square',ec='black',fc='white'))

    else:
        print('No hay suficientes datos para hacer un histograma de ' + word)


def plot_histogram(axe,word,init_year,num_years,dirFile,normed=True,tipo='both'):
    word_traject = np.loadtxt(dirFile+'/trajects/'+word)
    #word_steps = np.diff(word_traject)
    word_steps = np.diff(word_traject[init_year:init_year+num_years])
    if normed:
        word_steps = word_steps/word_traject[init_year:init_year+num_years-1]
    word_steps_clean  = word_steps[~np.isnan(word_steps)]

    if word_steps_clean.size > 10:
        pdf, bins, patches = axe.hist(word_steps_clean,bins=15,normed=1,color='cyan',alpha=0.5,histtype='stepfilled')        
        area = np.sum(pdf*np.diff(bins))       
 
        if normed :
            x = np.linspace(-0.75,0.75,num=200)
        else:
            x = np.linspace(bins[0],bins[-1],num=200)

        if tipo == 'gauss':
            mu,sigma =  norm.fit(word_steps_clean)
            y = norm.pdf(x,loc=mu,scale=sigma)*area
            axe.plot(x,y,'r-',lw=4,label='Gaussiana')
        elif tipo == 'lorentz':
            mu, sigma = cauchy.fit(word_steps_clean)
            y = cauchy.pdf(x,loc=mu,scale=sigma)*area
            axe.plot(x,y,'m--',lw=4,label='Lorentziana')
        elif tipo == 'both':
            mu,sigma =  norm.fit(word_steps_clean)
            y = norm.pdf(x,loc=mu,scale=sigma)*area
            axe.plot(x,y,'r-',lw=4,label='Gaussiana')
            mu, sigma = cauchy.fit(word_steps_clean)
            y = cauchy.pdf(x,loc=mu,scale=sigma)*area
            axe.plot(x,y,'m--',lw=4,label='Lorentziana')
            
        axe.legend(loc='center left',prop={'size':16})        
        axe.text(0.05,0.95, word, 
                horizontalalignment = 'left',
                verticalalignment = 'top',
                fontsize = 16,
                transform = axe.transAxes,
                bbox = dict(boxstyle='square',ec='black',fc='white'))
    else:
        print('No hay suficientes datos para hacer un histograma de ' + word)


def plot_histogram_sim(axe,rank,simulation,normalizado=True,tipo=None):
    word_traject = simulation.traject(rank)
    word_steps = np.diff(word_traject[1])

    if normalizado:
        word_steps = word_steps/map(float,word_traject[1][:-1])
    pdf,bins,patches = axe.hist(word_steps,bins=10,normed=True,alpha=0.5,histtype='stepfilled',color='blue')

    area = np.sum(pdf*np.diff(bins))       
 
    if normalizado:
        x = np.linspace(-0.75,0.75,num=200)
    else:
        x = np.linspace(bins[0],bins[-1],num=200)

    if tipo == 'gauss':
        mu,sigma =  norm.fit(word_steps)
        y = norm.pdf(x,loc=mu,scale=sigma)*area
        axe.plot(x,y,'r-',lw=4,label='Gaussiana')
    elif tipo == 'lorentz':
        mu, sigma = cauchy.fit(word_steps)
        y = cauchy.pdf(x,loc=mu,scale=sigma)*area
        axe.plot(x,y,'m--',lw=4,label='Lorentziana')
    elif tipo == 'both':
        mu,sigma =  norm.fit(word_steps)
        y = norm.pdf(x,loc=mu,scale=sigma)*area
        axe.plot(x,y,'r-',lw=4,label='Gaussiana')
        mu, sigma = cauchy.fit(word_steps)
        y = cauchy.pdf(x,loc=mu,scale=sigma)*area
        axe.plot(x,y,'m--',lw=4,label='Lorentziana')

    axe.set_xlim(-0.75,0.75)
    # axe.legend(loc='center left',prop={'size':16})        
    axe.text(0.05,0.95, "Rango " +str(rank), 
             horizontalalignment = 'left',
             verticalalignment = 'top',
             fontsize = 12,
             transform = axe.transAxes,
             bbox = dict(boxstyle='square',ec='black',fc='white'))

    #axe.text(0.05,0.70,"Gaussiana",color='red',fontsize=12,transform = axe.transAxes)
    #axe.text(0.05,0.65,"Lorentziana",color='magenta',fontsize=12,transform = axe.transAxes)

################################################################################
# Esta parte es de los histogramas. FIN
################################################################################


################################################################################
# Esta parte es de los modelos. INICIO
################################################################################


def fmt_tex(realnumber,maxRound=3,withdolar=False):
    significant, exponent = '{:.2e}'.format(realnumber).split('e')
    exponent = int(exponent)
    if abs(exponent) < maxRound:
        a,b = str(round(realnumber,2)).split('.')
        if b == '0':
            stringNumber = r'{}'.format(a)        
        else:
            stringNumber = r'{}.{}'.format(a,b[:2])        
    else:
        stringNumber = r'{} \times 10^{{{}}}'.format(significant, exponent)
    if withdolar:
        stringNumber = r'$'+stringNumber+r'$'
    return stringNumber

def latexarray(valuesarray,mathletter):
    mathletter = mathletter.split(',')
    equals, newline, textarray = r" & = & ", r" \\ ", r""
    for i in range(len(valuesarray)):
        textarray += mathletter[i]+equals+fmt_tex(valuesarray[i])+newline
    textarray = r"\begin{eqnarray*}"+textarray[:-3]+r"\end{eqnarray*}"
    return textarray


class Modelos:
    def __init__(self,Nmax):
        self.log10_inv = 1.0/np.log(10)
        self.Nmax = Nmax
        self.eq_model = {1:r"$m_1(k)$",2:r"$m_2(k)$",3:r"$m_3(k)$",4:r"$m_4(k)$",5:r"$m_5(k)$"}
        self.eq_full = {1:r"$m_1(k) =\mathcal{N}_1  \frac{1}{k^a}$",
                        2:r"$m_2(k) =\mathcal{N}_2  \frac{e^{-bk}}{k^a}$",
                        3:r"$m_3(k) =\mathcal{N}_3  \frac{(n +1 -k)^d}{k^a}$",
                        4:r"$m_4(k) =\mathcal{N}_4 \frac{(n +1 -k)^{d}e^{-bk}}{k^a}$",
                        5:r"$m_5(k) =\mathcal{N} \begin{cases} \frac{1}{k} & k \le k_c  \\ \frac{k_c^{a'-1}}{k^{a'}} & k > k_c \end{cases}$"}
    def m1(self,x,B,a):
        return B -a*x 
    def m2(self,x,B,a,b):
        return B -a*x -b*np.power(10.,x)*self.log10_inv
    def m3(self,x,B,a,d):
        return B -a*x +d*np.log10(self.Nmax+1-np.power(10.,x))
    def m4(self,x,B,a,b,d):
        return B -a*x +d*np.log10(self.Nmax+1-np.power(10.,x)) -b*np.power(10.,x)*self.log10_inv 
    def m5(self,x,B,a,xc):
        array_size = x.size
        y = np.empty(array_size,dtype=np.float)
        k = 0  
        for i in xrange(array_size):
            if x[i] < xc:
                y[i] = B -x[i]
            else:
                y[i] = B -a*x[i] + xc*(a-1)
            k += 1
        return y        
    def etiquet(self,NumModel):
        return self.eq_model[NumModel]
    def eq_latex(self,NumModel):
        return self.eq_full[NumModel]

def plot_model(axe_,freqs,num_model,labmodel=False,labparam=False,axe_diff=None):   
    data_ = get_data_to_fit(freqs)
    Kmax = freqs.size
    modelo = Modelos(Kmax)
    x_ = np.arange(Kmax) + 1
    axe_.loglog(x_,freqs,marker='o',ms=5,mec='None',linestyle='None',color='forestgreen',rasterized=True)

    if num_model == 1:
        letters = r"a,N"
        pmtrs = ajuste(data_,modelo.m1)
        a1,a2 = pmtrs
        data_fit_model = modelo.m1(np.log10(x_),a1,a2)
    elif num_model == 2:
        letters = r"a,b,N"
        pmtrs = ajuste(data_,modelo.m2)
        a1,a2,a3 = pmtrs
        data_fit_model = modelo.m2(np.log10(x_),a1,a2,a3) 
    elif num_model == 3:
        letters = r"a,d,N"
        pmtrs = ajuste(data_,modelo.m3)
        a1,a2,a3 = pmtrs
        data_fit_model = modelo.m3(np.log10(x_),a1,a2,a3)
    elif num_model == 4:
        letters = r"a,b,d,N"
        pmtrs = ajuste(data_,modelo.m4)
        a1,a2,a3,a4 = pmtrs
        data_fit_model = modelo.m4(np.log10(x_),a1,a2,a3,a4)
    elif num_model == 5:
        letters = r"a,k_c,N"
        pmtrs = ajuste(data_,modelo.m5)
        a1,a2,a3 = pmtrs
        data_fit_model = modelo.m5(np.log10(x_),a1,a2,a3)

    axe_.loglog(x_,np.power(10.,data_fit_model),color='red',linestyle='solid',lw=3)

    values = np.concatenate((pmtrs[1:],np.r_[Kmax]))

    if labparam:
        axe_.text(0.4,0.5, latexarray(values,letters),
                  horizontalalignment = 'right',
                  verticalalignment = 'top',
                  fontsize = 16,
                  transform = axe_.transAxes)
    if labmodel:
        axe_.text(0.1, 0.2,modelo.etiquet(num_model),fontsize=18,color="red",transform = axe_.transAxes)

    if axe_diff is not None:
        axe_diff.semilogx(x_,np.log10(freqs)-data_fit_model,label=eq_model[num_model])



class Modelos1:
    def __init__(self,Nmax,Y_max):
        self.log10_inv = 1.0/np.log(10)
        self.Nmax = Nmax
        self.logY_max = np.log10(Y_max)
        self.eq_model = {1:r"$m_1(k)$",2:r"$m_2(k)$",3:r"$m_3(k)$",4:r"$m_4(k)$",5:r"$m_5(k)$"}
        self.eq_full = {1:r"$m_1(k) =\mathcal{N}_1  \frac{1}{k^a}$",
                        2:r"$m_2(k) =\mathcal{N}_2  \frac{e^{-bk}}{k^a}$",
                        3:r"$m_3(k) =\mathcal{N}_3  \frac{(n +1 -k)^d}{k^a}$",
                        4:r"$m_4(k) =\mathcal{N}_4 \frac{(n +1 -k)^{d}e^{-bk}}{k^a}$",
                        5:r"$m_5(k) =\mathcal{N} \begin{cases} \frac{1}{k^c} & k \le k_c  \\ \frac{k_c^{a'-1}}{k^{a'}} & k > k_c \end{cases}$"}
    def m1(self,x,B,a):
        return B -a*x -self.logY_max
    def m2(self,x,B,a,b):
        return B -a*x -b*np.power(10.,x)*self.log10_inv-self.logY_max
    def m3(self,x,B,a,d):
        return B -a*x +d*np.log10(self.Nmax+1-np.power(10.,x)) -self.logY_max
    def m4(self,x,B,a,b,d):
        return B -a*x +d*np.log10(self.Nmax+1-np.power(10.,x)) -b*np.power(10.,x)*self.log10_inv -self.logY_max
    def m5(self,x,B,c,a,xc):
        array_size = x.size
        y = np.empty(array_size,dtype=np.float)
        k = 0  
        for i in xrange(array_size):
            if x[i] < xc:
                y[i] = B -c*x[i] -self.logY_max
            else:
                y[i] = B -a*(x[i]-xc)  -(xc*c) -self.logY_max
            k += 1
        return y        
    def etiquet(self,NumModel):
        return self.eq_model[NumModel]
    def eq_latex(self,NumModel):
        return self.eq_full[NumModel]

def plot_model1(axe_,freqs,num_model,labmodel=False,labparam=False,axe_diff=None):   
    data_ = get_data_to_fit(freqs[freqs>=1])
    Kmax = freqs.size
    modelo = Modelos1(Kmax,freqs[0])
    x_ = np.arange(Kmax) + 1
    axe_.loglog(x_,freqs/freqs[0],marker='o',ms=5,mec='None',linestyle='None',color='forestgreen',rasterized=True)

    if num_model == 1:
        letters = r"a,N"
        pmtrs = ajuste(data_,modelo.m1)
        a1,a2 = pmtrs
        data_fit_model = modelo.m1(np.log10(x_),a1,a2)
    elif num_model == 2:
        letters = r"a,b,N"
        pmtrs = ajuste(data_,modelo.m2)
        a1,a2,a3 = pmtrs
        data_fit_model = modelo.m2(np.log10(x_),a1,a2,a3) 
    elif num_model == 3:
        letters = r"a,d,N"
        pmtrs = ajuste(data_,modelo.m3)
        a1,a2,a3 = pmtrs
        data_fit_model = modelo.m3(np.log10(x_),a1,a2,a3)
    elif num_model == 4:
        letters = r"a,b,d,N"
        pmtrs = ajuste(data_,modelo.m4)
        a1,a2,a3,a4 = pmtrs
        data_fit_model = modelo.m4(np.log10(x_),a1,a2,a3,a4)
    elif num_model == 5:
        letters = r"c,a,k_c,N"
        pmtrs = ajuste(data_,modelo.m5)
        a1,a2,a3,a4 = pmtrs
        data_fit_model = modelo.m5(np.log10(x_),a1,a2,a3,a4)

    axe_.loglog(x_,np.power(10.,data_fit_model)/freqs[0],color='red',linestyle='solid',lw=3)

    values = np.concatenate((pmtrs[1:],np.r_[Kmax]))

    if labparam:
        axe_.text(0.4,0.5, latexarray(values,letters),
                  horizontalalignment = 'right',
                  verticalalignment = 'top',
                  fontsize = 16,
                  transform = axe_.transAxes)
    if labmodel:
        axe_.text(0.1, 0.2,modelo.etiquet(num_model),fontsize=18,color="red",transform = axe_.transAxes)

    if axe_diff is not None:
        axe_diff.semilogx(x_,np.log10(freqs)-data_fit_model,label=eq_model[num_model])



################################################################################
# Esta parte es de los modelos. FIN
################################################################################

