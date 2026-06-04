# -*- coding: utf-8 -*-
"""
This script generates the plots from compiled data, to reproduce the figures
and results from the "OpenCap: 3D human movement dynamics from smartphone videos" paper.

Before running the script, you need to download the data from simtk.org/opencap. 
See the README in this folder for more details.

Authors: Scott Uhlrich, Antoine Falisse
"""

import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import pingouin as pg
plt.close('all')
import sys

repoDir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../'))
dataDir = os.path.join(repoDir, 'Data')

sys.path.append(repoDir) # utilities from base repository directory
sys.path.append(os.path.join(repoDir,'DataProcessing')) # utilities in child directory

from matplotlib import gridspec
from sklearn.metrics import mean_absolute_error, roc_auc_score, accuracy_score, roc_curve
from utilsDataPostprocessing import calc_r2, calc_LSI


#%%  Directories
resultsDir = os.path.join(dataDir,'Outputs','Results-paper')
figDir = os.path.join(dataDir,'Outputs','Figures','Paper')

#%% User inputs
saveErrors = True
fieldStudy = True # True for both LabValidation and FieldStudy results. False for only LabValidation results.
saveFigures = True

if fieldStudy:
    subjects_fs = ['subject' + str(sub) for sub in range(100)]

subjects = ['subject' + str(sub) for sub in range(2,12)]


#%% Process settings
all_motions = ['squats', 'squatsAsym', 'walking', 'walkingTS', 'DJ', 'DJAsym', 'STS', 'STSweakLegs']

# Likely fixed settings
suffix_motion_name = '_videoAndMocap'
data_type = 'Video' # only set up for video now
modalityFolderName = data_type
poseDetector = 'HRnet'
cameraSetup = '2-cameras'

if fieldStudy:
    # variables to be flipped when loading fieldStudy
    fsIn = {}
    fsIn['suffix_motion_name'] = '_LSTM'
    fsIn['poseDetector'] = 'OpenPose'
    fsIn['cameraSetup'] = 'all-cameras'
    fsIn['data_type'] = 'Video' 
    fsIn['modalityFolderName'] = '' # no Video folder name
    fsIn['augmenter'] = '' # placeholder for results path
    fsIn['all_motions'] = ['squats', 'squatsAsym']


#%% Default plot settings
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'

#%% Loop over subjects
loadResultsNow = True
loadingfieldStudy = False
while loadResultsNow:
    loadResultsNow = False
    nSubjects = len(subjects)
    iSub=0
    for subject in subjects:
        print('loading subject {} of {}. '.format(iSub,nSubjects) + subject)

        
        if loadingfieldStudy:
            pathOSData = os.path.join(dataDir,'FieldStudy',subject,'OpenSimData', 
                                      'Dynamics')
            
            # Check for data folder
            if not os.path.isdir(pathOSData):
                raise Exception('The data is not found in ' + dataDir + '. Download it from https://simtk.org/projects/opencap, and save to the the repository directory. E.g., Data/FieldStudy')

            results = np.load(os.path.join(pathOSData,
                          'allActivityResults.npy'),allow_pickle=True).item()
        else:
            pathOSData = os.path.join(dataDir,'LabValidation',subject,'OpenSimData',
                                       modalityFolderName, poseDetector, 
                                       cameraSetup,'Dynamics')
            
            # Check for data folder
            if not  os.path.isdir(pathOSData):
                raise Exception('The data is not found in ' + dataDir + '. Download it from https://simtk.org/projects/opencap, and save to the the repository directory. E.g., Data/LabValidation')

            results = np.load(os.path.join(pathOSData, 
                          'allActivities_results.npy'),allow_pickle=True).item()
        
        # Preallocate dictionary 
        if iSub == 0:
            motTypes = list(results.keys())
            varNames = list(results[motTypes[0]]['video'].keys())
            varNames = [var for var in varNames if 'scalar' not in var]
            
            scalarNames = {}
            results_all = {}
            for motType in motTypes:
                if 'walking' in motType:
                    # if not subject in subjects_noMCF:
                    varNames = ['positions', 'velocities', 'accelerations', 'torques', 'torques_BWht', 'GRFs', 'GRFs_BW', 'GRMs', 'GRMs_BWht', 'activations', 'KAMs', 'KAMs_BWht', 'MCFs', 'MCFs_BW']
                elif 'DJ' in motType:
                    varNames = ['positions', 'velocities', 'accelerations', 'torques', 'torques_BWht', 'GRFs', 'GRFs_BW', 'GRMs', 'GRMs_BWht', 'activations', 'KAMs', 'KAMs_BWht']
                else:
                    varNames = ['positions', 'velocities', 'accelerations', 'torques', 'torques_BWht', 'GRFs', 'GRFs_BW', 'GRMs', 'GRMs_BWht', 'activations']
                results_all[motType] = {}
                for varName in varNames:
                    results_all[motType][varName] = {}
                    results_all[motType][varName]['headers'] = results[
                        motType]['video'][varName]['headers']
                    results_all[motType][varName]['ref'] = np.zeros(
                        (results[motType]['video'][varName]['ref_mean'].shape + (nSubjects,)))
                    results_all[motType][varName]['sim'] = np.zeros(
                        (results[motType]['video'][varName]['sim_mean'].shape + (nSubjects,)))
                    if 'so' in results[motType]['video'][varName]:
                        results_all[motType][varName]['so'] = np.zeros(
                            (results[motType]['video'][varName]['so_mean'].shape + (nSubjects,)))
                    results_all[motType][varName]['rmse'] = np.zeros(
                        (results[motType]['video'][varName]['rmse_mean'].shape + (nSubjects,)))
                    results_all[motType][varName]['mae'] = np.zeros(
                        (results[motType]['video'][varName]['mae_mean'].shape + (nSubjects,)))
                    results_all[motType][varName]['mape'] = np.zeros(
                        (results[motType]['video'][varName]['mape_mean'].shape + (nSubjects,)))
                results_all[motType]['scalars'] = {}
                results_all[motType]['scalar_mean'] = {}
                
                # Get scalar names as a function of mot type 
                scalarNames[motType] = list(results[motType]['video']['scalars'].keys())
                for scName in scalarNames[motType]:
                    results_all[motType]['scalars'][scName] = []
                    results_all[motType]['scalars'][scName + '_mean'] = []
    
            
        #%% Concatenate across subjects
        for motType in motTypes:
            if 'walking' in motType:
                varNames = ['positions', 'velocities', 'accelerations', 'torques', 'torques_BWht', 'GRFs', 'GRFs_BW', 'GRMs', 'GRMs_BWht', 'activations', 'KAMs', 'KAMs_BWht', 'MCFs', 'MCFs_BW']
            elif 'DJ' in motType:
                varNames = ['positions', 'velocities', 'accelerations', 'torques', 'torques_BWht', 'GRFs', 'GRFs_BW', 'GRMs', 'GRMs_BWht', 'activations', 'KAMs', 'KAMs_BWht']
            else:
                varNames = ['positions', 'velocities', 'accelerations', 'torques', 'torques_BWht', 'GRFs', 'GRFs_BW', 'GRMs', 'GRMs_BWht', 'activations']
            for varName in varNames:
                # Curves
                results_all[motType][varName]['ref'][:,:,iSub] = results[
                    motType]['video'][varName]['ref_mean']
                results_all[motType][varName]['sim'][:,:,iSub] = results[
                    motType]['video'][varName]['sim_mean']
                if 'so' in  results_all[motType][varName]:
                    results_all[motType][varName]['so'][:,:,iSub] = results[
                        motType]['video'][varName]['so_mean']
                
                # Error metrics
                results_all[motType][varName]['rmse'][:,:,iSub] = results[
                    motType]['video'][varName]['rmse_mean']
                results_all[motType][varName]['mae'][:,:,iSub] = results[
                    motType]['video'][varName]['mae_mean']
                results_all[motType][varName]['mape'][:,:,iSub] = results[
                    motType]['video'][varName]['mape_mean']
                
                # Mean and SD of curves and values
                if iSub == nSubjects-1:
                    results_all[motType][varName]['ref_mean'] = np.mean(
                        results_all[motType][varName]['ref'],axis=-1)
                    results_all[motType][varName]['sim_mean'] = np.mean(
                        results_all[motType][varName]['sim'],axis=-1)
                    results_all[motType][varName]['ref_std'] = np.std(
                        results_all[motType][varName]['ref'],axis=-1)
                    results_all[motType][varName]['sim_std'] = np.std(
                        results_all[motType][varName]['sim'],axis=-1)
                    if 'so' in results_all[motType][varName]:
                        results_all[motType][varName]['so_mean'] = np.mean(
                            results_all[motType][varName]['so'],axis=-1)
                        results_all[motType][varName]['so_std'] = np.std(
                            results_all[motType][varName]['so'],axis=-1)
                    # Error metrics
                    results_all[motType][varName]['rmse_mean'] = np.mean(
                        results_all[motType][varName]['rmse'],axis=-1)
                    results_all[motType][varName]['mae_mean'] = np.mean(
                        results_all[motType][varName]['mae'],axis=-1)
                    results_all[motType][varName]['mape_mean'] = np.mean(
                        results_all[motType][varName]['mape'],axis=-1)
                    
                
                
            for scalarName in scalarNames[motType]:
                # Scalars
                results_all[motType]['scalars'][scalarName].append(np.squeeze(results[
                    motType]['video']['scalars'][scalarName]).tolist())
                results_all[motType]['scalars'][scalarName + '_mean'].append(results[
                    motType]['video']['scalar_means'][scalarName + '_mean'])
        iSub +=1

            
    if not loadingfieldStudy:
        print('copied')
        results_all_validation = copy.deepcopy(results_all)
    
    if fieldStudy and not loadingfieldStudy:
        loadResultsNow = True
        loadingfieldStudy = True
        subjects = subjects_fs
        for key,val in fsIn.items():
            exec(key + '=val')
            
results_all_fieldStudy = copy.deepcopy(results_all)
del results_all
results_all = results_all_validation
del results_all_validation

# Save all results as a pickle
# with open(os.path.join(dataDir,'results_ALL.pkl'), 'wb') as f:
#     pickle.dump(results_all, f)

# %% Identify activities
possibleActivities = ['walking','DJ','STS','squats']
resultActivities = list(results_all.keys())
activities = {pA:any(pA in rA for rA in resultActivities) for pA in possibleActivities}


# %% Stats utilities
def ttestUtility(y1,y2,name=None):
    y1 = np.array(y1)
    y2 = np.array(y2)
    dY = y2-y1
    dY = dY.reshape(1,-1)
    _,p_normal1 = scipy.stats.shapiro(y1)
    _,p_normal2 = scipy.stats.shapiro(y2)
    
    if np.min([p_normal1, p_normal2]) <0.05:
        testType = 'wilcoxon'
        wilOut= pg.wilcoxon(y1,y2)
        centralTendency = np.median(dY)
        variance = scipy.stats.iqr(dY)/2
        # Bootstrap confidence interval
        bootOut=scipy.stats.bootstrap(dY, np.median,n_resamples=9999,random_state=1)
        CI = list(bootOut.confidence_interval)
        
        # repack
        testStat = wilOut['W-val'][0]
        p_output = wilOut['p-val'][0]
        ES = wilOut['CLES'][0]
        


        if name is not None:
            print(name + ' is not normally distributed. Using signed rank.')
            print('normality pVals=' + str(p_normal1) + ',' +  str(p_normal2))
    else:
        testType = 't test'
        tOut= pg.ttest(y2,y1,paired=True)
        p_output = tOut['p-val'][0]
        testStat = tOut['T'][0]
        CI = list(tOut['CI95%'][0])
        ES = tOut['cohen-d'][0]
        centralTendency = np.mean(dY)
        variance = np.std(dY)
        
    out = {'p_normal':[p_normal1,p_normal2],
        'p':p_output,
        'test_stat':testStat,
        'ES':ES,
        'CI':CI,
        'central_tendency':centralTendency,
        'variance':variance,
        'test_type':testType}
        
    return out

def posthocPower(x1,x2):
    # wilcoxon and t test are the same for paired samples (https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/PASS/Paired_Wilcoxon_Signed-Rank_Tests.pdf)
    dVec= np.subtract(x1,x2) 
    effectSize = np.mean(dVec)/np.std(dVec)
    
    from statsmodels.stats.power import TTestIndPower
    power_analysis = TTestIndPower()
    posthocPower = power_analysis.solve_power(effect_size = effectSize, 
                                             alpha=0.05, 
                                             nobs1=len(x1))
    
    return posthocPower

    
# %% Ensemble plotting function

def ensemblePlot(meanList,sdList,yLabelList,xLabelList,legList,titleList,yLim, fillSD=True,figSize=None,colors=None,nYticks=None,nXticks=None):
    
    SMALL_SIZE = 7  
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12
    
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    nSubplots = len(meanList)
    nCurves = len(meanList[0])
    
    if colors == None:
        rd = np.array((150, 68, 72)) / 255
        bl = np.array([88, 122, 191])/255
        colors = [0.3*np.ones(3),rd,bl,'k']
    
    if figSize == None:
        figSize = (nSubplots*2,2)
    figH = plt.figure(figsize=figSize,dpi=300)
    axH = []
    
    for iSub in range(nSubplots):
    
        
        if nSubplots > 1:
            plt.subplot(1,nSubplots,iSub+1)
            plt.plot([0,100],[0,0],color = [.5,.5,.5],linewidth = .5)
        
        for iC in range(nCurves):
            ax = plt.gca()   
            if fillSD:
                ax.fill_between(range(101),meanList[iSub][iC] + sdList[iSub][iC],
                    meanList[iSub][iC] - sdList[iSub][iC],facecolor=colors[iC], alpha=0.3)
            plt.plot(range(101),meanList[iSub][iC],color = colors[iC],label=legList[iSub][iC])
        
        plt.xlabel(xLabelList[iSub])
        plt.ylabel(yLabelList[iSub], multialignment='center')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if legList[iSub][0] != None:
            plt.legend(frameon = False)
        if titleList[iSub] != None:
            plt.title(titleList[iSub])
        xlim = [0, 100]
        ax.set_xlim(xlim)
        if yLim != None:
            if len(yLim) == 1:
                ax.set_ylim(yLim[0])
            else:
                ax.set_ylim(yLim[iSub])
        if nYticks != None:
            if yLim is None:
                yLim_t = list(ax.get_ylim())
                yticks = list(np.linspace(yLim_t[0], yLim_t[1], nYticks))            
            elif len(yLim) == 1:
                yticks = list(np.linspace(yLim[0][0], yLim[0][1], nYticks))
            else:
                 yticks = list(np.linspace(yLim[iSub][0], yLim[iSub][1], nYticks))
            ax.set_yticks(yticks)
        if nXticks != None:
            xticks = list(np.linspace(xlim[0], xlim[1], nXticks))
            ax.set_xticks(xticks)
                
        axH.append(ax)
        
    return figH,axH

# %% Ensemble plotting function: multiple rows

def ensemblePlotMRows(meanList,sdList,fields,fillSD=True,colors=None,
                      nColumns=3,nYticks=None,nXticks=None,labelsize=10,
                      titlesize=12,ticksize=9,bigger_size=12, dpi=100, 
                      figSize=(12,6)):
    
    # plt.rc('font', size=small_size)          # controls default text sizes
    plt.rc('axes', titlesize=titlesize)    # fontsize of the axes title
    plt.rc('axes', labelsize=labelsize)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=ticksize)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=ticksize)    # fontsize of the tick labels
    plt.rc('legend', fontsize=labelsize)    # legend fontsize
    plt.rc('figure', titlesize=bigger_size)  # fontsize of the figure title
    
    nSubplots = len(meanList)
    nCurves = len(meanList[0])
    
    if colors == None:
        rd = np.array((150, 68, 72)) / 255
        bl = np.array([88, 122, 191])/255
        colors = [0.3*np.ones(3),rd,bl,'k']
    
    fillSD = True
    figH, axs = plt.subplots(int(np.ceil(nSubplots/nColumns)),nColumns, dpi=dpi, figsize=figSize)
    coordinates = list(fields.keys())
    for iSub, ax in enumerate(axs.flat):
        if iSub < nSubplots:
            coord = coordinates[iSub]
            for iC in range(nCurves):
                if fillSD:
                    ax.fill_between(range(101),meanList[iSub][iC] + sdList[iSub][iC],
                        meanList[iSub][iC] - sdList[iSub][iC],facecolor=colors[iC], alpha=0.3)
                ax.plot(range(101),meanList[iSub][iC],color = colors[iC],label=fields[coord]['legend'][iC])
            
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)  
            
            ax.legend(frameon = False)
            
            # Title
            if 'title' in fields[coord]:
                ax.set_title(fields[coord]['title'], multialignment='center')
            # x limit
            xlim = [0, 100]
            ax.set_xlim(xlim)
            if nXticks != None:
                xticks = list(np.linspace(xlim[0], xlim[1], nXticks))
                ax.set_xticks(xticks)
            if 'xlabel' in fields[coord]:
                xticks = list(np.linspace(xlim[0], xlim[1], nXticks))
                ax.set_xticks(xticks)
                ax.set_xlabel(fields[coord]['xlabel'])
            else:
                ax.set_xticklabels([])                
            # y limit
            if 'yLim' in fields[coord]:
                ax.set_ylim(fields[coord]['yLim'])
            if nYticks != None:
                if not 'yLim' in fields[coord]:
                    yLim_t = list(ax.get_ylim())
                    yticks = list(np.linspace(yLim_t[0], yLim_t[1], nYticks))
                else:
                      yticks = list(np.linspace(fields[coord]['yLim'][0], fields[coord]['yLim'][1], nYticks))
                ax.set_yticks(yticks)
            if ax.get_ylim()[0] != 0 and ax.get_ylim()[1] != 0:
                ax.plot([0,100],[0,0],color = [.5,.5,.5],linewidth = .5)
                
                
            if 'ylabel' in fields[coord]:
                ax.set_ylabel(fields[coord]['ylabel'])
        else:
            ax.set_visible(False)            
    plt.setp(axs[:, 0], ylabel=fields['ylabel'])
        
    return figH,axs

#%% Plot Scalars 

def plotScalars(motTypes,colors,xName,yName,title=None,changes=False,bothLegs=False,
                figureNum=None,yLabel=None,xLabel=None,legendList=None):
    if figureNum is None:
        plt.figure()
    else:
        plt.figure(figureNum)
        
    if legendList == None:
        legendList = motTypes
        
    yVals = [] # for stacking them together
    xVals = []
    
    nLegs =1
    if bothLegs:
        nLegs = 2
    for i, motType in enumerate(motTypes):
        for iLeg in range(nLegs):
            if iLeg == 1:
                yName = yName.replace('_l','_r') # not robust, but don't always have _l_
                xName = xName.replace('_l','_r')
                legendList[i] = None
            plt.plot(results_all[motType]['scalars'][xName],results_all[motType]['scalars'][yName],
                marker='o',markerfacecolor=colors[i],markeredgecolor='none',linestyle='none',label=legendList[i])
            xVals.extend(results_all[motType]['scalars'][xName])
            yVals.extend(results_all[motType]['scalars'][yName])
            
            if changes and len(motTypes) == 2:
                plt.plot([results_all[motTypes[0]]['scalars'][xName],results_all[motTypes[1]]['scalars'][xName]],
                      [results_all[motTypes[0]]['scalars'][yName],results_all[motTypes[1]]['scalars'][yName]],
                      color=.3*np.array((1,1,1)),linewidth=.5)
            elif changes and len(motTypes) !=2:
                raise Exception('needs to be 2 motion types')
        if yLabel == None: 
            yLabel = yName
        plt.ylabel(yLabel)
        if xLabel == None:
            xLabel = xName
        plt.xlabel(xLabel)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    minAx = np.min([ylim,xlim])
    maxAx = np.max([ylim,xlim])
    plt.plot([minAx,maxAx],[minAx,maxAx],color='k')
    ax.set_aspect('equal')
    plt.legend(frameon=False)
    
    if title != None:
        plt.title(title)
    
    r2 = calc_r2(xVals,yVals)
    mae = mean_absolute_error(xVals,yVals)
    plt.text(.1,.8,'r2={:.2f} \nmae={:.2f}'.format(r2,mae),transform=ax.transAxes)
    
    return r2, mae

# %% Plot ROC and calculate AUC
def plotROC(y_class,y_pred,visualize=True,lineColor=None,title=None, ax=None):
    
    fpr,tpr,thresh = roc_curve(y_class,y_pred)
    aucVal = roc_auc_score(y_class, y_pred)
    
   
    
    if visualize:
        if lineColor is None:
            lineColor = np.array([88, 122, 191])/255
        if ax == None:
            plt.figure()
            ax = plt.gca()
        ax.plot(fpr,tpr,linewidth=1,color = lineColor)
        
        ax.text(.6,.2,'AUC={:.2f}'.format(aucVal),transform=ax.transAxes)
        ax.set_ylabel('True positive rate')
        ax.set_xlabel('False positive rate')
        ax.plot([0,1],[0,1],linestyle='--',color=0.5*np.ones(3),linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if title is not None:
            ax.set_title(title)
    
    return aucVal
    

#%% Scalars
r2, mae = {}, {}
r2['DJ'], mae['DJ'] = {}, {}
r2['squats'], mae['squats'] = {}, {}
r2['walking'], mae['walking'] = {}, {}
r2['STS'], mae['STS'] = {}, {}


# %% Average torque errors between limbs

acts = results_all.keys()
metrics = ['mae_mean','mape_mean','rmse_mean']
excludedBilateralDofs = ['elbow','arm','pro_sup','mtp']
for act in acts:
    results_all[act]['torques_BWht_limbAveraged'] = {}
    head = results_all[act]['torques_BWht']['headers']
    biHeadNames = [dof[:-2] for dof in head if dof[-2:] == '_l' and not any([eBD in dof for eBD in excludedBilateralDofs])]
    matchingPairs = [[head.index(bN+'_r'),head.index(bN+'_l')] for bN in biHeadNames]
    singHeadNames = [dof for dof in head[1:] if dof[-2:] not in ['_r','_l']]
    singHeadNames = [n for n in singHeadNames if 'lumbar' in n]
    nMP = len(matchingPairs)
    
    for met in metrics:
        results_all[act]['torques_BWht_limbAveraged'][met]= np.zeros(((len(singHeadNames)+len(biHeadNames)+1),1))
        results_all[act]['torques_BWht_limbAveraged']['headers'] = np.zeros((len(singHeadNames)+len(biHeadNames)+1)).tolist()
        results_all[act]['torques_BWht_limbAveraged']['headers'][0] = 'time'
        for i,mP in enumerate(matchingPairs):
            results_all[act]['torques_BWht_limbAveraged'][met][i+1] = np.mean(
                [results_all[act]['torques_BWht'][met][mP[0]],
                 results_all[act]['torques_BWht'][met][mP[1]]])
            results_all[act]['torques_BWht_limbAveraged']['headers'][i+1] = biHeadNames[i]
        for i,sHN in enumerate(singHeadNames):
             results_all[act]['torques_BWht_limbAveraged'][met][i+nMP+1] = results_all[
                 act]['torques_BWht'][met][head.index(sHN)]
             results_all[act]['torques_BWht_limbAveraged']['headers'][i+nMP+1] = sHN
             
# %% Average GRF errors between limbs            
acts = results_all.keys()
metrics = ['mae_mean','mape_mean','rmse_mean']
for act in acts:
    results_all[act]['GRFs_BW_limbAveraged'] = {}
    head = results_all[act]['GRFs_BW']['headers']
    biHeadNames = [dof for dof in head if dof[-5:-3] == '_l']
    matchingPairs = [[head.index(bN),head.index(np.char.replace(bN, '_l', '_r', count = 1))] for bN in biHeadNames]
    nMP = len(matchingPairs)
    biHeadReplacedNames = {'ground_force_l_vy':'GRF_vertical',
                           'ground_force_l_vx':'GRF_anterior',
                           'ground_force_l_vz':'GRF_medial'}
    
    
    for met in metrics:
        results_all[act]['GRFs_BW_limbAveraged'][met]= np.zeros(((len(biHeadNames)+1),1))
        results_all[act]['GRFs_BW_limbAveraged']['headers'] = np.zeros((len(biHeadNames)+1)).tolist()
        results_all[act]['GRFs_BW_limbAveraged']['headers'][0] = 'time'
        for i,mP in enumerate(matchingPairs):
            results_all[act]['GRFs_BW_limbAveraged'][met][i+1] = np.mean(
                [results_all[act]['GRFs_BW'][met][mP[0]],
                 results_all[act]['GRFs_BW'][met][mP[1]]])
            results_all[act]['GRFs_BW_limbAveraged']['headers'][i+1] = biHeadReplacedNames[biHeadNames[i]]

# %% Save errors
if saveErrors:
    dynamicQuants = ['torques_BWht','GRFs_BW','positions','torques_BWht_limbAveraged','GRFs_BW_limbAveraged']
    
    acts = ['DJ','walking','squats','STS']
    metrics = ['mae_mean','rmse_mean','mape_mean']
    
    for dynamicQuant in dynamicQuants:
    
        fileName = os.path.join(resultsDir,'KineticErrors',dynamicQuant + '_errors.xlsx')
        
        dfList = []
        errors = {}
        dofs = results_all[acts[0]][dynamicQuant]['headers']
        dofs = [dof for dof in dofs if dof != 'time']
        actNames = copy.deepcopy(acts)
        actNames.append('mean')
        for met in metrics:
            errors[met] = {}
            errors[met]['activities'] = actNames
            for dof in dofs:
                erList = []
                for iAct,act in enumerate(acts): 
                    idx = results_all[act][dynamicQuant]['headers'].index(dof)
                    erList.append(results_all[act][dynamicQuant][met][idx,0])
                erList.append(np.mean(erList))
                errors[met][dof] = copy.deepcopy(erList)
            
        
            dfList.append(pd.DataFrame(data=errors[met]))
        
        rootPath,_ = os.path.split(fileName)
        os.makedirs(rootPath,exist_ok=True)
            
        with pd.ExcelWriter(fileName) as writer:  
            for df,met in zip(dfList,metrics):
                df.to_excel(writer, sheet_name=met)
            

# %% Aggregated figures
SMALL_SIZE = 8
MEDIUM_SIZE = 9
BIGGER_SIZE = 10

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

rd = np.array((150, 68, 72)) / 255
bl = np.array([88, 122, 191])/255
ltbl = np.array([184,198,228])/255
gy = 0.5*np.ones(3)
bk = np.zeros(3)


# %% Figure 3: Knee loading
SMALL_SIZE = 8
MEDIUM_SIZE = 9
BIGGER_SIZE = 10

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title



if activities['walking']:
    dType = 'KAMs_BWht'
    colors = [gy,rd]
    
    idx = results_all['walking']['KAMs_BWht']['headers'].index('KAM_l')
    meanList = [
                [results_all['walking'][dType]['ref_mean'][idx,:],results_all['walkingTS'][dType]['ref_mean'][idx,:]],
                [results_all['walking'][dType]['sim_mean'][idx,:],results_all['walkingTS'][dType]['sim_mean'][idx,:]]
                ]
    sdList = [
              [results_all['walking'][dType]['ref_std'][idx,:],results_all['walkingTS'][dType]['ref_std'][idx,:]],
              [results_all['walking'][dType]['sim_std'][idx,:],results_all['walkingTS'][dType]['sim_std'][idx,:]]
              ]
    yLabelList = ['Mocap'+' KAM (%BW*ht)','OpenCap'+' KAM (%BW*ht)']
    xLabelList = ['% Stance','% Stance']
    legList = [[None,None],[None,None]]
    titleList = [None,None]
    yLim = [-1.9, 3.5]
    letterLabel = ['a','b']
    starLoc = [[27,3.1],[24,2.7]]
    
    #copied from ensemblePlot
    subplotNums = [1,2]
    nCurves = len(meanList[0])
    
    try :
        plt.close(plt.figure(3))
    except:
        _=None
    figH = plt.figure(3,figsize=(5.5,1.9),dpi=200)
    axH = []

   
    #KAM P1
    sim_KAM_cond1 = results_all['walking']['scalars']['sim_peakKAM_l_mean']
    sim_KAM_cond2 = results_all['walkingTS']['scalars']['sim_peakKAM_l_mean']
    stats_KAM_opencap = ttestUtility(sim_KAM_cond1,sim_KAM_cond2,name='KAM1 OpenCap')
    
    
    ref_KAM_cond1 = results_all['walking']['scalars']['ref_peakKAM_l_mean']
    ref_KAM_cond2 = results_all['walkingTS']['scalars']['ref_peakKAM_l_mean']
    stats_KAM_mocap = ttestUtility(ref_KAM_cond1,ref_KAM_cond2,name='KAM1 Mocap')
     
    sim_percDif = np.divide(np.subtract(sim_KAM_cond2,sim_KAM_cond1),sim_KAM_cond1) *100
    ref_percDif = np.divide(np.subtract(ref_KAM_cond2,ref_KAM_cond1),ref_KAM_cond1) *100
    
    # Post hoc power
    sim_power_KAM = posthocPower(sim_KAM_cond1,sim_KAM_cond2)
    ref_power_KAM = posthocPower(ref_KAM_cond1,ref_KAM_cond2)
    
    #MCF - not normally distributed
    sim_MCF_cond1 = results_all['walking']['scalars']['sim_peakMCF_l_mean']
    sim_MCF_cond2 = results_all['walkingTS']['scalars']['sim_peakMCF_l_mean']
    stats_MCF_opencap = ttestUtility(sim_MCF_cond1,sim_MCF_cond2,name='MCF1 OpenCap')
    
    ref_MCF_cond1 = results_all['walking']['scalars']['ref_peakMCF_l_mean']
    ref_MCF_cond2 = results_all['walkingTS']['scalars']['ref_peakMCF_l_mean']
    stats_MCF_mocap = ttestUtility(ref_MCF_cond1,ref_MCF_cond2,name='MCF1 Mocap')
     
    sim_percDif_MCF = np.divide(np.subtract(sim_MCF_cond2,sim_MCF_cond1),sim_MCF_cond1) *100
    ref_percDif_MCF = np.divide(np.subtract(ref_MCF_cond2,ref_MCF_cond1),ref_MCF_cond1) *100
        
    # Post hoc power
    sim_power_MCF = posthocPower(sim_MCF_cond1,sim_MCF_cond2)
    ref_power_MCF = posthocPower(ref_MCF_cond1,ref_MCF_cond2)
    
    # Correct for false discovery rate
    from statsmodels.stats.multitest import fdrcorrection
    _,p_corr_walking_opencap = fdrcorrection([stats_KAM_opencap['p'],stats_MCF_opencap['p']])
    _,p_corr_walking_mocap = fdrcorrection([stats_KAM_mocap['p'],stats_MCF_mocap['p']])

    
    print('Dif in P1 KAM between walking conditions. OpenCap: {:.1f}+/-{:.1f}% (p={:.3f}), power={:.2f}, testStat={:.2f}; Mocap: {:.1f}+/-{:.1f} (p={:.3f}), power={:.2f}, testStat={:.2f}'.format(
        np.mean(sim_percDif),np.std(sim_percDif),p_corr_walking_opencap[0],sim_power_KAM,stats_KAM_opencap['test_stat'],
        np.mean(ref_percDif),np.std(ref_percDif),p_corr_walking_mocap[0],ref_power_KAM,stats_KAM_mocap['test_stat']))
    
    print('Dif in P1 MCF between walking conditions. OpenCap: {:.1f}+/-{:.1f}% (p={:.3f}), power={:.2f}, testStat={:.2f}; Mocap: {:.1f}+/-{:.1f} (p={:.3f}), power={:.2f}, testStat={:.2f}'.format(
        np.mean(sim_percDif_MCF),np.std(sim_percDif_MCF),p_corr_walking_opencap[1],sim_power_MCF,stats_MCF_opencap['test_stat'],
        np.mean(ref_percDif_MCF),np.std(ref_percDif_MCF),p_corr_walking_mocap[1],ref_power_MCF,stats_MCF_mocap['test_stat']))
    
    print('\nstats_KAM_opencap:'); print(stats_KAM_opencap)
    print('\nstats_KAM_mocap:'); print(stats_KAM_mocap)
    print('\nstats_MCF_opencap:'); print(stats_MCF_opencap)
    print('\nstats_MCF_mocap:'); print(stats_MCF_mocap)

    
    # Mean power
    sim_meanPower = np.mean([sim_power_KAM,sim_power_MCF])
    ref_meanPower = np.mean([ref_power_KAM,ref_power_MCF])
    
    print('Mean power: OpenCap: {:.2f}; Mocap: {:.2f}.'.format(sim_meanPower,ref_meanPower))

        
# Accuracy of changes in KAM induced by trunk sway

#KAM magnitudes
plt.subplot(1,2,1)

yTrue = np.array(results_all['walking']['scalars']['ref_peakKAM_l_mean']  +
                  results_all['walkingTS']['scalars']['ref_peakKAM_l_mean'] ) #+
                  # results_all['walking']['scalars']['ref_peakKAM_r_mean']  +
                  # results_all['walkingTS']['scalars']['ref_peakKAM_r_mean'] )
yPred = np.array(results_all['walking']['scalars']['sim_peakKAM_l_mean'] +
                  results_all['walkingTS']['scalars']['sim_peakKAM_l_mean'] ) #+
                  # results_all['walking']['scalars']['sim_peakKAM_r_mean']  +
                  # results_all['walkingTS']['scalars']['sim_peakKAM_r_mean'] )
                  

plt.plot(yTrue,yPred,linestyle='none',marker='o',color='k',ms=2)

# y=x
xVals = np.array((0,10))
m, b = np.polyfit(yTrue,yPred, 1)
plt.plot(xVals,xVals,color=np.multiply([1,1,1],.8),linewidth=.5,ls='--')
plt.plot(xVals,m*xVals+b,color='k',linestyle='-',linewidth=1)
ax = plt.gca()

plt.ylabel('OpenCap peak 1\nKAM (%BW*ht)')
plt.xlabel('Mocap peak 1\nKAM (%BW*ht)')
plt.xlim([.5,3.5])
plt.ylim([.5,3.5])
ax.set_xticks(np.arange(1,4,1))
ax.set_yticks(np.arange(1,4,1))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_aspect('equal')
plt.text(0.04,.90,'a',weight='bold',transform=ax.transAxes,fontsize=10)

# acc = accuracy_score(yTrueBinary,yPredBinary)
mae = mean_absolute_error(yTrue,yPred)
r2 = calc_r2(yTrue,yPred)
plt.text(.45,.1,'MAE={:.2f}\n'.format(mae) + '$r^2$={:.2f}'.format(r2),transform=ax.transAxes)
plt.text(.07,0.02,'y=x',fontsize=SMALL_SIZE,color=np.multiply([1,1,1],.8),transform=ax.transAxes,rotation=45)

#KAM and MCF P1 changes
plt.subplot(1,2,2)

inData = [ref_percDif,sim_percDif,ref_percDif_MCF,sim_percDif_MCF]
inMean = [np.mean(d) for d in inData]
inStd = [np.std(d) for d in inData]
yErr = [inStd,np.zeros((len(inStd)))]

xPos = np.repeat(np.array((1,2)),2) + np.tile(np.array((-.17,.17)),2)
np.random.seed(0)
xScatter = [x + .2*(.5-np.random.random(10)) for x in xPos]
colors = [ltbl,bl]*4

barH = plt.bar(xPos,inMean,width=.25,yerr=yErr,error_kw={'linewidth':.5,'capthick':.5},capsize=1,capstyle='butt',
        color=colors,linewidth=.5,edgecolor='k')
plt.legend(barH,['Mocap', 'OpenCap'],frameon=False,
            loc=(.2,.75))

# plot individual datapoints
np.random.seed(2)
xRand = np.random.uniform(low=0, high=0,size=int(len(inData[0])/2))
xRand = np.concatenate((xRand,-xRand))
for x,y in zip(xPos,inData):
    plt.plot(x+xRand,y,'ko',mfc='none',markerSize=2.2,mew=.25)
plt.plot([-3,3],[0,0],color = 'k',lw=1)

plt.xlim([.5,2.5])
plt.ylim([-90,80])
plt.yticks(range(-80,81,40))
ax = plt.gca()
ax.set_aspect(.016)

plt.ylabel('Change in peak\nknee loading (%)')
plt.xticks([1,2], ['KAM','MCF'], rotation='horizontal')
ax.tick_params(axis='x', length=0)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)

dKamPredP1 = np.subtract(sim_KAM_cond2, sim_KAM_cond1)
dKamTrueP1 = np.subtract(ref_KAM_cond2, ref_KAM_cond1)
yTrueBinary = dKamTrueP1 < 0
yPredBinary = dKamPredP1 < 0

dMcfPredP1 = np.subtract(sim_MCF_cond2, sim_MCF_cond1)
dMcfTrueP1 = np.subtract(ref_MCF_cond2, ref_MCF_cond1)
yTrueBinary_MCF = dKamTrueP1 < 0
yPredBinary_MCF = dKamPredP1 < 0

# Add significance bars
for i in range(len(xPos)):
    yPos = inMean[i] + np.sign(inMean[i]) *(inStd[i] + 30)
    plt.text(xPos[i],yPos,'*',ha='center',va='bottom',fontweight='bold',fontsize=10)

acc = accuracy_score(yTrueBinary,yPredBinary)
acc_MCF = accuracy_score(yTrueBinary_MCF,yPredBinary_MCF)
r2 = calc_r2(yTrue,yPred)
plt.text(0.03,.9,'b',weight='bold',transform=ax.transAxes,fontsize=10)
print('Accuracy of P1KAM reduction classification = {:.2f}, P1MCF reduction classification = {:.2f}, nReduce KAM={:.0f}, MCF={:.0f}.'.format(
      acc,acc_MCF,np.sum(yTrueBinary),np.sum(yPredBinary)))

# Saving
if saveFigures:
    figPath = os.path.join(figDir,'WalkingUseCase')
    thisFigPath = os.path.join(figPath,'Walking.svg')
    os.makedirs(figPath,exist_ok=True)
    plt.savefig(thisFigPath, format='svg')


# %% Figure 4: Compare sagittal moment redistribution with different sit-to-stand kinematics

sim_KEM_cond1 = results_all['STS']['scalars']['sim_meanKEM_both_mean']
sim_KEM_cond2 = results_all['STSweakLegs']['scalars']['sim_meanKEM_both_mean']
sim_KEM_redux = np.mean(np.subtract(sim_KEM_cond2,sim_KEM_cond1))
sim_KEM_redux_sd = np.std(np.subtract(sim_KEM_cond2,sim_KEM_cond1))
stats_STS_KEM_opencap = ttestUtility(sim_KEM_cond1,sim_KEM_cond2,name='KEM OpenCap')

ref_KEM_cond1 = results_all['STS']['scalars']['ref_meanKEM_both_mean']
ref_KEM_cond2 = results_all['STSweakLegs']['scalars']['ref_meanKEM_both_mean']
ref_KEM_redux = np.mean(np.subtract(ref_KEM_cond2,ref_KEM_cond1))
ref_KEM_redux_sd = np.std(np.subtract(ref_KEM_cond2,ref_KEM_cond1))
stats_STS_KEM_mocap = ttestUtility(ref_KEM_cond1,ref_KEM_cond2,name='KEM Mocap')

# Post hoc power
sim_power_KEM = posthocPower(sim_KEM_cond1,sim_KEM_cond2)
ref_power_KEM = posthocPower(ref_KEM_cond1,ref_KEM_cond2)

sim_HEM_cond1 = results_all['STS']['scalars']['sim_meanHEM_both_mean']
sim_HEM_cond2 = results_all['STSweakLegs']['scalars']['sim_meanHEM_both_mean']
sim_HEM_redux = np.mean(np.subtract(sim_HEM_cond2,sim_HEM_cond1))
sim_HEM_redux_sd = np.std(np.subtract(sim_HEM_cond2,sim_HEM_cond1))
stats_STS_HEM_opencap = ttestUtility(sim_HEM_cond1,sim_HEM_cond2,name='HEM OpenCap')

ref_HEM_cond1 = results_all['STS']['scalars']['ref_meanHEM_both_mean']
ref_HEM_cond2 = results_all['STSweakLegs']['scalars']['ref_meanHEM_both_mean']
ref_HEM_redux = np.mean(np.subtract(ref_HEM_cond2,ref_HEM_cond1))
ref_HEM_redux_sd = np.std(np.subtract(ref_HEM_cond2,ref_HEM_cond1))
stats_STS_HEM_mocap = ttestUtility(ref_HEM_cond1,ref_HEM_cond2,name='HEM Mocap')

# Post hoc power
sim_power_HEM = posthocPower(sim_HEM_cond1,sim_HEM_cond2)
ref_power_HEM = posthocPower(ref_HEM_cond1,ref_HEM_cond2)

sim_APM_cond1 = results_all['STS']['scalars']['sim_meanAPM_both_mean']
sim_APM_cond2 = results_all['STSweakLegs']['scalars']['sim_meanAPM_both_mean']
sim_APM_redux = np.mean(np.subtract(sim_APM_cond2,sim_APM_cond1))
sim_APM_redux_sd = np.std(np.subtract(sim_APM_cond2,sim_APM_cond1))
stats_STS_APM_opencap = ttestUtility(sim_APM_cond1,sim_APM_cond2,name='APM OpenCap')

ref_APM_cond1 = results_all['STS']['scalars']['ref_meanAPM_both_mean']
ref_APM_cond2 = results_all['STSweakLegs']['scalars']['ref_meanAPM_both_mean']
ref_APM_redux = np.mean(np.subtract(ref_APM_cond2,ref_APM_cond1))
ref_APM_redux_sd = np.std(np.subtract(ref_APM_cond2,ref_APM_cond1))
stats_STS_APM_mocap = ttestUtility(ref_APM_cond1,ref_APM_cond2,name='APM Mocap')

# Post hoc power
sim_power_APM = posthocPower(sim_APM_cond1,sim_APM_cond2)
ref_power_APM = posthocPower(ref_APM_cond1,ref_APM_cond2)

# Mean power
sim_meanPower = np.mean([sim_power_KEM,sim_power_HEM,sim_power_APM])
ref_meanPower = np.mean([ref_power_KEM,ref_power_HEM,ref_power_APM])

# Correct for multiple comparisons usin Benjamini Hochberg
from statsmodels.stats.multitest import fdrcorrection
_,p_STS_opencap = fdrcorrection([stats_STS_KEM_opencap['p'],stats_STS_HEM_opencap['p'],stats_STS_APM_opencap['p']])
_,p_STS_mocap = fdrcorrection([stats_STS_KEM_mocap['p'],stats_STS_HEM_mocap['p'],stats_STS_APM_mocap['p']])


print('Dif in KEM between STS conditions. OpenCap: {:.2f}+-{:.2f}%BWht, p={:.3f}, power={:.2f}; Mocap: {:.2f}+-{:.2f}%BWht, p={:.3f}, power={:.2f}.'.format(
    sim_KEM_redux,sim_KEM_redux_sd,p_STS_opencap[0],sim_power_KEM,ref_KEM_redux,ref_KEM_redux_sd,p_STS_mocap[0],ref_power_KEM))

print('Dif in HEM between STS conditions. OpenCap: {:.2f}+-{:.2f}%BWht, p={:.3f}, power={:.2f}; Mocap: {:.2f}+-{:.2f}%BWht, p={:.3f}, power={:.2f}.'.format(
    sim_HEM_redux,sim_HEM_redux_sd,p_STS_opencap[1],sim_power_HEM,ref_HEM_redux,ref_HEM_redux_sd,p_STS_mocap[1],ref_power_HEM))

print('Dif in APM between STS conditions. OpenCap: {:.2f}+-{:.2f}%BWht, p={:.3f}, power={:.2f}; Mocap: {:.2f}+-{:.2f}%BWht, p={:.3f}, power={:.2f}.'.format(
    sim_APM_redux,sim_APM_redux_sd,p_STS_opencap[2],sim_power_APM,ref_APM_redux,ref_APM_redux_sd,p_STS_mocap[2],ref_power_APM))

print('Mean power: OpenCap: {:.2f}; Mocap: {:.2f}.'.format(sim_meanPower,ref_meanPower))

print('\nstats_KEM_opencap:'); print(stats_STS_KEM_opencap)
print('\nstats_KEM_mocap:'); print(stats_STS_KEM_mocap)
print('\nstats_HEM_opencap:'); print(stats_STS_HEM_opencap)
print('\nstats_HEM_mocap:'); print(stats_STS_HEM_mocap)
print('\nstats_APM_opencap:'); print(stats_STS_APM_opencap)
print('\nstats_APM_mocap:'); print(stats_STS_APM_mocap)


# Bar plot with dot plots nextdoor
SMALL_SIZE = 8
MEDIUM_SIZE = 9
BIGGER_SIZE = 10

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

plt.close(plt.figure(4))
fig = plt.figure(4,figsize=(5, 2.5),dpi=200)

plt.subplot(1,2,1)

inData = [np.subtract(ref_KEM_cond2,ref_KEM_cond1),np.subtract(sim_KEM_cond2,sim_KEM_cond1),
          np.subtract(ref_HEM_cond2,ref_HEM_cond1),np.subtract(sim_HEM_cond2,sim_HEM_cond1),
          np.subtract(ref_APM_cond2,ref_APM_cond1),np.subtract(sim_APM_cond2,sim_APM_cond1)]
inMean = [np.mean(d) for d in inData]
inStd = [np.std(d) for d in inData]
yErr =np.tile(np.array(inStd),2)

yErr = np.multiply(yErr,np.concatenate((np.less(inMean,0),np.greater(inMean,0))))
yErr = np.reshape(yErr,(2,int(len(yErr)/2)))

xPos = np.array((1,2,4,5,7,8))
gy = 0.3*np.ones((3))
rd = np.array((150, 68, 72)) / 255
bk = np.zeros((3))
colors = [ltbl,bl]*4


ax = plt.gca()
spineWidth =  ax.spines['left'].get_linewidth()
plt.plot([0,np.max(xPos)+1],[0,0],color='k',linewidth=spineWidth)
barH = plt.bar(xPos,inMean,width=.6,yerr=yErr,error_kw={'linewidth':.5,'capthick':.5},capsize=1,capstyle='butt',
        color=colors,linewidth=.5,edgecolor='k')

# plot individual datapoints
np.random.seed(2)
xRand = np.random.uniform(low=0, high=0,size=int(len(inData[0])/2))
xRand = np.concatenate((xRand,-xRand))
for x,y in zip(xPos,inData):
    plt.plot(x+xRand,y,'ko',mfc='none',markerSize=2.2,mew=.25)
  

plt.legend(barH,['Mocap', 'OpenCap'],frameon=False,fontsize='small',
           loc=(.5,.2))

ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.ylabel('Change in sagittal-plane\n moment (%BW*ht)')
plt.ylim([-2.1,2.9])
plt.xlim([0,9])
ax.set_xticks(np.array([1.5,4.5,7.5]))
ax.tick_params(axis='x', which='major', length=0)
ax.set_xticklabels(['Knee','Hip','Ankle'])
plt.text(0.03,.93,'a',weight='bold',transform=ax.transAxes,fontsize=BIGGER_SIZE)

# Add significance bars
for i in range(6):
    meanSign = np.divide(inMean,np.abs(inMean))
    yPos = np.add(inMean,np.multiply(meanSign,inStd))[i] + .1*meanSign[i]
    plt.text(xPos[i],yPos,'*',ha='center',va='center',fontweight='bold',fontsize=10)

motTypes = ['STS','STSweakLegs']
colors = [.3*np.array([1, 1, 1]),
          np.array([88, 122, 191])/255]
xName = 'ref_meanKEM_both_mean'
yName = 'sim_meanKEM_both_mean'

plt.subplot(1,2,2)

y_true = ref_KEM_cond1 + ref_KEM_cond2
y_pred = sim_KEM_cond1 + sim_KEM_cond2

plt.plot(y_true,y_pred,marker='o',mfc=bk,mec='none',linestyle='none',ms=3)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.ylabel('OpenCap' + ' knee extension\n moment (%BW*ht)')
plt.xlabel('Mocap' + ' knee extension\n moment (%BW*ht)')

ylim = ax.get_ylim()
xlim = ax.get_xlim()
minAx = np.min([ylim,xlim])
maxAx = np.max([ylim,xlim])

# y=x
xVals = np.array((minAx,maxAx))
m, b = np.polyfit(y_true,y_pred, 1)
plt.plot(xVals,xVals,color=np.multiply([1,1,1],.8),linewidth=.5,ls='--')
plt.plot(xVals,m*xVals+b,color='k',linestyle='-',linewidth=1)

ax.set_aspect('equal')
plt.ylim([.8,4.1])
plt.xlim([.8,4.1])
ax.set_xticks(np.arange(1,5))
ax.set_xticks(np.arange(1,5))
plt.legend(frameon=False)

r2 = calc_r2(y_true,y_pred)
mae = mean_absolute_error(y_true,y_pred)
plt.text(.1,.65,'$r^2$={:.2f} \nMAE={:.2f}'.format(r2,mae),transform=ax.transAxes)
plt.text(0.03,.93,'b',weight='bold',transform=ax.transAxes,fontsize=BIGGER_SIZE)
plt.text(.8,.87,'y=x',fontsize=SMALL_SIZE,color=np.multiply([1,1,1],.8),transform=ax.transAxes,rotation=45)


fig.tight_layout()

if saveFigures:
    figPath = os.path.join(figDir,'STS_UseCase')
    os.makedirs(figPath,exist_ok=True)
    thisFigPath = os.path.join(figPath,'STS.svg')
    plt.savefig(thisFigPath, format='svg')

# Evaluate changes in KEM to compare to literature changes in strength. Larrson
# 1976 found loss of isokinetic strength y = -.93+169.5 Nm

# calculate %BW*ht multiplier to put moments back in Nm
BWht_mult = np.divide(results_all['STS']['GRMs_BWht']['ref'][1,52,:],
          results_all['STS']['GRMs']['ref'][1,52,:])
KEM_BWht_MAE = np.mean(np.abs(np.subtract(y_pred,y_true)))
KEM_Nm_MAE = np.mean(np.abs(np.divide(
    np.subtract(y_pred,y_true),np.tile(BWht_mult,2))))

# From Larrson regression
KEM_MAE_in_years = KEM_Nm_MAE/0.93
print('MAE in STS KEM ({:.2}%BW*ht ({:.2}Nm)) is equivalent to {:.2} years loss in isokinetic strength (Larsson 1996).'.format(
    KEM_BWht_MAE,KEM_Nm_MAE,KEM_MAE_in_years))

# %% Figure 5: Squats Vasti activations + classification 

SMALL_SIZE = 8
MEDIUM_SIZE = 9
BIGGER_SIZE = 10

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

rd = np.array((150, 68, 72)) / 255
bl = np.array([88, 122, 191])/255
gy = 0.3*np.ones(3)
bk = np.zeros(3)

mcColor = bk
ocColor = bl


if activities['squats']:
    dType = 'activations'
    colors = [[gy,rd],
              [gy,rd]
              ]
    alphas = [[1,1],
              [1,1]
              ]
    lineStyles = [['-','--'],
              ['-','--']
              ]
    gs = gridspec.GridSpec(1, 3, width_ratios=[.5,.5, 1]) 

    
    idx = results_all['squats'][dType]['headers'].index('vasmed_l')
    idx2 = results_all['squats'][dType]['headers'].index('vaslat_l')
    
    acts = ['squats','squatsAsym']
    dSources = ['ref','sim']
    squatVasti = {}
    for act in acts:
        squatVasti[act]= {}
        for dSource in dSources:
            squatVasti[act][dSource+'_mean'] = np.mean(np.divide(np.mean(results_all[act][dType][dSource][[idx,idx2],:,:],axis=0),
                                               np.max(np.stack((np.max(np.mean(results_all[acts[0]][dType][dSource][[idx,idx2],:,:],axis=0),axis=0),
                                                       np.mean(np.mean(results_all[acts[0]][dType][dSource][[idx,idx2],:,:],axis=0),axis=0))),axis=0)),axis=1)
            squatVasti[act][dSource+'_std'] = np.std(np.divide(np.mean(results_all[act][dType][dSource][[idx,idx2],:,:],axis=0),
                                               np.max(np.stack((np.max(np.mean(results_all[acts[0]][dType][dSource][[idx,idx2],:,:],axis=0),axis=0),
                                                       np.mean(np.mean(results_all[acts[0]][dType][dSource][[idx,idx2],:,:],axis=0),axis=0))),axis=0)),axis=1)
    
    meanList = [
                [squatVasti['squats']['ref_mean'],squatVasti['squatsAsym']['ref_mean']],
                [squatVasti['squats']['sim_mean'],squatVasti['squatsAsym']['sim_mean']]
                ]
    sdList = [
              [squatVasti['squats']['ref_std'],squatVasti['squatsAsym']['ref_std']],
              [squatVasti['squats']['sim_std'],squatVasti['squatsAsym']['sim_std']]
              ]

    yLabelList = ['Normalized vasti activation',None]
    xLabelList = ['% Squat','% Squat']
    legList = [['symmetric','asymmetric'],[None,None]]
    titleList = ['Measured (EMG)','OpenCap']
    yLim = [0, 1.4]
    labelLetters = ['a','b']
    
    #copied from ensemblePlot
    subplotNums = [1,2]
    nCurves = len(meanList[0])
    
    try :
        plt.close(plt.figure(5))
    except:
        _=None
    figH = plt.figure(5,figsize=(6,2.3),dpi=200)
    axH = []
    for iSub,subNum in enumerate(subplotNums):
    
        
        plt.subplot(gs[iSub])
        plt.plot([0,100],[0,0],color = [.5,.5,.5],linewidth = .5)
        
        for iC in range(nCurves):
            ax = plt.gca()   
            ax.fill_between(range(101),meanList[iSub][iC] + sdList[iSub][iC],
                meanList[iSub][iC] - sdList[iSub][iC],facecolor=colors[iSub][iC], alpha=alphas[iSub][iC]*.3)
            plt.plot(range(101),meanList[iSub][iC],color = colors[iSub][iC],label=legList[iSub][iC],
                     alpha=alphas[iSub][iC],linestyle=lineStyles[iSub][iC])
        
        plt.xlabel(xLabelList[iSub])
        plt.ylabel(yLabelList[iSub])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.text(0.03,.93,labelLetters[iSub],weight='bold',transform=ax.transAxes,fontsize=BIGGER_SIZE)
        if legList[iSub][0] != None:
            plt.legend(bbox_to_anchor=(1.1, 1), bbox_transform=ax.transAxes,frameon=False)
        if titleList[iSub] != None:
            plt.title(titleList[iSub])
        ax.set_xlim(0,100)
        if yLim != None:
            ax.set_ylim(yLim)
        axH.append(ax)
        
    plt.setp(axH[1],yticklabels=[])    
    for ax in axH:
        plt.setp(ax,xticks=np.arange(0,150,50))
        lineH = ax.get_lines()[-1]
        # plt.setp(lineH,linestyle='-')
    
    # ROC curve
    
    # Symmetry definition - based on Salem 2003, knee extension moment
    # asymmetry in individuals 30 wk post ACLR. Here we create
    # truth labels based on EMG and this literature value.
    asymThresh = calc_LSI(.85,1) # Salem 2003

    y_pred_cont = np.array(results_all['squatsAsym']['scalars']['sim_peakVLVMAct_Asym_mean'] +
            results_all['squats']['scalars']['sim_peakVLVMAct_Asym_mean'])
    y_pred_cont_so = np.array(results_all['squatsAsym']['scalars']['so_peakVLVMAct_Asym_mean'] +
            results_all['squats']['scalars']['sim_peakVLVMAct_Asym_mean'])
    y_true_cont = np.array(results_all['squatsAsym']['scalars']['ref_peakVLVMAct_Asym_mean'] +
            results_all['squats']['scalars']['ref_peakVLVMAct_Asym_mean'])
    y_class_litThresh = y_true_cont > asymThresh
    
    # AUC and ROC
    fpr,tpr,thresh = roc_curve(y_class_litThresh,y_pred_cont)
    aucVal = roc_auc_score(y_class_litThresh, y_pred_cont)
    fpr_so,tpr_so,thresh_so = roc_curve(y_class_litThresh,y_pred_cont_so)
    aucVal_so = roc_auc_score(y_class_litThresh, y_pred_cont_so)
    idxOpt = np.argmax(tpr-fpr)
    threshOpt = thresh[idxOpt]
    idxOpt_so = np.argmax(tpr_so-fpr_so)
    threshOpt_so = thresh[idxOpt_so]
    
    # Accuracy of asymmetry predictions
    acc_pred = np.sum(~np.logical_xor(y_pred_cont>threshOpt, y_class_litThresh))/len(y_pred_cont)
    acc_pred_so = np.sum(~np.logical_xor(y_pred_cont_so>threshOpt_so, y_class_litThresh))/len(y_pred_cont_so)
    
    # Accuracy of change predictions
    improved = y_true_cont[10:20]-y_true_cont[0:10] <0
    improved_pred = y_pred_cont[10:20]-y_pred_cont[0:10] <0
    improved_pred_so = y_pred_cont_so[10:20]-y_pred_cont_so[0:10] <0
    
    acc_chgPred = np.sum(~np.logical_xor(improved,improved_pred))/len(improved)
    acc_chgPred_so = np.sum(~np.logical_xor(improved,improved_pred_so))/len(improved)
    
    print('At opt threshold of {:.3f}, accuracy of symmetry detection was {:.3f} for OpenCap, and {:.3f} for static opt.'.format(
        threshOpt,acc_pred,acc_pred_so))
    
    print('Accuracy of change in symmetry was {:.2f} for OpenCap, and {:.2f} for static opt.'.format(
        acc_chgPred, acc_chgPred_so))
        
    #Plot
    ax = plt.subplot(gs[2])
    ax.plot([0,1],[0,1],linestyle='--',color=0.5*np.ones(3),linewidth=0.75)
    ax.plot(fpr,tpr,linewidth=2.5,color = ocColor,label='video (' + 'AUC={:.2f})'.format(aucVal))
    ax.plot(fpr_so,tpr_so,linewidth=1,color=mcColor,label='mocap-based sim. (' + 'AUC={:.2f})'.format(aucVal_so))
    ax.set_ylabel('True positive rate')
    ax.set_xlabel('False positive rate')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.legend(frameon=False,loc='lower right')
    ax.set_aspect('equal')
    ax.set_title('Vasti activation asymmetry')
    plt.yticks(np.arange(0,1.05,.5))
    plt.xticks(np.arange(0,1.05,.5))
    plt.xlim([-.03,1.1])
    plt.ylim([-.03,1.1])
    plt.text(1.08,.13,'OpenCap AUC={:.2f}'.format(aucVal),color = ocColor,ha='right')
    plt.text(1.08,0.02,'Mocap AUC={:.2f}'.format(aucVal_so),color = mcColor,ha='right')
    plt.text(0.03,.93,'c',weight='bold',transform=ax.transAxes,fontsize=BIGGER_SIZE)


    figH.tight_layout()
    plt.subplots_adjust(wspace=0.2)
    
    if saveFigures:
        figFolder = os.path.join(figDir,'SquatUseCase')
        os.makedirs(figFolder,exist_ok=True)
        thisFigPath = os.path.join(figDir,'SquatUseCase','squatEMG.svg')
        plt.savefig(thisFigPath, format='svg')


# %% Figure 6: Field study. Can we detect asymmetries in squats and change between conditions?
if fieldStudy:
    try: 
        plt.close(plt.figure(202))
    except:
        _
    
    fig = plt.figure(202,figsize=(4.5,4.5), dpi=200)
    
    
    SMALL_SIZE = 8
    MEDIUM_SIZE = 9
    BIGGER_SIZE = 10
    
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    rd = np.array((150, 68, 72)) / 255
    bl = np.array([88, 122, 191])/255   
    
    nSubs = results_all_fieldStudy['squats']['torques_BWht']['sim'].shape[2]
    ySquats = [max(i,0) for i in results_all_fieldStudy['squats']['scalars']['sim_peakKEM_Asym_mean']]
    ySquatsAsym = [max(i,0) for i in results_all_fieldStudy['squatsAsym']['scalars']['sim_peakKEM_Asym_mean']]
    
    # histogram of groups
    ax=plt.subplot(2,2,1)
    hist,edges,_ = plt.hist(ySquats,bins=np.arange(-1.9,2.9,.2),color=gy,alpha=.7)
    hist_asym,edges_asym,_ = plt.hist(ySquatsAsym,bins=np.arange(-1.9,2.9,.2),color=rd,alpha=.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('Frequency (n={})'.format(nSubs))
    ax.set_xlabel('Knee extension moment\nsymmetry index')
    plt.xlim(-.2,2.4)
    plt.ylim(0,41)
    plt.text(.68,8,'sym.\nsquats',ha='right',color=gy) # left 1.25,23; right .77,8.5
    plt.text(2.35,8,'asym.\nsquats',ha='left',color=rd)
    plt.text(.75,20,'symmetric',ha='left',va='bottom',color='k',rotation=0)
    plt.text(0.03,.93,'a',weight='bold',transform=ax.transAxes,fontsize=9)
    plt.plot([1,1],[0,100],'k--',linewidth=.75)
    
    # ROC
    ax=plt.subplot(2,2,2)
    # For mean KEM
    y_pred_cont = np.array(ySquats + ySquatsAsym)
    y_class = np.concatenate((np.zeros((nSubs)),np.ones((nSubs))))
    
    # AUC and ROC
    fpr,tpr,thresh = roc_curve(y_class,y_pred_cont)
    aucVal = roc_auc_score(y_class, y_pred_cont)
    
    #Plot
    ax.plot(fpr,tpr,linewidth=1.5,color = 'k')
    ax.set_ylabel('True positive rate')
    ax.set_xlabel('False positive rate')
    ax.plot([0,1],[0,1],linestyle='--',color=0.5*np.ones(3),linewidth=0.75)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_aspect('equal')
    # ax.set_title('Knee ext. moment asym.')
    plt.text(.15,.65,'AUC={:.2f}'.format(aucVal))
    plt.ylim(-0.02,1.02)
    plt.xlim(-0.02,1.02)
    plt.xticks(np.arange(0,1.1,.5))
    plt.yticks(np.arange(0,1.1,.5))
    plt.text(0.03,.93,'b',weight='bold',transform=ax.transAxes,fontsize=9)
    
    idxOpt = np.argmax(tpr-fpr)
    sensOpt = tpr[idxOpt]
    specOpt = 1-fpr[idxOpt]
    threshOpt = thresh[idxOpt]
    accOpt = sum(~np.logical_xor(y_pred_cont>threshOpt, y_class))/len(y_class)
    print('Optimal thresh={:.3f}, sensitivity={:.3f}, specificity={:.3f}, accuracy={:.3f}.'.format(threshOpt,sensOpt,specOpt,accOpt))
    
    # histogram of change
    ax=plt.subplot(2,2,3)
    # dKEM = np.subtract(ySquatsAsym,ySquats) # subtracting the averages
    symms = results_all_fieldStudy['squats']['scalars']['sim_peakKEM_Asym']
    asyms = results_all_fieldStudy['squatsAsym']['scalars']['sim_peakKEM_Asym']
    
    dKEM = [max(-.5,a) for a in [np.mean([[a-s for a in asym] for s in symm]) for symm,asym in zip(symms,asyms)]]
    dKEM_baseline = [max(-.5,a) for a in results_all_fieldStudy['squatsAsym']['scalars']['sim_peakKEM_dAsym_baseline_mean']]
    
    hist_baseline,edges_baseline,_ = plt.hist(dKEM_baseline,bins=np.arange(-3.9,3.9,.2),color=gy,alpha=.7)
    hist,edges,_ = plt.hist(dKEM,bins=np.arange(-3.9,3.9,.2),color=rd,alpha=.7)
    ax.set_ylabel('True positive rate')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('Frequency (n={})'.format(nSubs))
    ax.set_xlabel('Change in knee extension\nmoment symmetry index')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlim(-.7,2.1)
    plt.ylim(0,82)
    ax.set_xticks(np.arange(0,3,1))
                
    plt.plot([0,0],[0,100],'k--',linewidth=.75)
    
    yPos = 71.5
    xPos = 2.23
    ax.annotate('more symmetric', xy=(xPos+.3,yPos),  xycoords='data',
                xytext=(xPos, yPos), textcoords='data',
                arrowprops=dict(facecolor='black', shrink=0.05,width=.1,headwidth=3,headlength=3),
                ha='right', va='center')
    plt.text(0.03,.93,'c',weight='bold',transform=ax.transAxes,fontsize=9)
    plt.text(0.2,35,'asym. vs.\nasym. squats',ha='left',color=gy) # left 1.25,23; right .77,8.5
    plt.text(1.18,10,'asym. vs.\nsym. squats',ha='left',color=rd)
    
    
    # AUC and ROC
    ax=plt.subplot(2,2,4)
    y_pred_cont = np.concatenate((dKEM_baseline,dKEM))
    y_class = np.concatenate((np.zeros((nSubs)),np.ones((nSubs))))
    
    fpr,tpr,thresh = roc_curve(y_class,y_pred_cont)
    aucVal = roc_auc_score(y_class, y_pred_cont)
    
    #Plot
    ax.plot(fpr,tpr,linewidth=1.5,color = 'k')
    ax.set_ylabel('True positive rate')
    ax.set_xlabel('False positive rate')
    ax.plot([0,1],[0,1],linestyle='--',color=0.5*np.ones(3),linewidth=0.75)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_aspect('equal')
    # ax.set_title('Knee ext. moment asym.')
    plt.text(.15,.65,'AUC={:.2f}'.format(aucVal))
    plt.ylim(-.02,1.02)
    plt.xlim(-.02,1.02)
    plt.xticks(np.arange(0,1.1,.5))
    plt.yticks(np.arange(0,1.1,.5))
    plt.text(0.03,.93,'d',weight='bold',transform=ax.transAxes,fontsize=9)
    
    idxOpt = np.argmax(tpr-fpr)
    sensOpt = tpr[idxOpt]
    specOpt = 1-fpr[idxOpt]
    threshOpt = thresh[idxOpt]
    accOpt = sum(~np.logical_xor(y_pred_cont>threshOpt, y_class))/len(y_class)
    print('Optimal thresh={:.3f}, sensitivity={:.3f}, specificity={:.3f}, accuracy={:.3f}.'.format(threshOpt,sensOpt,specOpt,accOpt))
    
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.7)    

    if saveFigures:
        figFolder = os.path.join(figDir,'fieldStudy')
        os.makedirs(figFolder,exist_ok=True)
        
        thisFigPath = os.path.join(figFolder,'fieldStudy.jpg')
        plt.savefig(thisFigPath, format='jpg',dpi=200)
    
        thisFigPath = os.path.join(figDir,'fieldStudy','fieldStudy.svg')
        plt.savefig(thisFigPath, format='svg')