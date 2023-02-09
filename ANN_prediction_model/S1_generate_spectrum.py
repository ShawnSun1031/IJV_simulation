#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 19:12:08 2022

@author: md703
"""

import numpy as np
import os
import random
import json
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm  as cm



# function
def calculateMus(wl, tissue, a, b):
    musp = a * (wl/muspBaseWl) ** (-b)
    if tissue == "blood":
        mus = musp/(1-0.95)
    else:
        mus = musp/(1-0.9)
    return mus

def plot_used_spectrum(tissue, spec, mua_or_mus):
    spec_numpy = pd.DataFrame(spec).to_numpy()
    for idx, i in enumerate(range(spec_numpy.shape[0])):
        if (idx%2) == 0:
            plt.plot(used_wl,spec_numpy[i], 'b')
        elif (idx%2) == 1:
            plt.plot(used_wl,spec_numpy[i], 'r')
    plt.xlabel('wavelength(nm)')
    if mua_or_mus == "mus":
        plt.ylabel("$\mu_s$($mm^{-1}$)")
        plt.title(f'{tissue} $\mu_s$ spectrum')
    elif mua_or_mus == "mua":
        plt.ylabel("$\mu_a$($mm^{-1}$)")
        plt.title(f'{tissue} $\mu_a$ spectrum')
    plt.legend(['testing','training'])
    
    plt.savefig(os.path.join("pic", f'{tissue}_{mua_or_mus}_spectrum.png'))
    plt.show()

def random_gen_mus(num, used_wl, tissue, a_max, a_min, b_max, b_min):
    a_list = list(np.linspace(a_max,a_min,num))
    # random.shuffle(a_list)
    b_list = list(np.linspace(b_max,b_min,num))
    # random.shuffle(b_list)
    spec = {}
    for wl in used_wl:
        spec[f'{wl}nm'] = []
    for i in range(num):
        for wl in used_wl:
            spec[f'{wl}nm'].append(calculateMus(wl, tissue, a_list[i], b_list[i]))
    
    return spec

def random_gen_mua(num, used_wl, tissue, mua_bound):
    spec = {}
    for wl in used_wl:
        spec[f'{wl}nm'] = []
    for wl in used_wl:
        [mua_max, mua_min] = mua_bound[f'{wl}nm'][tissue]
        mua = list(np.linspace(mua_max,mua_min,num))
        # random.shuffle(mua)
        for i in range(num):
            spec[f'{wl}nm'].append(mua[i])
    
    return spec 

def random_gen_ijv_mua(num, used_wl, tissue, mua_bound, blc, bloodConc, SO2):
    spec = {}
    for wl in used_wl:
        spec[f'{wl}nm'] = []
    for wl in used_wl:
        mua_min = mua_bound[f'{wl}nm'][f"ijv_bloodConc_{bloodConc[1]}_bloodSO2_{SO2:.2f}"][0]
        mua_max = mua_bound[f'{wl}nm'][f"ijv_bloodConc_{bloodConc[0]}_bloodSO2_{SO2:.2f}"][0]
        mua_new = mua_min + (blc-bloodConc[1])*(mua_max-mua_min)/(bloodConc[0]-bloodConc[1])
        spec[f'{wl}nm'].append(mua_new)

    return spec 

def random_gen_muscle_mua(num, used_wl, tissue, muscle_wl, blc, SO2, hbO2, hb, water):
    spec = {}
    for wl in used_wl:
        spec[f'{wl}nm'] = []
    for wl in used_wl:
        wl_idx = np.where(muscle_wl==wl)[0]
        # alpha = 1%
        spec[f'{wl}nm'] = list((2.303*0.12*blc*(SO2*hbO2[wl_idx]+(1-SO2)*hb[wl_idx])/64500) + (0.7*water[wl_idx]))
    
    return spec
    
if __name__ == "__main__":
    with open("mus_ab_bound.json","r") as f:
        mus_ab_bound = json.load(f)

    with open("mua_wl_bound.json","r") as f:
        mua_bound = json.load(f)

    # mus = [skin_mus, fat_mus, muscle_mus, blood_mus, blood_mus]
    # mua = [skin_mua, fat_mua, muscle_mua, ijv_mua, cca_mua]
    
    # used_wl = [730,760,790,810,850]
    # mus_gen
    used_wl = list(np.rint(np.linspace(700, 900, 20)).astype(int))
    muspBaseWl = 800 #nm
    num = 20 # number of spectrum used
    tissue = ['skin', 'fat', 'muscle', 'blood']
    mus_gen = {}
    for t in tissue:
        mus_spec = random_gen_mus(num, used_wl, t,  mus_ab_bound[f'{t}']['max'][0], mus_ab_bound[f'{t}']['min'][0], 
                                    mus_ab_bound[f'{t}']['max'][1], mus_ab_bound[f'{t}']['min'][1])
        mus_gen[t] = mus_spec
        plot_used_spectrum(t, mus_spec, "mus")
    
    # mua_gen --> for skin fat cca
    tissue = ['skin', 'fat', 'cca', 'muscle']
    mua_gen = {}
    for t in tissue:
        mua_spec = random_gen_mua(num, used_wl, t, mua_bound)
        mua_gen[t] = mua_spec
        plot_used_spectrum(t, mua_spec, "mua")
        
 

    # mua_gen --> for ijv     
    bloodConc = [174,138]
    # SO2 = [0.90, 0.80, 0.70, 0.60, 0.50, 0.40]
    SO2 = [i/100 for i in range(40,91,1)]
    tissue = 'ijv'
    ijv_gen = {}
    blc = list(np.linspace(bloodConc[0],bloodConc[1],num).astype(int))
    # blc = [150]
    random.shuffle(blc)
    for t in tissue:
        for b in blc:
            for idx, s in enumerate(SO2):
                ijv_spec = random_gen_ijv_mua(num, used_wl, t, mua_bound, b, bloodConc, s)
                ijv_gen[f"ijv_bloodConc_{b}_bloodSO2_{s}"] = ijv_spec
                spec_numpy = pd.DataFrame(ijv_spec).to_numpy()
        
    plt.figure(figsize=(28,14))
    for idx, b in enumerate(blc):
        for k in ijv_gen.keys():   
            if k.find((str(b))) != -1:
                spec_numpy = pd.DataFrame(ijv_gen[k]).to_numpy()
                if (idx%2) == 0:
                    plt.plot(used_wl,spec_numpy.reshape(-1), 'b', label=f'{k}')
                else:
                    plt.plot(used_wl,spec_numpy.reshape(-1), 'r', label=f'{k}')
    plt.xlabel('wavelength(nm)')
    plt.ylabel("$\mu_a$($mm^{-1}$)")
    plt.title(f'{tissue} $\mu_a$ spectrum')
    # plt.legend(ncol=3,loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join("pic", "ijv_mua.png"))
    plt.show()   
    
    
    # # mua_gen --> for muscle from excel
    # tissue = 'muscle'
    # df = pd.read_excel(os.path.join("absorption", "absorption.xlsx"),sheet_name="muscle_cal" )
    # muscle_wl = df['wl'].values
    # water = df['water'].values
    # hbO2 = df['hbo2'].values
    # hb = df['hb'].values
    # muscle_gen = {}
    # SO2 = [1.0,0.9,0.8]
    # for b in blc:
    #     for idx, s in enumerate(SO2):
    #         muscle_spec = random_gen_muscle_mua(num, used_wl, tissue, muscle_wl, b, s, hbO2, hb, water)
    #         muscle_gen[f"muscle_bloodConc_{b}_bloodSO2_{s}"] = muscle_spec
    #         spec_numpy = pd.DataFrame(ijv_spec).to_numpy()
 
    # plt.figure(figsize=(28,14))
    # for idx, b in enumerate(blc):
    #     for k in muscle_gen.keys():   
    #         if k.find((str(b))) != -1:
    #             spec_numpy = pd.DataFrame(muscle_gen[k]).to_numpy()
    #             plt.plot(used_wl,spec_numpy.reshape(-1),label=f'{k}', color=cm.rainbow(idx/len(blc)))
    # plt.xlabel('wavelength(nm)')
    # plt.ylabel("$\mu_a$($mm^{-1}$)")
    # plt.title(f'{tissue} spectrum')
    # plt.legend(ncol=3,loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.tight_layout()
    # plt.savefig(os.path.join("pic", "muscle_mua.png"))
    # plt.show() 
    
    # mua_gen.update(muscle_gen)
    mua_gen.update(ijv_gen)
    with open("mus_spectrum.json", "w") as f:
        json.dump(mus_gen, f, indent=4)
    with open("mua_spectrum.json", "w") as f:
        json.dump(mua_gen, f, indent=4)