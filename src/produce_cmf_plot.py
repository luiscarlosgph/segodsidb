#!/usr/bin/env python3
#
# @brief   Script to produce HSI to XYZ conversion colour matching function.
#
# @author  Luis C. Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
# @date    23 Aug 2022.

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# My imports
import torchseg.data_loader as dl 

def main():
    plt.rc('font', size=12)


    # Produce CMF function 
    cmf = 'cie_2_1931'
    min_wl = 380
    max_wl = 780
    steps = max_wl - min_wl + 1
    wl = new_wl=np.linspace(min_wl, max_wl, steps)
    f_xbar, f_ybar, f_zbar = \
        dl.OdsiDbDataLoader.LoadImage.get_corrected_cmf(cmf, wl)

    # Plot CMF
    xbar = f_xbar(wl) 
    ybar = f_ybar(wl)
    zbar = f_zbar(wl)
    cmf_plot = sns.lineplot(x=wl, y=xbar, color='red', label='X')
    cmf_plot = sns.lineplot(x=wl, y=ybar, color='green', label='Y')
    cmf_plot = sns.lineplot(x=wl, y=zbar, color='blue', label='Z')
    
    # Compute the integral of the CMFs
    xbar_integral = np.trapz(xbar, wl)
    ybar_integral = np.trapz(ybar, wl)
    zbar_integral = np.trapz(zbar, wl)
    print('Original CMF xbar integral:', xbar_integral)
    print('Original CMF ybar integral:', ybar_integral)
    print('Original CMF zbar integral:', zbar_integral)
    print()

    # Save CMF plot to disk
    plt.title('CIE XYZ standard observer CMFs')
    plt.xlim([380, 780])
    plt.ylim([0., 4.])
    plt.xticks([x for x in range(380, 790, 10)], ['' if x not in [y for y in range(380, 800, 40)] else x for x in range(380, 790, 10)])
    plt.legend()
    plt.xlabel('Wavelength (nm)')
    fig = cmf_plot.get_figure()
    fig.savefig('cie_2_1931.svg', format='svg', dpi=600)

    # Clear plot
    plt.clf()
    
    # Produce modified CMF function
    min_wl = 400
    max_wl = 780
    steps = max_wl - min_wl + 1
    wl = new_wl=np.linspace(min_wl, max_wl, steps)
    f_xbar, f_ybar, f_zbar = \
        dl.OdsiDbDataLoader.LoadImage.get_corrected_cmf(cmf, wl)

    # Plot modified CMF
    xbar = f_xbar(wl) 
    ybar = f_ybar(wl)
    zbar = f_zbar(wl)
    cmf_plot = sns.lineplot(x=wl, y=xbar, color='red', label='X')
    cmf_plot = sns.lineplot(x=wl, y=ybar, color='green', label='Y')
    cmf_plot = sns.lineplot(x=wl, y=zbar, color='blue', label='Z')

    # Compute the integral of the CMFs
    xbar_integral = np.trapz(xbar, wl)
    ybar_integral = np.trapz(ybar, wl)
    zbar_integral = np.trapz(zbar, wl)
    print('Specim IQ CMF xbar integral:', xbar_integral)
    print('Specim IQ CMF ybar integral:', ybar_integral)
    print('Specim IQ CMF zbar integral:', zbar_integral)
    print()

    # Save modified CMF to disk
    plt.title('Modified CIE XYZ CMFs for Specim IQ')
    plt.xlim([380, 780])
    plt.ylim([0., 4.])
    plt.xticks([x for x in range(380, 790, 10)], ['' if x not in [y for y in range(380, 800, 40)] else x for x in range(380, 790, 10)])
    plt.legend()
    plt.xlabel('Wavelength (nm)')
    fig = cmf_plot.get_figure()
    fig.savefig('cie_2_1931_specim_iq.svg', format='svg', dpi=600)

    # Clear plot
    plt.clf()

    # Produce modified CMF function
    min_wl = 450
    max_wl = 780
    steps = max_wl - min_wl + 1
    wl = new_wl=np.linspace(min_wl, max_wl, steps)
    f_xbar, f_ybar, f_zbar = \
        dl.OdsiDbDataLoader.LoadImage.get_corrected_cmf(cmf, wl)

    # Plot modified CMF
    xbar = f_xbar(wl) 
    ybar = f_ybar(wl)
    zbar = f_zbar(wl)
    cmf_plot = sns.lineplot(x=wl, y=xbar, color='red', label='X')
    cmf_plot = sns.lineplot(x=wl, y=ybar, color='green', label='Y')
    cmf_plot = sns.lineplot(x=wl, y=zbar, color='blue', label='Z')
    
    # Compute the integral of the CMFs
    xbar_integral = np.trapz(xbar, wl)
    ybar_integral = np.trapz(ybar, wl)
    zbar_integral = np.trapz(zbar, wl)
    print('Nuance EX CMF xbar integral:', xbar_integral)
    print('Nuance EX CMF ybar integral:', ybar_integral)
    print('Nuance EX CMF zbar integral:', zbar_integral)
    print()

    # Save modified CMF to disk
    plt.title('Modified CIE XYZ CMFs for Nuance EX')
    plt.xlim([380, 780])
    plt.ylim([0., 4.])
    plt.xticks([x for x in range(380, 790, 10)], ['' if x not in [y for y in range(380, 800, 40)] else x for x in range(380, 790, 10)])
    plt.legend()
    plt.xlabel('Wavelength (nm)')
    fig = cmf_plot.get_figure()
    fig.savefig('cie_2_1931_nuance_ex.svg', format='svg', dpi=600)


if __name__ == '__main__':
    main()
