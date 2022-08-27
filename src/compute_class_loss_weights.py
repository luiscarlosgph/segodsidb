"""
@brief   Compute the weights for each class of the ODSI-DB dataset based on the
         number of pixels of each class.

@author  Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date    21 Jun 2022.
"""

# My imports
import torchseg.data_loader as dl

pixels_per_class = {
    u'Skin':                18993686,
    u'Out of focus area':   8454491,
    u'Oral mucosa':         7917543,
    u'Enamel':              4913805,
    u'Tongue':              4081689,
    u'Lip':                 3920600,
    u'Hard palate':         2375998,
    u'Specular reflection': 1931845,
    u'Attached gingiva':    1922545,
    u'Soft palate':         1398594,
    u'Hair':                1383970,
    u'Marginal gingiva':    804393,
    u'Prosthetics':         755554,
    u'Shadow/Noise':        732209,
    u'Plastic':             255017,
    u'Metal':               196682,
    u'Gingivitis':          161874,
    u'Attrition/Erosion':   100919,
    u'Inflammation':        81098,
    u'Pigmentation':        43144,
    u'Calculus':            28615,
    u'Initial caries':      22008,
    u'Stain':               19428,
    u'Fluorosis':           17872,
    u'Microfracture':       14759,
    u'Root':                13962,
    u'Plaque':              10024,
    u'Dentine caries':      6616,
    u'Ulcer':               5552,
    u'Leukoplakia':         4623,
    u'Blood vessel':        3667,
    u'Mole':                2791,
    u'Malignant lesion':    1304,
    u'Fibroma':             593,
    u'Makeup':              406,
}


def main():
    # Make sure that all the classes of ODSI-DB are present in the dic
    dataloader_classes = [dl.OdsiDbDataLoader.OdsiDbDataset.classnames[k] \
        for k in dl.OdsiDbDataLoader.OdsiDbDataset.classnames]
    for k in dataloader_classes:
        if k not in pixels_per_class: 
            raise ValueError('[ERROR] The number of pixels of the class ' \
                             + k + ' is unknown.')

    # Make sure that the classes present in the dic are in ODSI-DB
    for k in pixels_per_class:
        if k not in dataloader_classes:
            raise ValueError('[ERROR] The class ' + k + ' is not known' \
                             + ' to the dataloader.')

    # Count the total number of pixels
    total = 0
    for k, v in pixels_per_class.items():
        total += v

    # Compute 1 - weight of each of the classes
    weights = {k: 1 - (float(v) / total) for k, v in pixels_per_class.items()}

    # Print result in Python format
    print('class_weights = {')
    for k, v in weights.items(): 
        print("    '" + k + "': " + "{:.9f}".format(v) + ',')
    print('}')

    # Print weights using class indices
    inverted = {v: k for k, v in dl.OdsiDbDataLoader.OdsiDbDataset.classnames.items()}
    print('class_weights = {')
    for k, v in weights.items(): 
        number = inverted[k]
        print("    '" + "{}".format(number) + "': " + "{:.9f}".format(v) + ',')
    print('}')

    # Print weights as a vector
    vector = [weights[dl.OdsiDbDataLoader.OdsiDbDataset.classnames[k]] for k in range(35)]
    print('class_weights = np.array(', vector, ')')
        

if __name__ == '__main__':
    main()
