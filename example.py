#!/usr/bin/env python

import numpy as np
import colorednoise as cn

def main():
    # Qd from allan deviation sigma
    #     WPM: sigma**2 x 1/3
    #     WFM: sigma**2
    #     FF: sigma**2 x (pi/(2 x log(2))

    # Example:
    nr = 2**22
    # generate white phase noise (WPM)
    X1 = cn.noiseGen(nr, 2.0e-11**2*1/3, 0)
    # generate white frequency noise (WFM)
    X2 = cn.noiseGen(nr, 3.5e-13**2, -2)
    # generate flicker floor noise (FF)
    X3 = cn.noiseGen(nr, 1.0e-16**2*(np.pi/(2*np.log(2))), -3)

    with open("noise.txt", 'w') as f:
        f.writelines( "%s\n" % item for item in X1+X2+X3)

if __name__=="__main__":
    main()
