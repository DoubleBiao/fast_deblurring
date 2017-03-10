# fast_deblurring
implement fast deblurring algorithm in C++ OpenCV and FFTW

This code implements the algorithm described in the paper :
paper D. Krishnan, R. Fergus: "Fast Image Deconvolution using Hyper-Laplacian Priors", Proceedings of NIPS 2009.

OpenCV 2.4.8 and FFTW3 is used.

The algorithm is further simplified in this program by replace the LUT with linear regression functions.

This program is developed in VS 2012 and, to compile it correctly, the library path in the configuration files FFTW.proprs. OpenCV_REALEASEconfig.props and OpenCVconfig.props should be modified.

Sorry for the possible inconvenience of comments in Chinese. If there is any question, feel free to contact me by e-mail.
My address:
648194212@qq.com or 213131603@seu.edu.cn
