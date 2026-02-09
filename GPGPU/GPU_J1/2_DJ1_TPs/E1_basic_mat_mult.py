

"""
Exercice 1 : MATRIX MULTIPLICATION WITH PARALLELIZED COMPUTATIONS IN A SINGLE BLOC

THE GOAL OF THIS FIRST EXERCICE IS TO GET FAMILIAR WITH THE BASIC CONCEPTS OF GPU COMPUTING
WITH CUDA. NOTE THAT THE ALGORITHMS WON'T BE VERY EFFICIENT HERE. EFFICIENT STRATEGIES
WILL START BEING STUDIED IN EXERCICE 2.

-> Remark 1: Code adapted from: https://shephexd.github.io/development/2017/02/19/pycuda.html

-> Remark 2:  If using google colab:
    * Click on Runtime (excecution) and select Change runtime type (modifier le type d'excecution).
    * Then select GPU in Hardware Acceleration (accélérateur matériel)
    * Start your session by installing pycuda with the command: \" !pip install pycuda \"

"""

#+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
#1) INITIALISATION
#+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +

import numpy as np
import pycuda
from pycuda import driver, compiler, gpuarray, tools
import time

# -- initialize the device
import pycuda.autoinit


#get device information
MyDevice=pycuda.driver.Device(0)
attributes=MyDevice.get_attributes()

for attribute in list(attributes.keys()):
  print(str(attribute)+": "+str(attributes[attribute]))

#+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
#2) MATRIX MULTIPLICATION ON THE CPU
#+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +

# define the (square) matrix size
#  note that we'll only use *one* block of threads here
#  as a consequence this number (squared) can't exceed max_threads
# -> use MyDevice.get_attributes() to get this information
MATRIX_SIZE = 32

# create two random square matrices
a_cpu = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
b_cpu = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)

# compute reference on the CPU to verify GPU computation
time_start=time.time()
c_cpu = np.dot(a_cpu, b_cpu)
time_end=time.time()
print('enlapsed time (CPU):',time_end-time_start,' seconds')



#+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
#3) MAKE AND COMPILE THE KERNEL FOR MATRIX MULTIPLICATION ON THE GPU
#+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +

#define the kernel (for the meaning of %()s, see: https://stackoverflow.com/questions/63862118/what-is-the-meaning-of-s-in-python)
kernel_code_templates = """
__global__ void MatrixMulKernel_sequential(float *a, float *b, float *c)
{
    for (int i = 0; i < %(MATRIX_SIZE)s; ++i) {
      for (int j = 0; j < %(MATRIX_SIZE)s; ++j) {
        c[i * %(MATRIX_SIZE)s + j] = 0.;
        for (int k = 0; k < %(MATRIX_SIZE)s; ++k) {
          c[i * %(MATRIX_SIZE)s + j] += a[i * %(MATRIX_SIZE)s + k] * b[k * %(MATRIX_SIZE)s + j];
        }
      }
    }
}
__global__ void MatrixMulKernel_sequential2(float *a, float *b, float *c)
{
    float c_value;
    
    for (int i = 0; i < %(MATRIX_SIZE)s; ++i) {
      for (int j = 0; j < %(MATRIX_SIZE)s; ++j) {
        c_value = 0.;
        for (int k = 0; k < %(MATRIX_SIZE)s; ++k) {
          c_value += a[i * %(MATRIX_SIZE)s + k] * b[k * %(MATRIX_SIZE)s + j];
        }
        c[i * %(MATRIX_SIZE)s + j]=c_value;
      }
    }
}
__global__ void MatrixMulKernel_parallel(float *a, float *b, float *c)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Pvalue is used to store the element of the matrix
    // that is computed by the thread
    float Pvalue = 0;

    // Each thread loads one row of M and one column of N,
    //   to produce one element of P.
    for (int k = 0; k < %(MATRIX_SIZE)s; ++k) {
        Pvalue += a[ty * %(MATRIX_SIZE)s + k] * b[k * %(MATRIX_SIZE)s + tx];
    }

    // Write the matrix to device memory;
    // each thread writes one element
    c[ty * %(MATRIX_SIZE)s + tx] = Pvalue;
}
"""

# get the kernel code from the template
# by specifying the constant MATRIX_SIZE
kernel_codes = kernel_code_templates % {
    'MATRIX_SIZE': MATRIX_SIZE
    }

# compile the kernel code
mod = compiler.SourceModule(kernel_codes)

# get the kernel function from the compiled module
matrixmul_sequential = mod.get_function("MatrixMulKernel_sequential") 
matrixmul_sequential2 = mod.get_function("MatrixMulKernel_sequential2")
matrixmul_parallel = mod.get_function("MatrixMulKernel_parallel") 

#+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
#4) MATRIX MULTIPLICATION ON THE GPU
#+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +


# transfer host (CPU) memory to device (GPU) memory
a_gpu = gpuarray.to_gpu(a_cpu)
b_gpu = gpuarray.to_gpu(b_cpu)

# create empty gpu array for the result (C = A * B)
c_gpu = gpuarray.empty((MATRIX_SIZE, MATRIX_SIZE), np.float32)



# call the kernel on the card -- sequential matrix multiplication 
time_start=time.time()
matrixmul_sequential(a_gpu , b_gpu , c_gpu , block = (1, 1, 1))
time_end=time.time()

print('enlapsed time (GPU - sequential):',time_end-time_start,' seconds')
print('norm(c_cpu-c_gpu)='+str(np.linalg.norm(c_cpu - c_gpu.get())))

# call the kernel on the card -- sequential (slighly improved) matrix multiplication 
time_start=time.time()
matrixmul_sequential2(a_gpu , b_gpu , c_gpu , block = (1, 1, 1))
time_end=time.time()

print('enlapsed time (GPU - sequential 2):',time_end-time_start,' seconds')
print('norm(c_cpu-c_gpu)='+str(np.linalg.norm(c_cpu - c_gpu.get())))


# call the kernel on the card
time_start=time.time()
matrixmul_parallel(a_gpu, b_gpu, c_gpu, block = (MATRIX_SIZE, MATRIX_SIZE, 1))
time_end=time.time()
print('enlapsed time (GPU - parallel):',time_end-time_start,' seconds')
print('norm(c_cpu-c_gpu)='+str(np.linalg.norm(c_cpu - c_gpu.get())))




"""
QUESTION 1: Comprenez bien chaque partie du code en vous attardant sur :
            1.1 -> les commandes utilisees pour ecrire, compiler et executer
                   les differents noyaux (kernels)
            1.2 -> le lien entre les algorithmes implémentés dans les 
                   differents noyaux utilises et la taille des blocs 
                   lors de leur appel. 

QUESTION 2: Comparez les temps de calucul entre la multiplication CPU et
            celles GPU (relancez plusieurs fois les calculs car les temps
            d'execution peuvent etres très variables d'une fois sur l'autre en
            fonction de l'utilisation du PC/serveur. Gardez alors les temps les
            plus courts). Il est tout a fait possible que les gains ne 
            soient pas bons (ce sera moins le cas dans les exercices 2 et 3
            avec des algorithmes plus adaptes au GPU). Pourquoi ?
             
QUESTION 3: Comparez les temps de calculs entre la methode 'sequential' et 
            celle 'sequential 2'. Pourquoi la seconde est un peu plus rapide ?
             
QUESTION 4: Comparez les temps de calculs entre la methode 'sequential' et 
            celle 'parallel'. Pourquoi les gains sont si faibles (voir 
            negatifs) alors que les calculs on ete parallelises sur 1024 noeuds ?
             
QUESTION 5: Il est très clair que les calculs parallélisés sur chaque noeud
            sont trop court avec une matrice de taille 32x32 par rapport au 
            temps perdu a deplacer l'information et a ordonancer la 
            parallelisation. Relancez alors les calculs avec une taille de 
            matrice plus grande (disons 128x128). Pourquoi cela ne marche pas ?
            La réponse va en tous cas nous motiver a utiliser dans l'exercice 2
            une grille de blocs et non un seul bloc... ce qui va rendre les 
            gains de temps bien plus interessants !
"""
