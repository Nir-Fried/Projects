#define PY_SSIZER_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <math.h>

#include "spkmeans.h"

double sum =0;
double norm = 0;
double epsilon = 1/100000;
int iter = 0;
double max =0;
int maxI = 0;
int maxJ=0;
int sign = 0;

int index = 0;
int i=0;
int j=0;
int z = 0;
int r = 0;
double c = 0;
double phi = 0;
int k = 0;
int flag = 1;

double offA = 0;
double offAtag = 0;
double s;
double t;
int K;
int d;
int N;
char * goal;

PyObject *int_list;
int obser_length;
int *obser;

PyObject *double_list;
int data_length;
double *data;

double ** W;
double ** D;
double ** Dtag;
double ** I;

double ** mult1;
double ** mult2;
double ** lNorm;

double ** A;
double ** P;
double ** Ptranspose;
double ** Atag;
double ** V;
double ** Vtag;

double* eigenvalues;
double temp =0;
double * delta;

static int wam(int K,int d,int N,int observations[],double dataPoints[],char* goal)
{
    printf("WAM!\n");
    printf("K=%d,d=%d,N=%d\n",K,d,N);
    printf("\n");
    for (i=0;i<N*d;i=i+d)
    {
        for(j=0;j<d;j++)
        {
             printf("%0.4f,",dataPoints[i+j]);
        }
        printf("\n");
    }
    
    W = (double **) realloc(W,N*sizeof(*W)); //Create nxn matrix
    for(i=0;i<N;i++)
    {
        W[i] = (double *)malloc(N*sizeof(double));
    }

    for(i=0;i<N;i++)
    {
        for(j=i+1;j<N;j++)
        {
            sum = 0;
            for(z=0;z<d;z++)
            {
                sum = sum + pow((dataPoints[(i*d)+z]-dataPoints[(j*d)+z]),2);
            }
            norm = sqrt(sum);
            norm = norm * (-1);
            norm = norm/2;
            W[i][j]= exp(norm);
            W[j][i]= exp(norm);
        }
    }
    printf(" W is \n");
    for(i = 0; i < N; i++)
  	{
  		for(j = 0; j < N; j++)
  		{
            if (j != N-1)
  			    printf("%0.4f,", W[i][j]);
            else
                printf("%0.4f", W[i][j]);
		}
   		printf("\n");
  	}  	
    if (strcmp(goal,"wam") != 0) //if goal != wam we move on
    {
        return ddg(K,d,N,observations,dataPoints,goal,W);
    }
    else { //if goal == wam we print W
        printf(" W is \n");
        for(i = 0; i < N; i++)
        {
            for(j = 0; j < N; j++)
            {
                if (j != N-1)
                    printf("%0.4f,", W[i][j]);
                else
                    printf("%0.4f", W[i][j]);
            }
            printf("\n");
        }  	
    }
    for (i = 0; i < N; i++)
        free(W[i]);
    free(W);
    return 0;
}
static int ddg(int K,int d,int N,int observations[],double dataPoints[],char* goal,double ** W)
{
    printf("DDG!!!\n");

    D = (double **) realloc(D,N*sizeof(*D)); //Create nxn matrix
    for(i=0;i<N;i++)
    {
        D[i] = (double *)malloc(N*sizeof(double));
    }
    
    for(i=0;i<N;i++)
    {
        sum = 0;
        for(j=0;j<N;j++)
        {
            sum = sum + W[i][j];
        }
        D[i][i] = sum;
    }

    for(i = 0; i < N; i++)
  	{
  		for(j = 0; j < N; j++)
  		{
            if (j != N-1)
  			    printf("%0.4f,", D[i][j]);
            else
                printf("%0.4f", D[i][j]);
		}
   		printf("\n");
  	}  	
    
    if (strcmp(goal,"ddg") != 0) //if goal != ddg we move on
    {
        return lnorm(K,d,N,observations,dataPoints,goal,W,D);
    }
    else { //if goal == ddg we print D
        for(i = 0; i < N; i++)
        {
            for(j = 0; j < N; j++)
            {
                if (j != N-1)
                    printf("%0.4f,", D[i][j]);
                else
                    printf("%0.4f", D[i][j]);
            }
            printf("\n");
        }  	
    }
    for (i = 0; i < N; i++)
        free(D[i]);
    free(D);
    return 0;
}
static int lnorm(int K,int d,int N,int observations[],double dataPoints[],char* goal,double ** W,double ** D)
{
    printf("LNORM\n");

    Dtag = (double **) realloc(Dtag,N*sizeof(*Dtag)); //Create nxn matrix
    for(i=0;i<N;i++)
    {
        Dtag[i] = (double *)malloc(N*sizeof(double));
    }

    for(i=0;i<N;i++)
    {
        Dtag[i][i] = 1/sqrt(D[i][i]);
    }
    for(i = 0; i < N; i++)
  	{
  		for(j = 0; j < N; j++)
  		{
            if (j != N-1)
  			    printf("%0.4f,", Dtag[i][j]);
            else
                printf("%0.4f", Dtag[i][j]);
		}
   		printf("\n");
  	}  	


    I = (double **) realloc(I,N*sizeof(*I)); //Create nxn indentity matrix
    for(i=0;i<N;i++)
    {
        I[i] = (double *)malloc(N*sizeof(double));
    }
    for(i=0;i<N;i++)
    {
        I[i][i] = 1;
    }


    mult1 = (double **) realloc(mult1,N*sizeof(*mult1)); //Create nxn matrix
    for(i=0;i<N;i++)
    {
        mult1[i] = (double *)malloc(N*sizeof(double));
    }
    mult2 = (double **) realloc(mult2,N*sizeof(*mult2)); //Create nxn matrix
    for(i=0;i<N;i++)
    {
        mult2[i] = (double *)malloc(N*sizeof(double));
    }
    lNorm = (double **) realloc(lNorm,N*sizeof(*lNorm)); //Create nxn matrix
    for(i=0;i<N;i++)
    {
        lNorm[i] = (double *)malloc(N*sizeof(double));
    }
    for(i=0;i<N;i++) //first mult
    {
        for(j=0;j<N;j++)
        {
            mult1[i][j]=0;
            for(k=0;k<N;k++)
            {
                mult1[i][j]+= Dtag[i][k]*W[k][j]; //Dtag*W
            }
        }
    }
    
    for(i=0;i<N;i++) //second mult
    {
        for(j=0;j<N;j++)
        {
            mult2[i][j]=0;
            for(k=0;k<N;k++)
            {
                mult2[i][j]+= mult1[i][k]*Dtag[k][j]; //(Dtag*W)*Dtag
            }
        }
    }

    for(i=0;i<N;i++) //I-Dtag*W*dTag
    {
        for(j=0;j<N;j++)
        {
            lNorm[i][j]=I[i][j]-mult2[i][j];
        }
    }
    if (strcmp(goal,"lnorm") != 0) //if goal != lnorm we move on
    {
        return jacobi(K,d,N,observations,dataPoints,goal,lNorm);
    }
    else { //else we print lNorm
        printf("final LNORM\n");
        for(i = 0; i < N; i++)
        {
            for(j = 0; j < N; j++)
            {
                if (j != N-1)
                    printf("%0.4f,", lNorm[i][j]);
                else
                    printf("%0.4f", lNorm[i][j]);
            }
            printf("\n");
        }  	
    }
    for (i = 0; i < N; i++) {
        free(Dtag[i]);
        free(I[i]);
        free(mult1[i]);
        free(mult2[i]);
        free(mult1[i]);
        free(lNorm[i]);
    }
    free(Dtag);
    free(I);
    free(mult1);
    free(mult2);
    free(lNorm);
    return 0;
}
static int jacobi(int K,int d,int N,int observations[],double dataPoints[],char* goal,double ** lNorm)
{
    printf("JACOBIII\n");

    A = (double **) realloc(A,N*sizeof(*A)); //Create nxn matrix
    for(i=0;i<N;i++)
    {
        A[i] = (double *)malloc(N*sizeof(double));
    }
    if (strcmp(goal,"jacobi") != 0) //if goal != jacobi we operate on lNorm
    {
        for(i=0;i<N;i++)
        {
            for(j=0;j<N;j++)
            {
                A[i][j] = lNorm[i][j];
            }
        }
    }
    else //if goal == jacobi then we have a symmetric matrix in the input file
    {
        index =0;
        for(i=0;i<N;i++)
        {
            for(j=0;j<N;j++)
            {
                A[i][j] = dataPoints[index];
                index++;
            }
        }
    }

    P = (double **) realloc(P,N*sizeof(*P)); //Create nxn matrix
    for(i=0;i<N;i++)
    {
        P[i] = (double *)malloc(N*sizeof(double));
    }
    Ptranspose = (double **) realloc(Ptranspose,N*sizeof(*Ptranspose)); //Create nxn matrix
    for(i=0;i<N;i++)
    {
        Ptranspose[i] = (double *)malloc(N*sizeof(double));
    }
    Atag = (double **) realloc(Atag,N*sizeof(*Atag)); //Create nxn matrix
    for(i=0;i<N;i++)
    {
        Atag[i] = (double *)malloc(N*sizeof(double));
    }
    V = (double **) realloc(V,N*sizeof(*V)); //Create nxn matrix
    for(i=0;i<N;i++)
    {
        V[i] = (double *)malloc(N*sizeof(double));
    }
    Vtag = (double **) realloc(Vtag,N*sizeof(*Vtag)); //Create nxn matrix
    for(i=0;i<N;i++)
    {
        Vtag[i] = (double *)malloc(N*sizeof(double));
    }

    iter = 0;
    while(flag==1 && iter<101)
    {
        max = -1;
        for(i=0;i<N;i++) //Find the largest absolute off diagonal value
        {
            for(j=i+1;j<N;j++)
            {
                if(fabs(A[i][j])>max)
                {
                    max = fabs(A[i][j]);
                    maxI = i;
                    maxJ = j;
                }
            }
        }
        //Calculate c,s:
        phi = (A[maxJ][maxJ]-A[maxI][maxI])/(2*A[maxI][maxJ]);

        sign = 0;
        if(phi>=0)
        {
            sign = 1;
        }
        else {
            sign = -1;
        }

        t = sign/(fabs(phi)+sqrt(pow(phi,2)+1));
        c = 1/(sqrt(pow(t,2)+1));
        s = t*c;

        for(i=0;i<N;i++) // Create P first by constructing the indentity matrix
        {
            for(j=0;j<N;j++)
            {
                if(i==j){
                    P[i][j]=1;
                }
                else{
                    P[i][j]=0;
                }
            }
        }
        P[maxI][maxI] = c;
        P[maxJ][maxJ] = c;
        P[maxI][maxJ] = s;
        P[maxJ][maxI] = (-1)*s;

        //calculate V:
        if(iter==0) //if this is the first iteration, we set V=P
        {
            for(i = 0; i < N; i++)
            {
                for(j = 0; j < N; j++)
                {
                    V[i][j] = P[i][j];
                }
            }
        }
        else //iter>1 so we want to mult by the new P from the right
        {
            for(i=0;i<N;i++) //first mult
            {
                for(j=0;j<N;j++)
                {
                    Vtag[i][j]=0;
                    for(k=0;k<N;k++)
                    {
                        Vtag[i][j]+= V[i][k]*P[k][j]; //P^t*A
                    }
                }
            }
            //Now we set V= V'
            for(i=0;i<N;i++) //first mult
            {
                for(j=0;j<N;j++)
                {
                    V[i][j] = Vtag[i][j];
                }
            }
        }

        for(i = 0; i < N; i++) //transpose P
        {
            for(j = 0; j < N; j++)
            {
                Ptranspose[i][j] = P[j][i];
            }
        }

        for(i=0;i<N;i++) //first mult
        {
            for(j=0;j<N;j++)
            {
                mult1[i][j]=0;
                for(k=0;k<N;k++)
                {
                    mult1[i][j]+= Ptranspose[i][k]*A[k][j]; //P^t*A
                }
            }
        }
        for(i=0;i<N;i++) //second mult
        {
            for(j=0;j<N;j++)
            {
                Atag[i][j]=0;
                for(k=0;k<N;k++)
                {
                    Atag[i][j]+= mult1[i][k]*P[k][j]; //(P^t*A)*P
                }
            }
        }

        //Calculate OFF(A)^2 and OFF(A')^2
        offA = 0;
        offAtag = 0;
        for(i=0;i<N;i++)
        {
            for(j=0;j<N;j++)
            {
                if(i != j)
                {
                    offA+= pow(A[i][j],2);
                    offAtag+= pow(Atag[i][j],2);
                }
            }
        }
        if(fabs(offA-offAtag)<=epsilon) //if off(A)-off(A') <= epsilon
        {
            flag = 0;
        }

        //Now we want to set A=A'
        for(i=0;i<N;i++)
        {
            for(j=0;j<N;j++)
            {
                A[i][j] = Atag[i][j];
            }
        }
        iter++;
    }

    eigenvalues= realloc(eigenvalues,N*sizeof(double));
    for(i=0;i<N;i++)
    {
        eigenvalues[i] = A[i][i];
    }

    if (strcmp(goal,"jacobi") != 0) //if goal != jacobi we move on
    {
        return eigengap(K,d,N,observations,eigenvalues,V);
    }
    else { //if goal == jacobi we print eigenvalues+eigenvectors
        printf("eigenvalues are:\n");
        for(i=0;i<N;i++)
        {   
            if(i != N-1)
                printf("%0.4f,",eigenvalues[i]);
            else
                printf("%0.4f",eigenvalues[i]);
        }
        printf("\n");
        printf("V is \n");
        for(i = 0; i < N; i++)
        {
            for(j = 0; j < N; j++)
            {
                if (j != N-1)
                    printf("%0.4f,", V[j][i]);
                else
                    printf("%0.4f", V[j][i]);
            }
            printf("\n");
        }  	

    }
    for (i = 0; i < N; i++) {
        free(A[i]);
        free(P[i]);
        free(Ptranspose[i]);
        free(Atag[i]);
        free(V[i]);
        free(Vtag[i]);
    }
    free(A);
    free(P);
    free(Ptranspose);
    free(Atag);
    free(V);
    free(Vtag);
    free(eigenvalues);
    return 0;
}
static int eigengap(int K,int d,int N,int observations[],double eigenvalues[],double ** V)
{
    printf("EIGENGAPPPPP\n");
    //Sort eigenvalues in decreasing order
    for(i=0;i<N;i++)
    {
        max = eigenvalues[i];
        maxI = i;
        for(j=i+1;j<N;j++)
        {
            if(eigenvalues[j]>max)
            {
                max = eigenvalues[j];
                maxI = j;
            }
        }
        temp = eigenvalues[i];
        eigenvalues[i] = max;
        eigenvalues[maxI] = temp;
        //We also want to sort the eigenvectors in the same order:
        for(j=0;j<N;j++)
        {
            temp = V[j][i];
            V[j][i] = V[j][maxI];
            V[j][maxI] = temp;
        }
    }
    printf("after sort!!\n");
    printf("eigenvalues are:\n");
    for(i=0;i<N;i++)
    {
        printf("%0.4f \n",eigenvalues[i]);
    }
    printf("V is \n");
    for(i = 0; i < N; i++)
    {
        for(j = 0; j < N; j++)
        {
            if (j != N-1)
                printf("%0.4f,", V[i][j]);
            else
                printf("%0.4f", V[i][j]);
        }
        printf("\n");
    }  	

    if(K==0) //if we need to determine K
    {
        delta = realloc(delta,(N-1)*sizeof(double));
        for(i=0;i<N-1;i++)
        {
            delta[i] = fabs(eigenvalues[i]-eigenvalues[i+1]);
        }
        printf("\n");
        for(i=0;i<N-1;i++)
        {
            printf("%0.4f ",delta[i]);
        }
        max=0;
        for(i=1;i<floor(N/2);i++)
        {
            if (delta[i]>max) {
                max = delta[i];
                K = i+1;
            }
        }
        printf("We got K = %d\n",K);
        return K;
    }


    spk(K,d,N,observations,V);
    return 0;
}

static int spk(int K,int d,int N,int observations[],double ** V)
{
    return 0; //here
}

static PyObject* spkmeans(PyObject *self,PyObject *args)
{
    if(!PyArg_ParseTuple(args,"iiiOOs",&K,&d,&N,&int_list,&double_list,&goal))
        return NULL;

    obser_length = (int) PyObject_Length(int_list);
    obser = (int *) malloc(sizeof(int *) * obser_length);
    for(i=0;i<obser_length;i++)
    {
        PyObject *item;
        item = PyList_GetItem(int_list,i);
        obser[i] = (int) PyFloat_AsDouble(item);
    }

    data_length = (int) PyObject_Length(double_list);
    data = (double *) malloc(sizeof(double *) *data_length);
    for(i=0;i<data_length;i++)
    {
        PyObject *item;
        item = PyList_GetItem(double_list,i);
        data[i] = PyFloat_AsDouble(item);
    }
    return Py_BuildValue("i",wam(K,d,N,obser,data,goal));
    
}
static PyMethodDef spkmeansMethods[] = {
    {"wam",
        (PyCFunction) spkmeans,
        METH_VARARGS,
        PyDoc_STR("test")},
    {NULL,NULL,0,NULL}
};
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "spkmeans",
    NULL,
    -1,
    spkmeansMethods
};
PyMODINIT_FUNC PyInit_spkmeans(void)
{
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    return m;
}