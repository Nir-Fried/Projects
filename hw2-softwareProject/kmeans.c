#define PY_SSIZER_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <math.h>

int i=0;
int j=0;
int iter =0;
int x=0;
int i1=0;
int t1=0;
int flag = 1;
int a = 0;
int b = 0;
int c = 0;
double sum =0;
double min=0;
double norm =0;
int counter1 = 0;
int z=0;
int z1=0;
int minIndex = 0;
int rip = 0;
int temp=0;
int count=0;
char line[256];

int K;
int d;
int N;
int max_iter;
double epsilon;

PyObject *int_list;
int obser_length;
int *obser;

PyObject *double_list;
int data_length;
double *data;

int* numInCluster;
double* centroids;
double* centroidsHolder;
double **clusters;


static int fit(int K,int d,int N,int max_iter,double epsilon,int observations[],double dataPoints[])
{
    numInCluster= realloc(numInCluster,K*sizeof(int));
    centroids = realloc(centroids,d*K * sizeof(double));
    centroidsHolder=realloc(centroidsHolder,d*K*sizeof(double));
    for(i=0;i<K;i++)
    {
        clusters = (double **) realloc(clusters,K*sizeof(*clusters));
        clusters[i] = (double *)malloc(N*d*sizeof(double));
    }

    for (i=0;i<K;i++) /* init numInCluster */
    {
        numInCluster[i] = 0;
    }
    for (i=0;i<K;i++) /* init clusters */
    {
        for(j=0;j<N*d;j++)
        {
            clusters[i][j]=0;
        }
    }
    for (i=0;i<K;i++)
    {
        for(j=0;j<d;j++)
        {
            centroids[(i*d)+j] = dataPoints[(observations[i]*d)+j];
        }
    }
        /* repeat: */
    while(flag==1 && iter<max_iter)
    {
        for (a=0;a<K;a++)
        {
            for(b=0;b<N*d;b++)
            {
                clusters[a][b]=0;
            }
        }
        for (x=0;x<K;x++) /*reset num in each cluster*/
        {
            numInCluster[x]=0;
        }
        for (i1=0; i1<N*d; i1=i1+d)
        {
            min = 10000000;
            for(t1=0;t1<K*d;t1=t1+d)
            {
                sum =0;
                for (z1=0;z1<d;z1++)
                {
                    sum = sum + pow(dataPoints[i1+z1]-centroids[t1+z1],2); /* (x_i-mu_j)^2*/
                }
                if (sum<min)
                {
                    min = sum;
                    minIndex = t1/d; /* d =/= 0 */
                }
            }
            for (z1=0;z1<d;z1++)
            {
                temp = numInCluster[minIndex];
                clusters[minIndex][temp] = dataPoints[i1+z1];
                numInCluster[minIndex]+=1;
            }
        }
        /* update the centroids */
        /* first we copy the current centroids to another array in order to check if new-old<epsilon) */
        for (c=0;c<K*d;c++)
        {
            centroidsHolder[c]=centroids[c];
        }
        /* calculate the new centroids*/
        rip=0;
        for (i1=0;i1<K;i1++)
        {   
            count=0;
            while(count<d) {
            sum=0;
            for(t1=count;t1<N*d;t1=t1+d)
            { 
                sum = sum + clusters[i1][t1];
            }
            if(numInCluster[i1] != 0)
            {
                centroids[rip+count] = sum/(numInCluster[i1]/d);
                count++;
            }
            else if(numInCluster[i1] == 0) /* if clsuter is empty*/
            {
                centroids[rip+count] = 0;
                count++;
            }
            }
            rip = rip +d;
        }

        /* now we want to check if ||new-old||<epsilon for every vectors in centroids[] */
        for(c=0;c<K*d;c++)
        {
            centroidsHolder[c]= fabs(centroidsHolder[c])-fabs(centroids[c]);
        }
        

        counter1=0;
        for(c=0;c<K*d;c=c+d)
        {
            sum=0;
            for(z=0;z<d;z++)
            {
                sum = sum + pow(centroidsHolder[c+z],2);
            }
            norm = sqrt(sum);
            if (norm < epsilon) {
                counter1++;
                if (counter1==K)
                {
                    flag=0;
                }
            }
        }

        iter++;
    } 
    for(i=0;i<K;i++)
    {
        for(j=0;j<d;j++)
        {
            printf("%0.4f",centroids[(i*d)+j]);
            if(j != (d-1))
            {
                printf(",");
            }
            else {
                printf("\n");
            }
        }
    }

    for (i = 0; i < K; i++)
        free(clusters[i]);
    free(clusters);
    free(centroids);
    free(centroidsHolder);
    free(numInCluster);
    free(observations);
    free(dataPoints);
    return 0;
}


static PyObject* kmeanspp(PyObject *self,PyObject *args)
{
    if(!PyArg_ParseTuple(args,"iiiidOO",&K,&d,&N,&max_iter,&epsilon,&int_list,&double_list))
        return NULL;
    
    obser_length = PyObject_Length(int_list);
    obser = (int *) malloc(sizeof(int *) * obser_length);
    for(i=0;i<obser_length;i++)
    {
        PyObject *item;
        item = PyList_GetItem(int_list,i);
        obser[i] = PyFloat_AsDouble(item);
    }

    data_length = PyObject_Length(double_list);
    data = (double *) malloc(sizeof(double *) *data_length);
    for(i=0;i<data_length;i++)
    {
        PyObject *item;
        item = PyList_GetItem(double_list,i);
        data[i] = PyFloat_AsDouble(item);
    }
    
    return Py_BuildValue("i",fit(K,d,N,max_iter,epsilon,obser,data));
}

static PyMethodDef kmeansMethods[] = {
    {"fit",
        (PyCFunction) kmeanspp,
        METH_VARARGS,
        PyDoc_STR("test")},
    {NULL,NULL,0,NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp",
    NULL,
    -1,
    kmeansMethods
};

PyMODINIT_FUNC PyInit_mykmeanssp(void)
{
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    return m;
}
