#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <math.h>

char *file_name;
char *goal;
FILE *file = NULL;

FILE *inputFile;
int d;
int N;

double* dataPoints;
int len=0;
char line[256];
char*token = NULL;
char *eptr;

int i;
int j;
int z;
int k;

double norm;
double sum;
double ** W;
double ** D;
double ** Dtag;
double ** I;
double ** mult1;
double ** mult2;
double ** lNorm;

double ** A;
int indexx;
double ** P;
double ** Ptranspose;
double ** Atag;
double ** V;
double ** Vtag;
int iter;
int flag;
double max =0;
int maxI = 0;
int maxJ=0;
int sign = 0;
double c = 0;
double phi = 0;
double s;
double t;
double offA = 0;
double offAtag = 0;
double epsilon = 1/100000;
double* eigenvalues;


int jacobi(int N,double dataPoints[]) /* there are NxN datapoints */
{
    A = (double **) realloc(A,N*sizeof(*A)); /*Create nxn matrix */
    for(i=0;i<N;i++)
    {
        A[i] = (double *)malloc(N*sizeof(double));
    }
    printf("using input for eigen\n");
    indexx =0;
    for(i=0;i<N;i++)
    {
        for(j=0;j<N;j++)
        {
            A[i][j] = dataPoints[indexx];
            indexx++;
        }
    }
    P = (double **) realloc(P,N*sizeof(*P)); /*Create nxn matrix */
    for(i=0;i<N;i++)
    {
        P[i] = (double *)malloc(N*sizeof(double));
    }
    Ptranspose = (double **) realloc(Ptranspose,N*sizeof(*Ptranspose)); /*Create nxn matrix */
    for(i=0;i<N;i++)
    {
        Ptranspose[i] = (double *)malloc(N*sizeof(double));
    }
    Atag = (double **) realloc(Atag,N*sizeof(*Atag)); /*Create nxn matrix */
    for(i=0;i<N;i++)
    {
        Atag[i] = (double *)malloc(N*sizeof(double));
    }
    V = (double **) realloc(V,N*sizeof(*V)); /*Create nxn matrix */
    for(i=0;i<N;i++)
    {
        V[i] = (double *)malloc(N*sizeof(double));
    }
    Vtag = (double **) realloc(Vtag,N*sizeof(*Vtag)); /*Create nxn matrix */
    for(i=0;i<N;i++)
    {
        Vtag[i] = (double *)malloc(N*sizeof(double));
    }
    mult1 = (double **) realloc(mult1,N*sizeof(*mult1)); /*Create nxn matrix */
    for(i=0;i<N;i++)
    {
        mult1[i] = (double *)malloc(N*sizeof(double));
    }
    iter = 0;
    flag = 1;
    while(flag==1 && iter<101)
    {
        max = -1;
        for(i=0;i<N;i++) /*Find the largest absolute off diagonal value */
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
        /* Calculate c,s: */
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
        for(i=0;i<N;i++) /* Create P first by constructing the indentity matrix */
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
        /*calculate V: */
        if(iter==0) /*if this is the first iteration, we set V=P */
        {
            for(i = 0; i < N; i++)
            {
                for(j = 0; j < N; j++)
                {
                    V[i][j] = P[i][j];
                }
            }
        }
        else /* iter>1 so we want to mult by the new P from the right */
        { 
            for(i=0;i<N;i++) 
            {
                for(j=0;j<N;j++)
                {
                    Vtag[i][j]=0;
                    for(k=0;k<N;k++)
                    {
                        Vtag[i][j]+= V[i][k]*P[k][j]; /* V' = V*P */
                    }
                }
            }
            /*Now we set V= V' */
            for(i=0;i<N;i++) 
            {
                for(j=0;j<N;j++)
                {
                    V[i][j] = Vtag[i][j];
                }
            }
        }
        for(i = 0; i < N; i++) /* transpose P */
        {
            for(j = 0; j < N; j++)
            {
                Ptranspose[i][j] = P[j][i];
            }
        }
 
        for(i=0;i<N;i++) /* first mult */
        {
            for(j=0;j<N;j++)
            {
                mult1[i][j]=0;
                for(k=0;k<N;k++)
                {
                    mult1[i][j]+= Ptranspose[i][k]*A[k][j]; /* P^t*A */
                }
            }
        }
        for(i=0;i<N;i++) /* second mult */
        {
            for(j=0;j<N;j++)
            {
                Atag[i][j]=0;
                for(k=0;k<N;k++)
                {
                    Atag[i][j]+= mult1[i][k]*P[k][j]; /* (P^t*A)*P */
                }
            }
        }
        /* Calculate OFF(A)^2 and OFF(A')^2 */
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
        if(fabs(offA-offAtag)<=epsilon) /* if off(A)-off(A') <= epsilon */
        {
            flag = 0;
        }

        /* Now we want to set A=A' */
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

    printf("eigenvalues are:\n");
    for(i=0;i<N;i++)
    {   
        if(i != N-1)
            printf("%0.4f,",eigenvalues[i]);
        else
            printf("%0.4f",eigenvalues[i]);
    }
    printf("\n");
    printf("V is (printed as columns\n");
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

int lnorm(int N,double ** W,double ** D) 
{
    printf("LNORM\n");

    Dtag = (double **) realloc(Dtag,N*sizeof(*Dtag)); /*Create nxn matrix */
    for(i=0;i<N;i++)
    {
        Dtag[i] = (double *)malloc(N*sizeof(double));
    }
    for(i=0;i<N;i++) /* init Dtag */
    {
        memset(Dtag[i],0,N*sizeof(double));
    } 

    for(i=0;i<N;i++)
    {
        if(D[i][i]>0)
        {
            Dtag[i][i] = 1/sqrt(D[i][i]);
        }
    }

    I = (double **) malloc(N*sizeof(*I)); /* Create nxn indentity matrix */
    for(i=0;i<N;i++)
    {
        I[i] = (double *)malloc(N*sizeof(double));
    }
    for(i=0;i<N;i++) /* init I */
    {
        memset(I[i],0,N*sizeof(double));
    } 

    for(i=0;i<N;i++)
    {
        I[i][i] = 1;
    }
    mult1 = (double **) realloc(mult1,N*sizeof(*mult1)); /*Create nxn matrix */
    for(i=0;i<N;i++)
    {
        mult1[i] = (double *)malloc(N*sizeof(double));
    }
    mult2 = (double **) realloc(mult2,N*sizeof(*mult2)); /*Create nxn matrix */
    for(i=0;i<N;i++)
    {
        mult2[i] = (double *)malloc(N*sizeof(double));
    }
    lNorm = (double **) realloc(lNorm,N*sizeof(*lNorm)); /*Create nxn matrix */
    for(i=0;i<N;i++)
    {
        lNorm[i] = (double *)malloc(N*sizeof(double));
    }
    for(i=0;i<N;i++) /*first mult */
    {
        for(j=0;j<N;j++)
        {
            mult1[i][j]=0;
            for(k=0;k<N;k++)
            {
                mult1[i][j]+= Dtag[i][k]*W[k][j]; /* Dtag*W */
            }
        }
    }
    
    for(i=0;i<N;i++) /* second mult */
    {
        for(j=0;j<N;j++)
        {
            mult2[i][j]=0;
            for(k=0;k<N;k++)
            {
                mult2[i][j]+= mult1[i][k]*Dtag[k][j]; /* (Dtag*W)*Dtag */
            }
        }
    }
    for(i=0;i<N;i++) /* I-Dtag*W*dTag */
    {
        for(j=0;j<N;j++)
        {
            lNorm[i][j]=I[i][j]-mult2[i][j];
        }
    }

    printf("final LNORM\n"); /* if we reached here, goal == lnorm, hence we print lNorm */
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
    
    for (i = 0; i < N; i++) {
        free(Dtag[i]);
        free(I[i]);
        free(mult1[i]);
        free(mult2[i]);
        free(lNorm[i]); 
    }
    free(Dtag);
    free(I);
    free(mult1);
    free(mult2);
    free(lNorm);
    
    return 0;
}

static int ddg(int N,char* goal,double ** W)
{
    printf("DDG!!!\n");

    D = (double **) realloc(D,N*sizeof(*D)); /* Create nxn matrix */
    for(i=0;i<N;i++)
    {
        D[i] = (double *)malloc(N*sizeof(double));
    }
    for(i=0;i<N;i++) /* init D */
    {
        memset(D[i],0,N*sizeof(double));
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
    if (strcmp(goal,"ddg") != 0) /* if goal != ddg we move on */
    {
        return lnorm(N,W,D);
    }
    else { /* if goal == ddg we print D */
        printf("D is \n");
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

static int wam(int d,int N,double dataPoints[],char* goal)
{
printf("CWAM!\n");
printf("d=%d,N=%d\n",d,N);
    printf("\n");
    for (i=0;i<N*d;i=i+d)
    {
        for(j=0;j<d;j++)
        {
            printf("%0.4f,",dataPoints[i+j]);
        }
        printf("\n");
    }
    
    W = (double **) realloc(W,N*sizeof(*W)); /* Create nxn matrix */
    for(i=0;i<N;i++)
    {
        W[i] = (double *)malloc(N*sizeof(double));
    }
    for(i=0;i<N;i++) /* init W */
    {
        memset(W[i],0,N*sizeof(double));
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
    if (strcmp(goal,"wam") != 0) /* if goal != wam we move on */
    {
        return ddg(N,goal,W); 
    }
    else { /* if goal == wam we print W */
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

int main(int argc,char * argv[])
{

    printf("--------------STARTC------------\n");
    if (argc != 3)
    {
        printf("Invalid Input!");
        return 1;
    }
    goal = malloc(sizeof(char)*(strlen(argv[1])+1));
    strcpy(goal,argv[1]);
    file_name = malloc(sizeof(char)*(strlen(argv[2])+1));
    strcpy(file_name,argv[2]);

    printf("%s,%s\n",goal,file_name);

    inputFile = fopen(file_name,"r");

    if(strcmp(goal,"wam") != 0 && strcmp(goal,"ddg") != 0 && strcmp(goal,"lnorm") != 0 && strcmp(goal,"jacobi") != 0)
    { /* invalid goal */
        printf("Invalid Input!");
        return 1;
    }

    if (inputFile == NULL) /* if file was not found */
    {
        printf("An Error Has Occurred");
        return 1;
    }
    
    d=1;
    fgets (line , 256 , inputFile);
    len = strlen(line);
    for (i=0;i<len;i++)
    {
        d += (line[i] == ','); /* calculating our d value */
    }   
    fclose(inputFile);

    N=0;
    inputFile = fopen(file_name,"r");
    
    while(fgets(line,256,inputFile)) /* calculating our N value */
    {
        line[strcspn(line,"\n")] =0; 
        N++;
    }
    fclose(inputFile);

    printf("d is %d , and N is %d\n",d,N);

    dataPoints = malloc(d*N*sizeof(double));
    inputFile = fopen(file_name,"r");
    i=0;
    while(fgets(line,256,inputFile))
    {
        token = strtok(line,",");
        while (token != NULL) {
            dataPoints[i] = strtod(token,&eptr); /* initializing dataPoints */
            i++;
            token = strtok(NULL, ",");
        }
    }
    fclose(inputFile);

    printf("dataPoints:\n");
    for(i=0;i<N*d;i++)
    {
        if((i+1) % d ==0)
        {
            printf("%0.4f\n",dataPoints[i]);
        }
        else {
            printf("%0.4f,",dataPoints[i]);
        }
    }
    if (strcmp(goal,"jacobi") == 0) /* if goal == jacobi we dont need to calculate anything else */
    {
        jacobi(N,dataPoints);
    }
    else { /*goal != jacobi */
        wam(d,N,dataPoints,goal);
    }

    printf("--------------ENDC------------!\n");
    return 0;
}